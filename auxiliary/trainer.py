import os
import torch
from tqdm import tqdm
import datetime as dt
import torchvision.utils as tvutils

from .utils import Logger, check_path
from auxiliary.metrics import FID


class Trainer:
    def __init__(self,
                 Gen, Dis, optim_G, optim_D, lr_sche_G=None, lr_sched_D=None, criterion=None,
                 dim_z=100, weight_clip_val=0.01, batch_size=32, total_epoch=200,
                 train_gen_interval=5, img_sample_interval=10, save_interval=10, log_interval=10,
                 dataloader=None, exp_name="", root_path="./runtime",
                 resume=False, device="cpu"):
        self.Gen = Gen
        self.Dis = Dis
        self.optim_G = optim_G
        self.optim_D = optim_D
        self.lr_sche_G = lr_sche_G
        self.lr_sche_D = lr_sched_D
        self.criterion = criterion
        self.dim_z = dim_z
        self.weight_clip_val = weight_clip_val
        self.batch_size = batch_size
        self.start_epoch = 0
        self.total_epoch = total_epoch
        self.train_gen_interval = train_gen_interval
        self.img_sample_interval = img_sample_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.dataloader = dataloader
        self.resume = resume
        self.device = device

        if exp_name == "":
            exp_name = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.exp_name = exp_name
        runtime_path = os.path.join(root_path, exp_name)
        check_path(runtime_path)
        self.runtime_path = runtime_path
        img_path = os.path.join(runtime_path, "sample_images")
        check_path(img_path)
        self.img_path = img_path
        ckpt_path = os.path.join(runtime_path, "checkpoints")
        check_path(ckpt_path)
        self.ckpt_path = ckpt_path
        log_path = os.path.join(runtime_path, "train.log")
        self.logger = Logger(log_path)

    def train(self):
        self.Gen.to(self.device)
        self.Dis.to(self.device)
        fid = .0
        for epoch in range(self.start_epoch + 1, self.total_epoch + 1):
            loss_sum_gen = .0
            loss_sum_dis = .0
            num_sample = 0

            total_iters = len(self.dataloader)
            with tqdm(total=total_iters,
                      desc="Epoch {}/{}".format(epoch, self.total_epoch),
                      postfix=dict) as pbar:
                sample_imgs = None
                for i, data in enumerate(self.dataloader):

                    imgs = data[0]
                    if epoch == 1 and i == 0:
                        tvutils.save_image(imgs.data, "{}/real_images_epoch_{}.png".format(self.img_path, epoch))
                    num_sample += imgs.shape[0]

                    lossDis, lossGen, fake_imgs = self.updater(i, imgs)
                    loss_sum_dis += lossDis.item()

                    if epoch % self.img_sample_interval == 0 and i == total_iters-2:
                        sample_imgs = fake_imgs

                    pbar.set_postfix(**{
                        'loss_D': loss_sum_dis / num_sample, 'lr_D': fetch_lr(self.optim_D),
                        'loss_G': loss_sum_gen / num_sample, 'lr_G': fetch_lr(self.optim_G),
                        'FID': fid,
                    })
                    pbar.update(1)

                if epoch % self.img_sample_interval == 0:
                    tvutils.save_image(sample_imgs.data,
                                       "{}/fake_images_epoch_{}.png".format(self.img_path, epoch),
                                       normalize=True)
                if epoch % self.save_interval == 0 or epoch == self.total_epoch:
                    print(epoch % self.save_interval)
                    ckpd_G = {
                        "model_dict": self.Gen.state_dict(),
                        "optimizer": self.optim_G.state_dict(),
                        "epoch": epoch
                    },
                    ckpd_D = {
                        "model_dict": self.Dis.state_dict(),
                        "optimizer": self.optim_D.state_dict(),
                        "epoch": epoch
                    },
                    torch.save(ckpd_G, "{}/ckpt_Gen_epoch{}.pt".format(self.ckpt_path, epoch))
                    torch.save(ckpd_D, "{}/ckpt_Dis_epoch{}.pt".format(self.ckpt_path, epoch))
                    self.logger.log("Epoch {} model saved at {}".format(epoch, self.ckpt_path))

                if epoch % self.log_interval == 0:
                    fid = FID(imgs, fake_imgs, device=self.device)
                    self.logger.log("Epoch {}, dis_loss={:.3f}, gen_loss={:.3f}, FID={:.3f}".
                                    format(epoch, loss_sum_dis, loss_sum_gen,fid))

    def updater(self, i, imgs):
        lossGen = torch.tensor(0).to(self.device)
        real_imgs = imgs.to(self.device)

        noise = torch.randn(self.batch_size, self.dim_z)
        noise = noise.to(self.device)

        self.optim_D.zero_grad()
        fake_imgs = self.Gen(noise).detach()
        score_real = self.Dis(real_imgs)
        score_fake = self.Dis(fake_imgs)

        lossDis = self.criterion(score_fake, score_real)
        lossDis.backward()
        self.optim_D.step()

        for param in self.Dis.parameters():
            param.data.clamp_(-self.weight_clip_val, self.weight_clip_val)

        if i % self.train_gen_interval == 0:
            fake_imgs = self.Gen(noise)
            score_fake = self.Dis(fake_imgs)

            self.optim_G.zero_grad()
            lossGen = self.criterion(score_fake)
            lossGen.backward()

            self.optim_G.step()

            if self.lr_sche_G is not None:
                self.lr_sche_G.step()

        if self.lr_sche_D is not None:
            self.lr_sche_D.step()

        return lossDis, lossGen, fake_imgs


def fetch_lr(optimizer: torch.optim):
    for param_group in optimizer.param_groups:
        return param_group['lr']

