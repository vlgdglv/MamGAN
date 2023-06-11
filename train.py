import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from discriminator.discriminator import Discriminator
from generator.generator import Generator
from discriminator.convdis import ConvDiscriminator
from generator.convgen import ConvGenerator

from auxiliary.losses import Hinge
from auxiliary.trainer import Trainer
from auxiliary.utils import init_model
from dataset.cifar10 import get_cifar10_dataloader, get_cifar10_fs_dataloader
from dataset.mnist import get_mnist_dataloader, get_mnist_fs_dataloader
from maml.maml import maml_init_cifar10, maml_init_MNIST


def set_seed(seed=2023):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


TARGET = 8


class MNISTTrainer(Trainer):
    def __init__(self, Gen, Dis, optim_G, optim_D, **kwargs):
        super().__init__(Gen, Dis, optim_G, optim_D, **kwargs)

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


class Cifar10Trainer(Trainer):
    def __init__(self, Gen, Dis, optim_G, optim_D, **kwargs):
        super().__init__(Gen, Dis, optim_G, optim_D, **kwargs)

    def updater(self, i, imgs):
        real_imgs = imgs.to(self.device)

        # step 1: Update Discriminator
        self.Dis.zero_grad()

        label_real = torch.full((real_imgs.shape[0],), 0.9).to(self.device)
        real_imgs = 0.9 * real_imgs + 0.1 * torch.randn(real_imgs.shape, device=self.device)

        score_real = self.Dis(real_imgs)
        lossD_real = self.criterion(score_real, label_real)
        lossD_real.backward()

        noise = torch.randn(imgs.shape[0], self.dim_z, 1, 1).to(self.device)
        fake_imgs = self.Gen(noise)
        label_fake = torch.full((fake_imgs.shape[0],), 0.1).to(self.device)
        fake_imgs = 0.9 * fake_imgs + 0.1 * torch.randn(fake_imgs.shape, device=self.device)
        score_fake = self.Dis(fake_imgs.detach())
        lossD_fake = self.criterion(score_fake, label_fake)
        lossD_fake.backward()
        lossD = lossD_fake + lossD_real

        self.optim_D.step()

        # step2 : Update Generator
        self.Gen.zero_grad()

        score_fake = self.Dis(fake_imgs)
        lossG = self.criterion(score_fake, label_real)
        lossG.backward()

        self.optim_G.step()

        return lossD, lossG, fake_imgs

    def old_updater(self, i, imgs):
        label_real = torch.ones((imgs.shape[0], 1), requires_grad=False).to(self.device)
        label_fake = torch.zeros((imgs.shape[0], 1), requires_grad=False).to(self.device)
        real_imgs = imgs.to(self.device)

        self.optim_G.zero_grad()
        noise = torch.randn(imgs.shape[0], self.dim_z)
        noise = noise.to(self.device)

        fake_imgs = self.Gen(noise)
        score_fake = self.Dis(fake_imgs)

        lossGen = self.criterion(score_fake, label_real)
        lossGen.backward()
        self.optim_G.step()

        self.optim_D.zero_grad()
        score_real = self.Dis(real_imgs)
        score_fake = self.Dis(fake_imgs.detach())
        loss_real = self.criterion(score_real, label_real)
        loss_fake = self.criterion(score_fake, label_fake)
        lossDis = (loss_fake + loss_real) / 2
        lossDis.backward()
        self.optim_D.step()

        if self.lr_sche_G is not None:
            self.lr_sche_G.step()
        if self.lr_sche_D is not None:
            self.lr_sche_D.step()

        return lossDis, lossGen, fake_imgs


def run_cifar10(total_epoch=100, maml=False, batch_size=128, gen_weight_path="", dis_weight_path="", num_sample=10,
                train_gen_interval=5, image_sample_interval=5, exp_name="", few_shot=False):
    set_seed(2023)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lrGen = 1e-4
    lrDis = 4e-4
    dim_z = 100
    img_channels = 3
    hidden_size = 64
    img_size = 64
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for cls in classes:

        Gen = ConvGenerator(dim_z=dim_z, img_size=img_size, hidden_size=hidden_size, out_channels=img_channels)
        Dis = ConvDiscriminator(hidden_size=hidden_size, img_size=img_size, in_channels=img_channels)

        if maml:
            gen_dict = torch.load(gen_weight_path.format(cls))
            dis_dict = torch.load(dis_weight_path.format(cls))
            Gen.load_state_dict(gen_dict)
            Dis.load_state_dict(dis_dict)
        # DO NOT USE INIT !!!
        init_model(Gen)
        init_model(Dis)

        optimizerGen = optim.Adam(Gen.parameters(), lr=lrGen, betas=(0.5, 0.999))
        optimizerDis = optim.Adam(Dis.parameters(), lr=lrDis, betas=(0.5, 0.999))

        # lrSchedulerGen = optim.lr_scheduler.CosineAnnealingLR(optimizerGen, T_max=10, eta_min=1e-6)
        # lrSchedulerDis = optim.lr_scheduler.CosineAnnealingLR(optimizerDis, T_max=10, eta_min=1e-6)
        if few_shot:
            trainloader, testloader = get_cifar10_fs_dataloader(cls, batch_size, img_size, num_sample=num_sample)
        else:
            trainloader, testloader = get_cifar10_dataloader(cls, batch_size, img_size)

        trainer = Cifar10Trainer(
            Gen, Dis, optimizerGen, optimizerDis, criterion=nn.BCELoss(), lr_sche_G=None, lr_sche_D=None,
            dim_z=dim_z, batch_size=batch_size, total_epoch=total_epoch,
            train_gen_interval=train_gen_interval, img_sample_interval=image_sample_interval,
            save_interval=50, log_interval=10,
            dataloader=trainloader, exp_name=exp_name.format(cls), device=device, maml=maml
        )

        trainer.train()


def run_mnist(total_epoch=100, maml=False, batch_size=128, lrGen=1e-5, lrDis=5e-5,
              gen_weight_path="", dis_weight_path="", num_sample=10,
              train_gen_interval=5, image_sample_interval=5, save_interval=50, log_interval=10,
              exp_name="", few_shot=False, speed_test=False):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim_z = 100
    param_clip_val = 0.01
    img_size = 28

    time_sum = 0

    NUM_COUNT = 10
    for i in range(0, NUM_COUNT):
        print("BEGIN MNIST: {}".format(i))
        start = time.time()
        Gen = Generator(dim_z=dim_z, img_size=img_size)
        Dis = Discriminator(nf=img_size * img_size)

        if maml:
            gen_dict = torch.load(gen_weight_path.format(i))
            dis_dict = torch.load(dis_weight_path.format(i))
            Gen.load_state_dict(gen_dict)
            Dis.load_state_dict(dis_dict)

        optimizerGen = optim.RMSprop(Gen.parameters(), lr=lrGen)
        optimizerDis = optim.RMSprop(Dis.parameters(), lr=lrDis)

        # lrSchedulerGen = optim.lr_scheduler.CosineAnnealingLR(optimizerGen, T_max=10, eta_min=1e-6)
        # lrSchedulerDis = optim.lr_scheduler.CosineAnnealingLR(optimizerDis, T_max=10, eta_min=1e-6)
        if few_shot:
            trainloader, testloader = get_mnist_fs_dataloader(i, batch_size, num_sample=num_sample)
        else:
            trainloader, testloader = get_mnist_dataloader(i, batch_size)

        trainer = MNISTTrainer(
            Gen, Dis, optimizerGen, optimizerDis, criterion=Hinge(),
            dim_z=dim_z, weight_clip_val=param_clip_val, batch_size=batch_size, total_epoch=total_epoch,
            train_gen_interval=train_gen_interval, img_sample_interval=image_sample_interval,
            save_interval=save_interval, log_interval=log_interval,
            dataloader=trainloader, exp_name=exp_name.format(i), device=device
        )

        if speed_test:
            # if u want to measure train time,
            # which means no unnecessary step in training process such as
            # record an image or write some log or calculate FID
            # then use lightning train
            trainer.lightning_train()
        else:
            trainer.train()
        tu = time.time() - start
        time_sum += tu
        print("END MNIST: {}, time used={}".format(i, tu))

    print("Avg training time: {}".format(time_sum / NUM_COUNT))


def exp_fs_cifar10():
    run_cifar10(total_epoch=1000, maml=False, batch_size=5,
                exp_name="FS_CIFAR_NEW_CONV_{}", few_shot=False)


def exp_fs_maml_cifar10():
    maml_max_iters = 50
    maml_num_tasks = 10
    maml_num_samples = 10
    maml_init_cifar10(max_iters=maml_max_iters, num_tasks=maml_num_tasks, num_samples=maml_num_samples, )

    train_total_epoch = 500
    run_cifar10(total_epoch=train_total_epoch, maml=True,
                gen_weight_path="runtime/MAML/CIFAR_MAML_INITS_{}/checkpoints" +
                                "/maml_ckpt_Gen_epoch{}.pt".format(maml_max_iters - 1),
                dis_weight_path="runtime/MAML/CIFAR_MAML_INITS_{}/checkpoints" +
                                "/maml_ckpt_Dis_epoch{}.pt".format(maml_max_iters - 1),
                exp_name="FS_MAML_CIFAR_{}")


def exp_mnist(speed_test=False):
    run_mnist(total_epoch=10000, maml=False, batch_size=64, exp_name="MNIST_BASE_{}",
              few_shot=False, speed_test=speed_test)


def exp_fs_mnist(speed_test=False):
    run_mnist(total_epoch=10000, maml=False, batch_size=8, exp_name="FS_MNIST_{}", few_shot=True, num_sample=16,
              log_interval=100, speed_test=speed_test)


def exp_fs_maml_mnist(speed_test=False):
    maml_max_iters = 50
    maml_num_tasks = 10
    maml_num_samples = 10
    avg_maml_time = maml_init_MNIST(max_iters=maml_max_iters, num_tasks=maml_num_tasks, num_samples=maml_num_samples,
                                    data_root="./data", root_path="./runtime/MAML",
                                    exp_name="INIT_MAML_MNIST_{}")
    print("Average MAMLing time: {}".format(avg_maml_time))

    train_total_epoch = 7500
    run_mnist(total_epoch=train_total_epoch, maml=True, few_shot=True, batch_size=8, log_interval=100, num_sample=16,
              image_sample_interval=10, save_interval=100,
              gen_weight_path="runtime/MAML/INIT_MAML_MNIST_{}/checkpoints" +
                              "/maml_ckpt_Gen_epoch{}.pt".format(maml_max_iters - 1),
              dis_weight_path="runtime/MAML/INIT_MAML_MNIST_{}/checkpoints" +
                              "/maml_ckpt_Dis_epoch{}.pt".format(maml_max_iters - 1),
              exp_name="FS_MAML_MNIST_{}", speed_test=speed_test)


if __name__ == "__main__":
    # run_mnist()
    set_seed(2023)
    # run_cifar10(False)

    exp_mnist(speed_test=True)
    # exp_fs_mnist(speed_test=True)
    # exp_fs_maml_mnist(speed_test=True)
