import copy
import argparse
import os
import datetime as dt
import time

import torch
from torch import nn
import torch.optim
from multiprocessing import Pool

from tasks import Tasks4Cifar10, Tasks4MNIST
from auxiliary.utils import Logger, check_path
from generator.convgen import ConvGenerator
from generator.generator import Generator
from discriminator.convdis import ConvDiscriminator
from discriminator.discriminator import Discriminator
from auxiliary.losses import Hinge


class MamlGAN:
    def __init__(self,
                 gen_model, gen_outer_lr, gen_inner_lr, gen_outer_optim,
                 dis_model, dis_outer_lr, dis_inner_lr, dis_outer_optim,
                 inner_criterion, outer_criterion,
                 max_iters, task_loader,

                 dis_inner_scheduler=None, dis_outer_scheduler=None,
                 gen_inner_scheduler=None, gen_outer_scheduler=None,
                 train_gen_interval=5, img_sample_interval=10, save_interval=10, log_interval=10,
                 exp_name="", root_path="../runtime/MAML", device="cpu", ):
        self.gen_model = gen_model
        self.gen_outer_lr = gen_outer_lr
        self.gen_inner_lr = gen_inner_lr
        self.gen_outer_optim = gen_outer_optim
        self.gen_inner_scheduler = gen_inner_scheduler
        self.gen_outer_scheduler = gen_outer_scheduler

        self.dis_model = dis_model
        self.dis_outer_lr = dis_outer_lr
        self.dis_inner_lr = dis_inner_lr
        self.dis_outer_optim = dis_outer_optim
        self.dis_inner_scheduler = dis_inner_scheduler
        self.dis_outer_scheduler = dis_outer_scheduler
        self.inner_criterion = inner_criterion
        self.outer_criterion = outer_criterion

        self.dim_z = 100
        self.max_iters = max_iters
        self.task_loader = task_loader
        self.num_task = len(task_loader)
        self.device = device

        self.train_gen_interval = train_gen_interval
        self.img_sample_interval = img_sample_interval
        self.log_interval = log_interval
        self.save_interval = save_interval

        if exp_name == "":
            exp_name = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
        self.exp_name = exp_name
        runtime_path = os.path.join(root_path, exp_name)
        check_path(runtime_path)
        self.runtime_path = runtime_path
        ckpt_path = os.path.join(runtime_path, "checkpoints")
        check_path(ckpt_path)
        self.ckpt_path = ckpt_path
        log_path = os.path.join(runtime_path, "train.log")
        self.logger = Logger(log_path)

        self.logger.log("MamlGAN initailized, tasks:{}, max iter: {}, device: {}, happy training."
                        .format(self.num_task, max_iters, self.device))

    def optim(self):
        raise NotImplementedError

    def save_model(self, epoch):
        torch.save(self.gen_model.state_dict(), "{}/maml_ckpt_Gen_epoch{}.pt".format(self.ckpt_path, epoch))
        torch.save(self.dis_model.state_dict(), "{}/maml_ckpt_Dis_epoch{}.pt".format(self.ckpt_path, epoch))
        self.logger.log("Epoch {} model saved at {}".format(epoch, self.ckpt_path))


class Cifar10Maml(MamlGAN):
    def __init__(self,
                 gen_model, gen_outer_lr, gen_inner_lr, gen_outer_optim,
                 dis_model, dis_outer_lr, dis_inner_lr, dis_outer_optim,
                 inner_criterion, outer_criterion,
                 max_iters, task_loader, **kwargs):
        super().__init__(
            gen_model, gen_outer_lr, gen_inner_lr, gen_outer_optim,
            dis_model, dis_outer_lr, dis_inner_lr, dis_outer_optim,
            inner_criterion, outer_criterion,
            max_iters, task_loader,

        )

    def optim(self):
        self.gen_model.to(self.device)
        self.dis_model.to(self.device)

        for it in range(self.max_iters):
            print("---------Iter {} begin ---------".format(it))

            gen_outer_loss = torch.tensor(.0).to(self.device)
            dis_outer_loss = torch.tensor(.0).to(self.device)
            self.gen_outer_optim.zero_grad()
            self.dis_outer_optim.zero_grad()

            task_iter = iter(self.task_loader)
            for task_id in range(self.num_task):
                meta_train_dl, meta_test_dl = next(task_iter)
                gen_model_copy = copy.deepcopy(self.gen_model)
                dis_model_copy = copy.deepcopy(self.dis_model)
                # gen_model_copy.load_state_dict(self.gen_model.state_dict())
                gen_inner_optim = torch.optim.Adam(params=gen_model_copy.parameters(),
                                                   lr=self.gen_inner_lr, betas=(0.5, 0.99))
                dis_inner_optim = torch.optim.Adam(params=dis_model_copy.parameters(),
                                                   lr=self.dis_inner_lr, betas=(0.5, 0.99))
                gen_inner_loss_sum = .0
                dis_inner_loss_sum = .0

                for i, data in enumerate(meta_train_dl):
                    real_imgs = data[0]
                    label_real = torch.ones((real_imgs.shape[0], 1), requires_grad=False).to(self.device)
                    label_fake = torch.zeros((real_imgs.shape[0], 1), requires_grad=False).to(self.device)
                    real_imgs = real_imgs.to(self.device)

                    gen_inner_optim.zero_grad()
                    nosie = torch.randn(real_imgs.shape[0], self.dim_z).to(self.device)

                    fake_imgs = gen_model_copy(nosie)
                    score_fake = dis_model_copy(fake_imgs)

                    loss_gen = self.inner_criterion(score_fake, label_real)
                    loss_gen.backward()
                    gen_inner_optim.step()

                    dis_inner_optim.zero_grad()
                    score_real = dis_model_copy(real_imgs)
                    score_fake = dis_model_copy(fake_imgs.detach())
                    loss_real = self.inner_criterion(score_real, label_real)
                    loss_fake = self.inner_criterion(score_fake, label_fake)
                    loss_dis = (loss_fake + loss_real) / 2
                    loss_dis.backward()
                    dis_inner_optim.step()

                    gen_inner_loss_sum += loss_gen.item()
                    dis_inner_loss_sum += loss_dis.item()

                self.gen_model = copy.deepcopy(gen_model_copy)
                self.dis_model = copy.deepcopy(dis_model_copy)

                for i, data in enumerate(meta_test_dl):
                    real_imgs = data[0]
                    label_real = torch.ones((real_imgs.shape[0], 1), requires_grad=False).to(self.device)
                    label_fake = torch.zeros((real_imgs.shape[0], 1), requires_grad=False).to(self.device)
                    real_imgs = real_imgs.to(self.device)

                    nosie = torch.randn(real_imgs.shape[0], self.dim_z).to(self.device)

                    fake_imgs = self.gen_model(nosie)
                    score_fake = self.dis_model(fake_imgs)

                    loss_gen = self.outer_criterion(score_fake, label_real)
                    gen_outer_loss += loss_gen

                    score_real = self.dis_model(real_imgs)
                    score_fake = self.dis_model(fake_imgs.detach())
                    loss_real = self.outer_criterion(score_real, label_real)
                    loss_fake = self.outer_criterion(score_fake, label_fake)
                    dis_outer_loss += (loss_fake + loss_real) / 2

                print("Task: {}, gen inner loss: {:.3f}, dis inner loss: {:.3f}"
                      .format(task_id, gen_inner_loss_sum, dis_inner_loss_sum))

            gen_outer_loss.backward()
            dis_outer_loss.backward()

            self.gen_outer_optim.step()
            self.dis_outer_optim.step()

            self.logger.log("(Iter {}), generator outer loss: {},  discriminator outer loss: {}"
                            .format(it, gen_outer_loss, dis_outer_loss))

            if it % self.save_interval == 0 or it == self.max_iters - 1:
                self.save_model(it)
                self.logger.log("models saved at {}".format(self.ckpt_path))


class MNISTMaml(MamlGAN):
    def __init__(self,
                 gen_model, gen_outer_lr, gen_inner_lr, gen_outer_optim,
                 dis_model, dis_outer_lr, dis_inner_lr, dis_outer_optim,
                 inner_criterion, outer_criterion,
                 max_iters, task_loader,
                 dis_inner_scheduler=None, dis_outer_scheduler=None,
                 gen_inner_scheduler=None, gen_outer_scheduler=None,
                 train_gen_interval=5, img_sample_interval=10, save_interval=10, log_interval=10,
                 exp_name="", root_path="../runtime/MAML", device="cpu",
                 ):
        super().__init__(
            gen_model, gen_outer_lr, gen_inner_lr, gen_outer_optim,
            dis_model, dis_outer_lr, dis_inner_lr, dis_outer_optim,
            inner_criterion, outer_criterion, max_iters, task_loader,
            dis_inner_scheduler, dis_outer_scheduler, gen_inner_scheduler, gen_outer_scheduler,
            train_gen_interval, img_sample_interval, save_interval, log_interval, exp_name, root_path, device,
        )
        self.WEIGHT_CLIP_VAL = 0.05

    def optim(self):
        start = time.time()
        self.gen_model.to(self.device)
        self.dis_model.to(self.device)

        for it in range(self.max_iters):
            # print("---------Iter {} begin ---------".format(it))

            gen_outer_loss, dis_outer_loss = .0, .0
            self.gen_outer_optim.zero_grad()
            self.dis_outer_optim.zero_grad()

            task_iter = iter(self.task_loader)
            for task_id in range(self.num_task):
                meta_train_dl, meta_test_dl = next(task_iter)
                gen_model_copy = copy.deepcopy(self.gen_model)
                dis_model_copy = copy.deepcopy(self.dis_model)
                # gen_model_copy.load_state_dict(self.gen_model.state_dict())
                gen_inner_optim = torch.optim.RMSprop(params=gen_model_copy.parameters(), lr=self.gen_inner_lr, )
                dis_inner_optim = torch.optim.Adam(params=dis_model_copy.parameters(), lr=self.dis_inner_lr, )
                gen_inner_loss_sum = .0
                dis_inner_loss_sum = .0

                # Inner update:
                for i, data in enumerate(meta_train_dl):
                    real_imgs = data[0]
                    real_imgs = real_imgs.to(self.device)
                    nosie = torch.randn(real_imgs.shape[0], self.dim_z).to(self.device)

                    dis_inner_optim.zero_grad()
                    fake_imgs = gen_model_copy(nosie).detach()
                    score_real = dis_model_copy(real_imgs)
                    score_fake = dis_model_copy(fake_imgs)

                    loss_dis = self.inner_criterion(score_fake, score_real)
                    loss_dis.backward()
                    dis_inner_optim.step()

                    for param in dis_model_copy.parameters():
                        param.data.clamp_(-self.WEIGHT_CLIP_VAL, self.WEIGHT_CLIP_VAL)

                    gen_inner_optim.zero_grad()

                    fake_imgs = gen_model_copy(nosie)
                    score_fake = dis_model_copy(fake_imgs)

                    loss_gen = self.inner_criterion(score_fake)
                    loss_gen.backward()
                    gen_inner_optim.step()

                    gen_inner_loss_sum += loss_gen.item()
                    dis_inner_loss_sum += loss_dis.item()

                # Get theta_N
                #
                # self.gen_model = copy.deepcopy(gen_model_copy)
                # self.dis_model = copy.deepcopy(dis_model_copy)

                gen_outer_loss = torch.tensor(.0).to(self.device)
                dis_outer_loss = torch.tensor(.0).to(self.device)

                gen_model_copy.zero_grad()
                dis_model_copy.zero_grad()

                for i, data in enumerate(meta_test_dl):
                    real_imgs = data[0]
                    real_imgs = real_imgs.to(self.device)
                    nosie = torch.randn(real_imgs.shape[0], self.dim_z).to(self.device)

                    fake_imgs = gen_model_copy(nosie).detach()
                    score_real = dis_model_copy(real_imgs)
                    score_fake = dis_model_copy(fake_imgs)

                    loss_dis = self.outer_criterion(score_fake, score_real)
                    loss_dis.backward()
                    dis_outer_loss += loss_dis.item()

                    for param in dis_model_copy.parameters():
                        param.data.clamp_(-self.WEIGHT_CLIP_VAL, self.WEIGHT_CLIP_VAL)

                    fake_imgs = gen_model_copy(nosie)
                    score_fake = dis_model_copy(fake_imgs)
                    loss_gen = self.outer_criterion(score_fake)
                    loss_gen.backward()
                    gen_outer_loss += loss_gen.item()

                self.transfer_grad(dis_model_copy, self.dis_model)
                self.transfer_grad(gen_model_copy, self.gen_model)

                # print("Task: {}, gen inner loss: {:.3f}, dis inner loss: {:.3f}"
                #       .format(task_id, gen_inner_loss_sum, dis_inner_loss_sum))
                self.gen_outer_optim.step()
                self.dis_outer_optim.step()
        #     self.logger.log("(Iter {}), generator outer loss: {},  discriminator outer loss: {}"
        #                     .format(it, gen_outer_loss, dis_outer_loss))
        #
        #     if it % self.save_interval == 0 or it == self.max_iters - 1:
        #         self.save_model(it)
        #         self.logger.log("models saved at {}".format(self.ckpt_path))
        # self.logger.log("Exp: {} MAMLed, time used: ".format(self.exp_name, time.time() - start))

    def transfer_grad(self, model_src, model_dst):
        for param_src, param_dst in zip(model_src.parameters(), model_dst.parameters()):
            if param_src.grad is not None:
                param_dst.grad = param_src.grad.clone().detach()

    def check_weights(self):
        for param in self.gen_model.parameters():
            print(param.data)
        for param in self.dis_model.parameters():
            print(param.data)

    def check_grad(self, model):
        gs = torch.tensor(.0).to(self.device)
        for param in model.parameters():
            if param.grad is not None:
                gs += torch.abs(param.grad).sum()
        return gs

    def optim_issued(self):
        start = time.time()
        self.gen_model.to(self.device)
        self.dis_model.to(self.device)

        for it in range(self.max_iters):
            # print("---------Iter {} begin ---------".format(it))

            gen_outer_loss, dis_outer_loss = .0, .0
            self.gen_outer_optim.zero_grad()
            self.dis_outer_optim.zero_grad()

            task_iter = iter(self.task_loader)
            for task_id in range(self.num_task):
                meta_train_dl, meta_test_dl = next(task_iter)
                gen_model_copy = copy.deepcopy(self.gen_model)
                dis_model_copy = copy.deepcopy(self.dis_model)
                # gen_model_copy.load_state_dict(self.gen_model.state_dict())
                gen_inner_optim = torch.optim.RMSprop(params=gen_model_copy.parameters(), lr=self.gen_inner_lr, )
                dis_inner_optim = torch.optim.Adam(params=dis_model_copy.parameters(), lr=self.dis_inner_lr, )
                gen_inner_loss_sum = .0
                dis_inner_loss_sum = .0

                # Inner update:
                for i, data in enumerate(meta_train_dl):
                    real_imgs = data[0]
                    real_imgs = real_imgs.to(self.device)
                    nosie = torch.randn(real_imgs.shape[0], self.dim_z).to(self.device)

                    dis_inner_optim.zero_grad()
                    fake_imgs = gen_model_copy(nosie).detach()
                    score_real = dis_model_copy(real_imgs)
                    score_fake = dis_model_copy(fake_imgs)

                    loss_dis = self.inner_criterion(score_fake, score_real)
                    loss_dis.backward()
                    dis_inner_optim.step()

                    for param in dis_model_copy.parameters():
                        param.data.clamp_(-self.WEIGHT_CLIP_VAL, self.WEIGHT_CLIP_VAL)

                    gen_inner_optim.zero_grad()

                    fake_imgs = gen_model_copy(nosie)
                    score_fake = dis_model_copy(fake_imgs)

                    loss_gen = self.inner_criterion(score_fake)
                    loss_gen.backward()
                    gen_inner_optim.step()

                    gen_inner_loss_sum += loss_gen.item()
                    dis_inner_loss_sum += loss_dis.item()

                # Get theta_N
                #
                self.gen_model = copy.deepcopy(gen_model_copy)
                self.dis_model = copy.deepcopy(dis_model_copy)

                for i, data in enumerate(meta_test_dl):
                    real_imgs = data[0]
                    real_imgs = real_imgs.to(self.device)
                    nosie = torch.randn(real_imgs.shape[0], self.dim_z).to(self.device)

                    fake_imgs = self.gen_model(nosie).detach()
                    score_real = self.dis_model(real_imgs)
                    score_fake = self.dis_model(fake_imgs)

                    loss_dis = self.outer_criterion(score_fake, score_real)
                    loss_dis.backward()
                    dis_outer_loss += loss_dis.item()

                    for param in self.dis_model.parameters():
                        param.data.clamp_(-self.WEIGHT_CLIP_VAL, self.WEIGHT_CLIP_VAL)

                    fake_imgs = self.gen_model(nosie)
                    score_fake = self.dis_model(fake_imgs)
                    loss_gen = self.outer_criterion(score_fake)
                    loss_gen.backward()
                    gen_outer_loss += loss_gen.item()

                # print("Task: {}, gen inner loss: {:.3f}, dis inner loss: {:.3f}"
                #       .format(task_id, gen_inner_loss_sum, dis_inner_loss_sum))

            # Outer update
            self.gen_outer_optim.step()
            self.dis_outer_optim.step()

        #     self.logger.log("(Iter {}), generator outer loss: {},  discriminator outer loss: {}"
        #                     .format(it, gen_outer_loss, dis_outer_loss))
        #
        #     if it % self.save_interval == 0 or it == self.max_iters - 1:
        #         self.save_model(it)
        #         self.logger.log("models saved at {}".format(self.ckpt_path))
        # self.logger.log("Exp: {} MAMLed, time used: ".format(self.exp_name, time.time() - start))


def run_maml_cifar10(clz, max_iters, num_tasks, num_samples, exp_name="CIFAR_MAML_INITS_{}"):
    lrGen = 1e-4
    lrDis = 4e-4
    dim_z = 100
    img_channels = 3
    img_size = 32

    Gen = ConvGenerator(dim_z=dim_z, img_size=img_size, hidden_size=64, out_channels=img_channels)
    Dis = ConvDiscriminator(hidden_size=64, img_size=img_size, in_channels=img_channels)

    optimizerGen = torch.optim.Adam(Gen.parameters(), lr=lrGen, betas=(0.5, 0.99))
    optimizerDis = torch.optim.Adam(Dis.parameters(), lr=lrDis, betas=(0.5, 0.99))

    maml = Cifar10Maml(
        gen_model=Gen, gen_outer_lr=1e-4, gen_inner_lr=1e-5,
        gen_outer_optim=optimizerGen,
        dis_model=Dis, dis_outer_lr=5e-5, dis_inner_lr=5e-5,
        dis_outer_optim=optimizerDis,
        inner_criterion=nn.BCELoss(), outer_criterion=nn.BCELoss(),
        max_iters=max_iters,
        task_loader=Tasks4Cifar10(root="../data", batch_size=5,
                                  num_tasks=num_tasks, num_samples=num_samples, class_name=clz),
        exp_name=exp_name.format(clz), device="cuda" if torch.cuda.is_available() else "cpu"
    )
    maml.optim()


def maml_init_cifar10(max_iters=10, num_tasks=10, num_samples=10):
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for cls in classes:
        run_maml_cifar10(cls, max_iters, num_tasks, num_samples)


def run_maml_NMIST(target_num, max_iters, num_tasks, num_samples, data_root="../data", root_path="../runtime/MAML",
                   exp_name="INIT_MAML_INITS_TEST_{}"):
    lrGen = 1e-5
    lrDis = 5e-5
    dim_z = 100
    img_size = 28

    Gen = Generator(dim_z=dim_z, img_size=img_size)
    Dis = Discriminator(nf=img_size * img_size)

    optimizerGen = torch.optim.RMSprop(Gen.parameters(), lr=lrGen)
    optimizerDis = torch.optim.RMSprop(Dis.parameters(), lr=lrDis)

    maml = MNISTMaml(
        gen_model=Gen, gen_outer_lr=lrGen, gen_inner_lr=lrGen,
        gen_outer_optim=optimizerGen,
        dis_model=Dis, dis_outer_lr=lrDis, dis_inner_lr=lrDis,
        dis_outer_optim=optimizerDis,
        inner_criterion=Hinge(), outer_criterion=Hinge(),
        max_iters=max_iters, root_path=root_path,
        task_loader=Tasks4MNIST(root=data_root, batch_size=5, download=True,
                                num_tasks=num_tasks, num_samples=num_samples, target_number=target_num),
        exp_name=exp_name.format(target_num), device="cuda" if torch.cuda.is_available() else "cpu"
    )
    maml.optim()


def maml_init_MNIST(max_iters=10, num_tasks=10, num_samples=10, data_root="../data", root_path="../runtime/MAML",
                    exp_name="INIT_MAML_MNIST_TEST_{}"):
    time_sum = 0
    for i in range(0, 1):
        start = time.time()
        run_maml_NMIST(i, max_iters, num_tasks, num_samples, data_root, root_path, exp_name)
        tu = time.time() - start
        time_sum += tu
        print("MAMLed MNIST {}, time used {}".format(i, tu))
    return time_sum / 10


if __name__ == "__main__":
    maml_init_MNIST(max_iters=10, num_tasks=10, exp_name="FUCK_INIT_MAML_MNIST_TEST_{}")
