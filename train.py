import os
import random
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
from dataset.cifar10 import get_cifar10_dataloader

from dataset.mnist import get_mnist_dataloader
from auxiliary.utils import init_model


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


def run_mnist():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lrGen = 5e-5
    lrDis = 5e-5
    batch_size = 64
    total_epoch = 200
    dim_z = 100
    train_gen_interval = 5
    image_sample_interval = 1
    param_clip_val = 0.01
    img_size = 28

    for i in range(0, 1):

        Gen = Generator(dim_z=dim_z, img_size=img_size)
        Dis = Discriminator(nf=img_size * img_size)

        # DO NOT USE INIT !!!
        # init_model(Gen)
        # init_model(Dis)

        optimizerGen = optim.RMSprop(Gen.parameters(), lr=lrGen)
        optimizerDis = optim.RMSprop(Dis.parameters(), lr=lrDis)

        # lrSchedulerGen = optim.lr_scheduler.CosineAnnealingLR(optimizerGen, T_max=10, eta_min=1e-6)
        # lrSchedulerDis = optim.lr_scheduler.CosineAnnealingLR(optimizerDis, T_max=10, eta_min=1e-6)

        trainloader, testloader = get_mnist_dataloader(i, batch_size)

        trainer = Trainer(
            Gen, Dis, optimizerGen, optimizerDis, criterion=Hinge(),
            dim_z=dim_z, weight_clip_val=param_clip_val, batch_size=batch_size, total_epoch=total_epoch,
            train_gen_interval=train_gen_interval, img_sample_interval=image_sample_interval,
            save_interval=50, log_interval=10,
            dataloader=trainloader, exp_name="NMIST_TEMP_{}".format(i), device=device
        )

        trainer.train()


class CustomTrainer(Trainer):
    def __init__(self, Gen, Dis, optim_G, optim_D, **kwargs):
        super().__init__(Gen, Dis, optim_G, optim_D, **kwargs)

    def updater(self, i, imgs):
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


def run_cifar10():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lrGen = 2e-4
    lrDis = 2e-4
    batch_size = 64
    total_epoch = 20
    dim_z = 100
    img_channels = 3
    train_gen_interval = 5
    image_sample_interval = 5
    img_size = 32
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    for cls in classes:

        Gen = ConvGenerator(dim_z=dim_z, img_size=img_size, hidden_size=512, out_channels=img_channels)
        Dis = ConvDiscriminator(hidden_size=128, img_size=img_size, in_channels=img_channels)

        # DO NOT USE INIT !!!
        # init_model(Gen)
        # init_model(Dis)

        optimizerGen = optim.Adam(Gen.parameters(), lr=lrGen, betas=(0.5, 0.99))
        optimizerDis = optim.Adam(Dis.parameters(), lr=lrDis, betas=(0.5, 0.99))

        # lrSchedulerGen = optim.lr_scheduler.CosineAnnealingLR(optimizerGen, T_max=10, eta_min=1e-6)
        # lrSchedulerDis = optim.lr_scheduler.CosineAnnealingLR(optimizerDis, T_max=10, eta_min=1e-6)

        trainloader, testloader = get_cifar10_dataloader(cls, batch_size, img_size)

        trainer = CustomTrainer(
            Gen, Dis, optimizerGen, optimizerDis, criterion=nn.BCELoss(), lr_sche_G=None, lr_sched_D=None,
            dim_z=dim_z, batch_size=batch_size, total_epoch=total_epoch,
            train_gen_interval=train_gen_interval, img_sample_interval=image_sample_interval,
            save_interval=50, log_interval=10,
            dataloader=trainloader, exp_name="CIFAR_TEMP_{}".format(cls), device=device
        )

        trainer.train()


if __name__ == "__main__":
    # run_mnist()

    run_cifar10()
