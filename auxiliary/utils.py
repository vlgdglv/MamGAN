import os
import torch
import torch.nn as nn
import datetime as dt


def init_model(model: nn.Module):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.3)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


class Logger:
    def __init__(self, log_path, end="\n", stdout=True):
        self.log_path = log_path
        self.end = end
        self.stdout = stdout

    def log(self, message):
        time_str = dt.datetime.strftime(dt.datetime.now(), '%Y-%m-%d %H:%M:%S')
        log_str = "[{}] {}{}".format(time_str, message, self.end)
        with open(self.log_path, "a") as f:
            f.write(log_str)
        if self.stdout:
            print(log_str, end="")

