import os
import time
import json
import logging
from collections import namedtuple
import uuid


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MultiAverageMeter(object):
    def __init__(self, names):
        self.AMs = {name: AverageMeter(name) for name in names}

    def __getitem__(self, name):
        return self.AMs[name].avg

    def reset(self):
        for am in self.AMs.values():
            am.reset()

    def update(self, vals, n=1):
        for name, value in vals.items():
            self.AMs[name].update(value, n=n)

    def __str__(self):
        return ' '.join([str(am) for am in self.AMs.values()])


def get_config(config_path):
    with open(config_path, "r") as f :
        config = json.load(f)
    return config

def config_to_namedtuple(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value)
        return namedtuple("GeneralDict", obj.keys())(**obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else :
        return obj
