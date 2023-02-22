import logging
import os
import random
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F


def accuracy_mse(predict, target, dataset, scale=100.):
    predict = dataset.denormalize(predict.detach()) * scale
    target = dataset.denormalize(target) * scale
    return F.mse_loss(predict, target)


def set_seed(seed):
    """
        Fix all seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    cudnn.enabled = True
    cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def to_cuda(obj, device):
    if torch.is_tensor(obj):
        # return obj.cuda()
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(to_cuda(t, device) for t in obj)
    if isinstance(obj, list):
        return [to_cuda(t, device) for t in obj]
    if isinstance(obj, dict):
        return {k: to_cuda(v, device) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class AverageMeterGroup:
    """Average meter group for multiple average meters"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeterV1(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())


class AverageMeterV1:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def time_record(start):
    import logging
    import time
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    logging.info('Elapsed Time: %dh %dmin %ds' % (hour, minute, second))


def gpu_monitor(gpu, sec, used=100):
    import time
    import pynvml
    import logging

    wait_min = sec // 60
    divisor = 1024 * 1024
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    if meminfo.used / divisor < used:
        logging.info('GPU-{} is free, start runing!'.format(gpu))
        return False
    else:
        logging.info('GPU-{}, Memory: total={}MB used={}MB free={}MB, waiting {}min...'.format(
            gpu,
            meminfo.total / divisor,
            meminfo.used / divisor,
            meminfo.free / divisor,
            wait_min)
        )
        time.sleep(sec)
        return True


def run_func(args, main):
    import time
    if torch.cuda.is_available():
        while gpu_monitor(args.gpu_id, sec=60, used=5000):
            pass
    start_time = time.time()
    result = main()
    time_record(start_time)
    # email_sender(result=result, config=args)
