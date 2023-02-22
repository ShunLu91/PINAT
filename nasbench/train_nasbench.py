from argparse import ArgumentParser
import logging
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import kendalltau

from datasets.data_factory import create_dataloader
from models.model_factory import create_model
from utils import AverageMeterGroup, set_seed, to_cuda, run_func, accuracy_mse

parser = ArgumentParser()
# exp and dataset
parser.add_argument("--exp_name", type=str, default='PINAT')
parser.add_argument("--bench", type=str, default='101')
parser.add_argument("--train_split", type=str, default="100")
parser.add_argument("--eval_split", type=str, default="all")
parser.add_argument("--dataset", type=str, default='cifar10')
# training settings
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu_id", type=int, default=0)
parser.add_argument("--epochs", default=300, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--wd", default=1e-3, type=float)
parser.add_argument("--train_batch_size", default=10, type=int)
parser.add_argument("--eval_batch_size", default=10240, type=int)
parser.add_argument("--train_print_freq", default=1e6, type=int)
parser.add_argument("--eval_print_freq", default=10, type=int)
args = parser.parse_args()

# initialize log info
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logging.info(args)

# set cpu/gpu device
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def check_arguments():
    # set seed
    set_seed(args.seed)

    # check ckpt and results dir
    assert args.exp_name is not None
    ckpt_dir = './checkpoints/nasbench_%s/' % args.bench
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    res_dir = './results/'
    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    # check data split
    if args.bench == '101':
        train_splits = ['100', '172', '424', '4236']
        test_splits = ['100', 'all']
    elif args.bench == '201':
        train_splits = ['78', '156', '469', '781', '1563']
        test_splits = ['all']
    else:
        raise ValueError('No defined NAS bench!')
    assert args.train_split in train_splits
    assert args.eval_split in test_splits


def train(train_set, train_loader, model, optimizer, lr_scheduler, criterion):
    model.train()
    for epoch in range(args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        meters = AverageMeterGroup()
        for step, batch in enumerate(train_loader):
            batch = to_cuda(batch, device)
            target = batch["val_acc"]
            optimizer.zero_grad()
            predict = model(batch)
            loss = criterion(predict, target.float())
            loss.backward()
            optimizer.step()
            mse = accuracy_mse(predict.squeeze(), target.squeeze(), train_set)
            meters.update({"loss": loss.item(), "mse": mse.item()}, n=target.size(0))
            if (step + 1) % args.train_print_freq == 0 or step + 1 == len(train_loader):
                logging.info("Epoch [%d/%d] Step [%d/%d] lr = %.3e  %s",
                             epoch + 1, args.epochs, step + 1, len(train_loader), lr, meters)
        lr_scheduler.step()
    return model


def evaluate(test_set, test_loader, model, criterion):
    model.eval()
    meters = AverageMeterGroup()
    predicts, targets = [], []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            batch = to_cuda(batch, device)
            target = batch["test_acc"]
            predict = model(batch)
            predicts.append(predict.cpu().numpy())
            targets.append(target.cpu().numpy())
            meters.update({"loss": criterion(predict, target).item(),
                           "mse": accuracy_mse(predict.squeeze(), target.squeeze(), test_set).item()},
                          n=target.size(0))
            if step % args.eval_print_freq == 0 or step + 1 == len(test_loader):
                logging.info("Evaluation Step [%d/%d]  %s", step + 1, len(test_loader), meters)
    predicts = np.concatenate(predicts)
    targets = np.concatenate(targets)
    kendall_tau = kendalltau(predicts, targets)[0]
    return kendall_tau, predicts, targets


def main():
    # check arguments
    check_arguments()

    # create dataloader and model
    train_loader, test_loader, train_set, test_set = create_dataloader(args)
    model = create_model(args)
    model = model.to(device)
    print(model)
    logging.info('PINAT params.: %f M' %
                 (sum(_param.numel() for _param in model.parameters()) / 1e6))
    logging.info('Training on NAS-Bench-%s, train_split: %s, eval_split: %s' %
                 (args.bench, args.train_split, args.eval_split))

    # define loss, optimizer, and lr_scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # train and evaluate predictor
    model = train(train_set, train_loader, model, optimizer, lr_scheduler, criterion)
    kendall_tau, predict_all, target_all = evaluate(test_set, test_loader, model, criterion)
    logging.info("Kendalltau: %.6f", kendall_tau)

    # save checkpoint
    ckpt_dir = './checkpoints/nasbench_%s/' % args.bench
    ckpt_path = os.path.join(ckpt_dir, '%s_tau%.6f_ckpt.pt' % (args.exp_name, kendall_tau))
    torch.save(model.state_dict(), ckpt_path)
    logging.info('Save model to %s' % ckpt_path)

    # write results
    with open('./results/preds_%s.txt' % args.bench, 'a') as f:
        f.write("EXP:%s\tlr: %s\ttrain: %s\ttest: %s\tkendall_tau: %.6f\n"
                % (args.exp_name, args.lr, args.train_split, args.eval_split, kendall_tau))


if __name__ == "__main__":
    run_func(args, main)
