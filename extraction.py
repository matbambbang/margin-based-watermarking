import os
import argparse
import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

from models import mnist, cifar10, queries, resnet
from loaders import get_mnist_loaders, get_cifar10_loaders, get_svhn_loaders, get_cifar100_loaders
from utils import MultiAverageMeter
import waitGPU


MNIST_QUERY_SIZE = (1, 28, 28)
CIFAR_QUERY_SIZE = (3, 32, 32)


def extract_loop(model, teacher, query_model, loader, opt, lr_scheduler, epoch, logger, output_dir,
                temperature=1.0, max_epoch=100, mode='train', device='cuda'):
    if mode == 'train':
        meters = MultiAverageMeter(['extract loss'])
    else:
        meters = MultiAverageMeter(['nat acc', 'teach acc', 'extract loss'])
    query_meters = MultiAverageMeter(['query loss', 'query acc', 'teach loss', 'teach acc'])
    T = temperature

    for batch_idx, batch in enumerate(loader):
        if mode == 'train':
            model.train()
        else:
            model.eval()
        images = batch[0]
        labels = batch[1].long()
        epoch_with_batch = epoch + (batch_idx+1) / len(loader)
        if lr_scheduler is not None:
            lr_new = lr_scheduler(epoch_with_batch)
            for param_group in opt.param_groups:
                param_group.update(lr=lr_new)

        images = images.to(device)
        labels = labels.to(device)
        if mode == 'train':
            model.train()
            opt.zero_grad()

        preds = model(images)
        teacher_preds = teacher(images)
        if mode != 'train':
            nat_acc = (preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()
            teacher_acc = (teacher_preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()

        extract_loss = F.kl_div(F.log_softmax(preds / T, dim=-1), F.softmax(teacher_preds / T, dim=-1), reduction='batchmean')
        # if you want to perform extraction with hard labels, use below two commented lines:
        # teacher_preds_hard = teacher_preds.topk(1, dim=1).indices.squeeze()
        # extract_loss = F.cross_entropy(preds, teacher_preds_hard)
        if mode == 'train':
            extract_loss.backward()
            opt.step()

        if mode == 'train':
            meters.update({
                'extract loss': extract_loss.item()
            }, n=images.size(0))
        else:
            meters.update({
                'nat acc': nat_acc.item(),
                'teach acc': teacher_acc.item(),
                'extract loss': extract_loss.item()
            }, n=images.size(0))

        if batch_idx % 100 == 0 and mode == 'train':
            logger.info('=====> {} {}'.format(mode, str(meters)))

    with torch.no_grad():
        model.eval()
        query, response = query_model()
        query_preds = model(query)
        teacher_query_preds = teacher(query)
        query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
        query_loss = F.cross_entropy(query_preds, response)
        teacher_query_acc = (teacher_query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
        teacher_query_loss = F.cross_entropy(teacher_query_preds, response)

        query_meters.update({
            'query loss': query_loss.mean().item(),
            'query acc': query_acc.item(),
            'teach loss': teacher_query_loss.mean().item(),
            'teach acc': teacher_query_acc.item()
        }, n=query.size(0))

    logger.info("({:3.1f}%) Epoch {:3d} - {} {}".format(100*epoch/max_epoch, epoch, mode.capitalize().ljust(6), str(meters)))
    logger.info("({:3.1f}%) Query {:3d} - {} {}".format(100*epoch/max_epoch, epoch, mode.capitalize().ljust(6), str(query_meters)))

    return meters, query_meters

def save_extract_ckpt(model, model_type, opt, nat_acc, query_acc, epoch, name):
    torch.save({
        "model": {
            "state_dict": model.state_dict(),
            "type": model_type
        },
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "val_nat_acc": nat_acc,
        "val_query_acc": query_acc
    }, name)

def extraction(args, output_dir):
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(output_dir, 'output.log')),
                logging.StreamHandler()
                ])

    if args.dataset == 'cifar10':
        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models
        if args.dataset_source == 'cifar100':
            train_loader, _, _ = get_cifar100_loaders()
        elif args.dataset_source == 'cifar10':
            train_loader, _, _ = get_cifar10_loaders()
        elif args.dataset_source == 'svhn':
            train_loader, _, _ = get_svhn_loaders()
        _, valid_loader, test_loader = get_cifar10_loaders()
        extract_model_archive = resnet.models
    elif args.dataset == 'cifar100':
        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models_cifar100
        if args.dataset_source == 'cifar100':
            train_loader, _, _ = get_cifar100_loaders()
        elif args.dataset_source == 'cifar10':
            train_loader, _, _ = get_cifar10_loaders()
        _, valid_loader, test_loader = get_cifar100_loaders()
        extract_model_archive = resnet.models
    elif args.dataset == 'svhn':
        query_size = CIFAR_QUERY_SIZE
        model_archive = resnet.models
        if args.dataset_source == 'svhn':
            train_loader, _, _ = get_svhn_loaders()
        elif args.dataset_source == 'cifar10':
            train_loader, _, _ = get_cifar10_loaders()
        _, valid_loader, test_loader = get_svhn_loaders()
        extract_model_archive = resnet.models

    resume = os.path.join(output_dir, "../", "checkpoints", "checkpoint_nat_best.pt")
    d = torch.load(resume)
    logger.info(f"logging model checkpoint {d['epoch']}...")
    
    num_classes = 100 if args.dataset == 'cifar100' else 10
    model = extract_model_archive[args.model_type](num_classes=num_classes)
    teacher = model_archive[d['model']['type']](num_classes=num_classes)
    teacher.load_state_dict(d['model']['state_dict'])
    query = queries.queries[d['query_model']['type']](query_size=(args.num_query, *query_size),
                                response_size=(args.num_query,), query_scale=255, response_scale=num_classes)
    query.load_state_dict(d['query_model']['state_dict'], strict=False)

    model.to(args.device)
    teacher.to(args.device)
    query.to(args.device)
    teacher.eval()

    if args.dataset in ['cifar10', 'cifar100']:
        opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
        lr_scheduler = lambda t: np.interp([t],\
            [0, 100, 100, 150, 150, 200],\
            [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]
    elif args.dataset == 'svhn':
        opt = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001)
        lr_scheduler = lambda t: np.interp([t],\
            [0, 100, 100, 150, 150, 200],\
            [0.1, 0.1, 0.01, 0.01, 0.001, 0.001])[0]

    best_val_nat_acc = 0
    best_val_nat_query_acc = 0
    best_val_query_acc = 0
    best_val_query_nat_acc = 0
    best_test_nat_acc = 0
    best_test_query_acc = 0

    query.eval()
    for epoch in range(args.epoch):
        model.train()
        train_meters, train_q_meters = extract_loop(model, teacher, query, train_loader,
                opt, lr_scheduler, epoch, logger, output_dir,
                max_epoch=args.epoch, mode='train', device=args.device)

        with torch.no_grad():
            model.eval()
            val_meters, val_q_meters = extract_loop(model, teacher, query, valid_loader,
                opt, lr_scheduler, epoch, logger, output_dir,
                max_epoch=args.epoch, mode='val', device=args.device)
            test_meters, test_q_meters = extract_loop(model, teacher, query, test_loader,
                opt, lr_scheduler, epoch, logger, output_dir,
                max_epoch=args.epoch, mode='test', device=args.device)

            if not os.path.exists(os.path.join(output_dir, "checkpoints")):
                os.makedirs(os.path.join(output_dir, "checkpoints"))
            
            if (epoch+1) % 25 == 0:
                save_extract_ckpt(model, args.model_type, opt, val_meters['nat acc'], val_q_meters['query acc'], epoch,
                                os.path.join(output_dir, "checkpoints", f"checkpoint_{epoch}.pt"))
            
            if best_val_nat_acc <= val_meters['nat acc']:
                save_extract_ckpt(model, args.model_type, opt, val_meters['nat acc'], val_q_meters['query acc'], epoch,
                                os.path.join(output_dir, "checkpoints", f"checkpoint_nat_best.pt"))
                best_val_nat_acc = val_meters['nat acc']
                best_val_nat_query_acc = val_q_meters['query acc']
                best_test_nat_acc = test_meters['nat acc']
                best_test_query_acc = test_q_meters['query acc']
                
            if best_val_query_acc <= val_q_meters['query acc']:
                save_extract_ckpt(model, args.model_type, opt, val_meters['nat acc'], val_q_meters['query acc'], epoch,
                                os.path.join(output_dir, "checkpoints", f"checkpoint_query_best.pt"))
                best_val_query_acc = val_q_meters['query acc']
                best_val_query_nat_acc = val_meters['nat acc']

            save_extract_ckpt(model, args.model_type, opt, val_meters['nat acc'], val_q_meters['query acc'], epoch,
                            os.path.join(output_dir, "checkpoints", f"checkpoint_latest.pt"))
            
    logger.info("="*100)
    logger.info("Best valid query acc : {:.4f}".format(best_val_nat_query_acc))
    logger.info("Best valid nat acc   : {:.4f}".format(best_val_nat_acc))
    logger.info("Best query acc : {:.4f}".format(best_test_query_acc))
    logger.info("Best nat acc   : {:.4f}".format(best_test_nat_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='sanity check for distillation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--exp-name',
        type=str,
        default='experiments')
    parser.add_argument("-dt", "--dataset",
        type=str,
        default='mnist',
        choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument("-dmt", "--model-type",
        type=str,
        help='extraction model type',
        default='res34',
        choices=['res18', 'res34', 'res50', 'res101', 'res152', 'vggc', 'vggc-large', 'vggc-xlarge'])
    parser.add_argument('-tmt', '--teacher-model-type',
        type=str,
        help='teacher model type',
        default='res34',
        choices=['res18', 'res34', 'res50', 'res101', 'res152'])
    parser.add_argument("-tt", "--train-type",
        type=str,
        default='standard',
        help='train type, none: no watermark, ood: rand-noise watermark, base: baseline for ind watermark',
        choices=['none', 'base', 'margin'])
    parser.add_argument("-dsrc", "--dataset-source",
        type=str,
        default='cifar100',
        choices=['cifar100', 'cifar10', 'svhn'])
    parser.add_argument('-msg', '--message',
        type=str,
        help='additional message for naming the exps.',
        default='')
    parser.add_argument('-admsg', '--additional-message',
        type=str,
        default='')
    parser.add_argument('-qt', '--query-type',
        type=str,
        default='')
    parser.add_argument('-nq', "--num-query",
        type=int,
        help='# of queries',
        default=10)
    parser.add_argument('-ep', "--epoch",
        type=int,
        default=200,
        required=False)
    parser.add_argument("--device",
        default='cuda')
    parser.add_argument("--seed",
        type=int,
        default=0)

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
        assert args.model_type in ['res18', 'res34', 'res50', 'res101', 'res152']
        
    if args.query_type == '':
        exp_name = "_".join([args.dataset, args.teacher_model_type, args.train_type, str(args.num_query), args.message])
    else:
        exp_name = "_".join([args.dataset, args.teacher_model_type, args.query_type, args.train_type, str(args.num_query), str(args.num_mixup), args.message])
    output_dir = os.path.join(args.exp_name, exp_name, 'extraction'+args.additional_message)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for s in ['images', 'checkpoints']:
        extra_dir = os.path.join(output_dir, s)
        if not os.path.exists(extra_dir):
            os.makedirs(extra_dir)
    
    extraction(args, output_dir)