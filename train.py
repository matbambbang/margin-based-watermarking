import os
import argparse
import json
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image

from models import mnist, cifar10, resnet, queries
from loaders import get_mnist_loaders, get_cifar10_loaders, get_cifar100_loaders, get_svhn_loaders
from utils import MultiAverageMeter
import waitGPU


CIFAR_QUERY_SIZE = (3, 32, 32)


def loop(model, query_model, loader, opt, lr_scheduler, epoch, logger, output_dir, max_epoch=100, train_type='standard', mode='train', device='cuda', addvar=None):
    meters = MultiAverageMeter(['nat loss', 'nat acc', 'query loss', 'query acc'])

    for batch_idx, batch in enumerate(loader):
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
        nat_acc = (preds.topk(1, dim=1).indices == labels.unsqueeze(1)).all(1).float().mean()
        nat_loss = F.cross_entropy(preds, labels, reduction='none')

        if train_type == 'none':
            with torch.no_grad():
                model.eval()
                query, response = query_model()
                query_preds = model(query)
                query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
                query_loss = F.cross_entropy(query_preds, response)
                if mode == 'train':
                    model.train()

            loss = nat_loss.mean()
            
        elif train_type == 'base':
            query, response = query_model()
            query_preds = model(query)
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = F.cross_entropy(query_preds, response, reduction='none')
            loss = torch.cat([nat_loss, query_loss]).mean()
            
        elif train_type == 'margin':
            num_sample_fn = lambda x: np.interp([x], [0, max_epoch], [25, 25])[0]
            num_sample = int(num_sample_fn(epoch))
            if mode == 'train':
                query, response = query_model(discretize=False, num_sample=num_sample)
                for _ in range(5):
                    query = query.detach()
                    query.requires_grad_(True)
                    query_preds = model(query)
                    query_loss = F.cross_entropy(query_preds, response)
                    query_loss.backward()
                    query = query + query.grad.sign() * (1/255)
                    query = query_model.project(query)
                    model.zero_grad()
            else:
                query, response = query_model(discretize=(mode!='train'))
            query_preds = model(query)
            query_acc = (query_preds.topk(1, dim=1).indices == response.unsqueeze(1)).all(1).float().mean()
            query_loss = addvar * F.cross_entropy(query_preds, response, reduction='none')
            loss = torch.cat([nat_loss, query_loss]).mean()

        if mode == 'train':
            loss.backward()
            opt.step()
        
        meters.update({
            'nat loss': nat_loss.mean().item(),
            'nat acc': nat_acc.item(),
            'query loss': query_loss.mean().item(),
            'query acc': query_acc.item()
        }, n=images.size(0))

        if batch_idx % 100 == 0 and mode == 'train':
            logger.info('=====> {} {}'.format(mode, str(meters)))
        
    logger.info("({:3.1f}%) Epoch {:3d} - {} {}".format(100*epoch/max_epoch, epoch, mode.capitalize().ljust(6), str(meters)))
    if mode == 'test' and (epoch+1) % 20 == 0:
        save_image(query.cpu(), os.path.join(output_dir, "images", f"query_image_{epoch}.png"), nrow=query.size(0))
    return meters


def save_ckpt(model, model_type, query_model, query_type, opt, nat_acc, query_acc, epoch, name):
    torch.save({
        "model": {
            "state_dict": model.state_dict(),
            "type": model_type
        },
        "query_model": {
            "state_dict": query_model.state_dict(),
            "type": query_type
        },
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "val_nat_acc": nat_acc,
        "val_query_acc": query_acc
    }, name)


def train(args, output_dir):
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
        train_loader, valid_loader, test_loader = get_cifar10_loaders()
    elif args.dataset == 'cifar100':
        query_size = CIFAR_QUERY_SIZE
        model_archive = cifar10.models_cifar100
        train_loader, valid_loader, test_loader = get_cifar100_loaders()
    elif args.dataset == 'svhn':
        query_size = CIFAR_QUERY_SIZE
        model_archive = resnet.models
        train_loader, valid_loader, test_loader = get_svhn_loaders()
    
    response_scale = 100 if args.dataset == 'cifar100' else 10
    model = model_archive[args.model_type](num_classes=response_scale)
    query = queries.queries[args.query_type](query_size=(args.num_query, *query_size),
                                response_size=(args.num_query,), query_scale=255, response_scale=response_scale)
    if args.train_type not in ['none']:
        query_init_set, _ = torch.utils.data.random_split(valid_loader.dataset, [args.num_mixup*args.num_query, 1000-args.num_mixup*args.num_query])
        query.initialize(query_init_set)
        
    query.eval()
    init_query, _ = query()
    query.train()
    save_image(init_query, os.path.join(output_dir, "images", f"query_image_init.png"), nrow=10)
    model.to(args.device)
    query.to(args.device)

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
    best_val_query_acc = 0
    
    # save init #
    save_ckpt(model, args.model_type, query, args.query_type, opt, None, None, 0, os.path.join(output_dir, "checkpoints", "checkpoint_init.pt"))

    for epoch in range(args.epoch):
        model.train()
        query.train()
        train_meters = loop(model, query, train_loader, opt, lr_scheduler, epoch, logger, output_dir,
                        train_type=args.train_type, max_epoch=args.epoch, mode='train', device=args.device, addvar=args.variable)

        with torch.no_grad():
            model.eval()
            query.eval()
            val_meters = loop(model, query, valid_loader, opt, lr_scheduler, epoch, logger, output_dir,
                            train_type=args.train_type, max_epoch=args.epoch, mode='val', device=args.device, addvar=args.variable)
            test_meters = loop(model, query, test_loader, opt, lr_scheduler, epoch, logger, output_dir,
                            train_type=args.train_type, max_epoch=args.epoch, mode='test', device=args.device, addvar=args.variable)

            if not os.path.exists(os.path.join(output_dir, "checkpoints")):
                os.makedirs(os.path.join(output_dir, "checkpoints"))
            
            if (epoch+1) % 20 == 0:
                save_ckpt(model, args.model_type, query, args.query_type, opt, val_meters['nat acc'], val_meters['query acc'], epoch,
                        os.path.join(output_dir, "checkpoints", f"checkpoint_{epoch}.pt"))
            
            if best_val_nat_acc <= val_meters['nat acc']:
                save_ckpt(model, args.model_type, query, args.query_type, opt, val_meters['nat acc'], val_meters['query acc'], epoch,
                        os.path.join(output_dir, "checkpoints", "checkpoint_nat_best.pt"))
                best_val_nat_acc = val_meters['nat acc']
            
            if best_val_query_acc <= val_meters['query acc']:
                save_ckpt(model, args.model_type, query, args.query_type, opt, val_meters['nat acc'], val_meters['query acc'], epoch,
                        os.path.join(output_dir, "checkpoints", "checkpoint_query_best.pt"))
                best_val_query_acc = val_meters['query acc']

            save_ckpt(model, args.model_type, query, args.query_type, opt, val_meters['nat acc'], val_meters['query acc'], epoch,
                    os.path.join(output_dir, "checkpoints", "checkpoint_latest.pt"))

    logger.info("="*100)
    logger.info("Best valid nat acc   : {:.4f}".format(best_val_nat_acc))
    logger.info("Best valid query acc : {:.4f}".format(best_val_query_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='sanity check for watermarking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dir",
        type=str,
        help='output dir',
        default='experiments',
        required=False)
    parser.add_argument("-dt", "--dataset",
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100', 'svhn'])
    parser.add_argument("-tt", "--train-type",
        type=str,
        default='margin',
        help='train type, none: no watermark, base: baseline for watermark',
        choices=['none', 'base', 'margin'])
    parser.add_argument("-mt", "--model-type",
        type=str,
        help='model type',
        default='res34',
        choices=['res18', 'res34', 'res50', 'res101', 'res152'])
    parser.add_argument("-qt", "--query-type",
        type=str,
        help='type of query',
        default='stochastic',
        choices=['stochastic'])
    parser.add_argument('-msg', '--message',
        type=str,
        help='additional message for naming the exps.',
        default='')
    parser.add_argument('-nq', "--num-query",
        type=int,
        help='# of queries',
        default=10)
    parser.add_argument('-nm', "--num-mixup",
        type=int,
        help='# of mixup',
        default=1)
    parser.add_argument('-ep', "--epoch",
        type=int,
        default=200,
        required=False)
    parser.add_argument('-v', "--variable",
        type=float,
        default=0.1)
    parser.add_argument("--device",
        default='cuda')
    parser.add_argument("--seed",
        type=int,
        default=0)

    waitGPU.wait(gpu_ids=[0,1,2,3,4,5,6,7], nproc=0, interval=60)

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'svhn':
        assert args.model_type in ['res18', 'res34', 'res50', 'res101', 'res152']
        
    exp_name = "_".join([args.dataset, args.model_type, args.train_type, str(args.num_query), args.message])
    output_dir = os.path.join(args.dir, exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for s in ['images', 'checkpoints']:
        extra_dir = os.path.join(output_dir, s)
        if not os.path.exists(extra_dir):
            os.makedirs(extra_dir)
    
    train(args, output_dir)