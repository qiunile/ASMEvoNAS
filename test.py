import os
import sys
import glob
import numpy as np
import torch
import logging
import asmea.utils.utils as utils
import argparse
import torch.nn as nn
import asmea.network.genotypes as genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import asmea.data.transforms as eeea_transforms
import torch.nn as nn
import torch.utils
from asmea.network.model import NetworkCIFAR as Network
from asmea.network.model import PyramidNetworkCIFAR as PyramidNetwork
import torchvision.transforms as transforms

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='/raid/data/zhangchuang/code/zhangchuang/EEEA-Net/datasets', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpus', type=str, default='1', help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--model_path', type=str, default='/raid/data/zhangchuang/code/zhangchuang/EEEA-Net/outputs10/model_best.pth.tar',
                    help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--classes', type=int, default=10, help='classes')
parser.add_argument('--mode', type=str, default='FP32', choices=['FP32', 'FP16', 'amp'])



parser.add_argument('--arch', type=str, default='ASMEA_53', help='which architecture to use')
parser.add_argument('--set', type=str, default="cifar10", help='data set')
parser.add_argument('--pyramid', action='store_true', default=True, help='pyramid')
parser.add_argument('--se', action='store_true', default=True, help='use se')



parser.add_argument('--increment', type=int, default=6, help='filter increment')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use autoaugment')
parser.add_argument('--cutmix', action='store_true', default=False, help='use cutmix')
parser.add_argument('--duplicates', type=int, default=1, help='duplicates')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpus)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CIFAR_CLASSES = 10
if args.set == 'cifar100':
    CIFAR_CLASSES = 100


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info('gpu device = %s' % args.gpus)
    logging.info("args = %s", args)

    genotype = eval("genotypes.%s" % args.arch)
    if args. pyramid== False:
        model = Network(args.init_channels, args.classes, args.layers, args.auxiliary, genotype, args.drop_path_prob,
                        args.mode, args.se)
    else:
        model = PyramidNetwork(args.init_channels, args.classes, args.layers, args.auxiliary, genotype,
                               args.drop_path_prob, args.mode, args.se, args.increment)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.eval()
    pretrained_dict = torch.load(args.model_path)
    model.load_state_dict(pretrained_dict['state_dict'],False)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    if args.set == 'cifar10':
        _, test_transform  = eeea_transforms._data_transforms_cifar10(args)
        test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

    elif args.set == 'cifar100':
        _, test_transform  = eeea_transforms._data_transforms_cifar100(args)
        test_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=2)

    model.module.drop_path_prob =  0.2
    test_acc, test_obj = infer(test_queue, model, criterion)
    logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(test_queue):

        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
             logits, _ = model(input)
             loss = criterion(logits, target)

        # logits, _ = model(input)
        # loss = criterion(logits, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()