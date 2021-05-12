import random
import time
import warnings
import sys
import argparse
import shutil
import os.path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset

sys.path.append('../../..')
from dalib.adaptation.multi_source.mdan import MDANet
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log, args.phase)
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.center_crop:
        train_transform = T.Compose([
            ResizeImage(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        train_transform = T.Compose([
            ResizeImage(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    val_transform = T.Compose([
        ResizeImage(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])

    dataset = datasets.__dict__[args.data]

    # If sources is not set,
    # then use all domains except the target domain.
    global num_domains
    num_domains = len(dataset.domains())

    if args.sources is None:
        args.sources = dataset.domains()
        args.sources.remove(args.target)

    print("Source: {} Target: {}".format(args.sources, args.target))

    # 多个source_dataset
    train_source_datasets = [dataset(root=args.root, task=source, download=True, transform=train_transform) for source in args.sources]
    train_source_loaders = [DataLoader(train_source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
                            for train_source_dataset in train_source_datasets]
    train_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, split='test', download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    train_source_iters = [ForeverDataIterator(train_source_loader) for train_source_loader in train_source_loaders]
    train_target_iter = ForeverDataIterator(train_target_loader)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True).to(device)
    global num_classes
    num_classes = val_dataset.num_classes

    error_dicts = {}

    # model
    mdan = MDANet(backbone, num_classes, num_domains-1, bottleneck_dim=args.bottleneck_dim).to(device)
    optimizer = optim.Adadelta(mdan.parameters(), lr=args.lr)

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iters, train_target_iter, mdan, optimizer, epoch, args)

        # validation
        acc1 = validate(val_loader, mdan, args)

        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    logger.close()

def train(train_source_iters: list, train_target_iter: ForeverDataIterator,
          model: MDANet, optimizer: optim.Adadelta, epoch: int, args: argparse.Namespace):

    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    Losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, Losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    xs = [next(train_source_iter)[0] for train_source_iter in train_source_iters]
    ys = [next(train_source_iter)[1] for train_source_iter in train_source_iters]
    sz = [next(train_source_iter)[0].shape[0] for train_source_iter in train_source_iters]

    slabels = torch.ones(args.batch_size, requires_grad=False).type(torch.LongTensor).to(device)
    tlabels = torch.zeros(args.batch_size, requires_grad=False).type(torch.LongTensor).to(device)

    tinputs, _ = next(train_target_iter)

    # model forward
    logprobs, sdomains, tdomains = model(xs, tinputs)

    # Compute prediction accuracy on multiple training sources.
    losses = torch.stack([F.nll_loss(logprobs[j], ys[j]) for j in range(num_domains - 1)])
    domain_losses = torch.stack([F.nll_loss(sdomains[j], slabels) +
                                 F.nll_loss(tdomains[j], tlabels) for j in range(num_domains - 1)])
    # Different final loss function depending on different training modes.
    if args.mode == "maxmin":
        loss = torch.max(losses) + args.mu * torch.min(domain_losses)
    elif args.mode == "dynamic":
        loss = torch.log(torch.sum(torch.exp(args.gamma * (losses + args.mu * domain_losses)))) / args.gamma
    else:
        raise ValueError("No support for the training mode on madnNet: {}.".format(args.mode))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    cls_acces = [accuracy(logprobs[j], ys[j])[0] for j in range(num_domains - 1)]
    domain_acces = [ (accuracy(sdomains[j], slabels)[0] + accuracy(tdomains[j], tlabels)[0]) / 2 for j in range(num_domains - 1)]

    Losses.update(loss.item(), xs[0].size(0))
    cls_accs.update(torch.mean(torch.Tensor(cls_acces)).item(), xs[0].size(0))
    domain_accs.update(torch.mean(torch.Tensor(domain_acces)).item(), xs[0].size(0))

    progress.display(epoch)


def validate(val_loader: DataLoader, model: MDANet , args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    if args.per_class_eval:
        classes = val_loader.dataset.classes
        confmat = ConfusionMatrix(len(classes))
    else:
        confmat = None

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output (log_probs)
            output = model.inference(images)
            loss = F.nll_loss(output, target)

            # measure accuracy and record loss
            k = min(num_classes, 5)
            acc1, acck = accuracy(output, target, topk=(1, k))
            if confmat:
                confmat.update(target, output.argmax(1))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acck.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        if confmat:
            print(confmat.format(classes))

    return top1.avg


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='DANN for Unsupervised Domain Adaptation')
    # dataset parameters
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--sources', nargs='+', default=None,
                        help='source domain(s). Use all domains except the target domain if set None')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('--center-crop', default=False, action='store_true',
                        help='whether use center crop during training')
    # model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('--bottleneck-dim', default=256, type=int,
                        help='Dimension of bottleneck')
    # training parameters
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-gamma', default=0.001, type=float, help='parameter for lr scheduler')
    parser.add_argument('--lr-decay', default=0.75, type=float, help='parameter for lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-i', '--iters-per-epoch', default=1000, type=int,
                        help='Number of iterations per epoch')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--per-class-eval', action='store_true',
                        help='whether output per-class accuracy during evaluation')
    parser.add_argument("--log", type=str, default='mdan',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    parser.add_argument("-f", "--frac", help="Fraction of the supervised training data to be used.",
                        type=float, default=1.0)
    parser.add_argument("-m", "--model", help="Choose a model to train: [mdan]",
                        type=str, default="mdan")
    parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient for the domain adversarial loss",
                        type=float, default=1e-2)
    parser.add_argument("-g", "--gamma", type=float, default=10.0)
    parser.add_argument("-o", "--mode", help="Mode of combination rule for MDANet: [maxmin|dynamic]", type=str,
                        default="dynamic")
    args = parser.parse_args()
    main(args)

