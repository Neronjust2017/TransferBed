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
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset

sys.path.append('../../..')
from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.adaptation.dann import DomainAdversarialLoss, ImageClassifier
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.data import ForeverDataIterator
from common.utils.metric import accuracy, ConfusionMatrix
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.logger import CompleteLogger
from common.utils.analysis import collect_feature, tsne, a_distance


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
    classifier = ImageClassifier(backbone, num_classes, bottleneck_dim=args.bottleneck_dim).to(device)

    # MDAN soft version
    domain_discris = [DomainDiscriminator(in_feature=classifier.features_dim, hidden_size=1024).to(device) for i in range(num_domains-1)]

    # define optimizer and lr scheduler
    paramters = classifier.get_parameters()
    for domain_discri in domain_discris:
        paramters += domain_discri.get_parameters()
    optimizer = SGD(paramters, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = LambdaLR(optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))

    # define loss function
    domain_advs = [DomainAdversarialLoss(domain_discri).to(device) for domain_discri in domain_discris]

    # resume from the best checkpoint
    if args.phase != 'train':
        checkpoint = torch.load(logger.get_checkpoint_path('best'), map_location='cpu')
        classifier.load_state_dict(checkpoint)

    # analysis the model
    # if args.phase == 'analysis':
    #     # extract features from both domains
    #     feature_extractor = nn.Sequential(classifier.backbone, classifier.bottleneck).to(device)
    #     source_feature = collect_feature(train_source_loader, feature_extractor, device)
    #     target_feature = collect_feature(train_target_loader, feature_extractor, device)
    #     # plot t-SNE
    #     tSNE_filename = osp.join(logger.visualize_directory, 'TSNE.png')
    #     tsne.visualize(source_feature, target_feature, tSNE_filename)
    #     print("Saving t-SNE to", tSNE_filename)
    #     # calculate A-distance, which is a measure for distribution discrepancy
    #     A_distance = a_distance.calculate(source_feature, target_feature, device)
    #     print("A-distance =", A_distance)
    #     return

    if args.phase == 'test':
        acc1 = validate(test_loader, classifier, args)
        print(acc1)
        return

    # start training
    best_acc1 = 0.
    for epoch in range(args.epochs):
        # train for one epoch
        train(train_source_iters, train_target_iter, classifier, domain_advs, optimizer,
              lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, classifier, args)

        # remember best acc@1 and save checkpoint
        torch.save(classifier.state_dict(), logger.get_checkpoint_path('latest'))
        if acc1 > best_acc1:
            shutil.copy(logger.get_checkpoint_path('latest'), logger.get_checkpoint_path('best'))
        best_acc1 = max(acc1, best_acc1)

    print("best_acc1 = {:3.1f}".format(best_acc1))

    # evaluate on test set
    classifier.load_state_dict(torch.load(logger.get_checkpoint_path('best')))
    acc1 = validate(test_loader, classifier, args)
    print("test_acc1 = {:3.1f}".format(acc1))

    logger.close()


def train(train_source_iters: list, train_target_iter: ForeverDataIterator,
          model: ImageClassifier, domain_advs: list, optimizer: SGD,
          lr_scheduler: LambdaLR, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':5.2f')
    data_time = AverageMeter('Data', ':5.2f')
    losses = AverageMeter('Loss', ':6.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')
    domain_accs = AverageMeter('Domain Acc', ':3.1f')
    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses, cls_accs, domain_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    for domain_adv in domain_advs:
        domain_adv.train()

    end = time.time()
    for i in range(args.iters_per_epoch):

        x_ss = [next(train_source_iter)[0] for train_source_iter in train_source_iters]
        labels_ss = [next(train_source_iter)[1] for train_source_iter in train_source_iters]

        x_t, _ = next(train_target_iter)

        x_ss = torch.cat(x_ss, dim=0)
        labels_ss = torch.cat(labels_ss, dim=0)

        x_ss = x_ss.to(device)
        x_t = x_t.to(device)
        labels_ss = labels_ss.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        x = torch.cat((x_ss, x_t), dim=0)
        y, f = model(x)
        y = y.chunk(num_domains, dim=0)
        f = f.chunk(num_domains, dim=0)
        labels_ss = labels_ss.chunk(num_domains-1, dim=0)

        y_ss = y[:-1]
        f_ss = f[:-1]

        y_t = y[-1]
        f_t = f[-1]

        transfer_losses = 0
        cls_losses = 0

        for y_s, labels_s, f_s, domain_adv in zip(y_ss, labels_ss, f_ss, domain_advs):
            cls_loss = F.cross_entropy(y_s, labels_s)
            transfer_loss = domain_adv(f_s, f_t)
            transfer_losses += transfer_loss
            cls_losses += cls_loss

        cls_losses /= (num_domains-1)
        transfer_losses /= (num_domains-1)
        # loss = cls_losses + transfer_losses * args.trade_off
        loss = cls_losses + transfer_losses

        domain_acces = [domain_adv.domain_discriminator_accuracy for domain_adv in domain_advs]
        cls_acces = [accuracy(y_s, labels_s)[0] for y_s, labels_s in zip(y_ss, labels_ss)]

        losses.update(loss.item(), x_ss[0].size(0))
        cls_accs.update(torch.mean(torch.Tensor(cls_acces)).item(), x_ss[0].size(0))
        domain_accs.update(torch.mean(torch.Tensor(domain_acces)).item(), x_ss[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
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

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

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
    parser.add_argument('--trade-off', default=1., type=float,
                        help='the trade-off hyper-parameter for transfer loss')
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
    parser.add_argument("--log", type=str, default='dann',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)

