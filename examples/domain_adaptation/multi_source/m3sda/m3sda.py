import random
import warnings
import sys
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torch.nn.functional as F

sys.path.append('../../..')
from dalib.adaptation.multi_source.m3sda import feature_extractor, predictor
import common.vision.datasets as datasets
import common.vision.models as models
from common.vision.transforms import ResizeImage
from common.utils.logger import CompleteLogger
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
    N = len(dataset.domains())

    if args.sources is None:
        args.sources = dataset.domains()
        args.sources.remove(args.target)

        source_domain = args.sources
        target_domain = args.target
        BATCH_SIZE = args.batch_size
        os.makedirs('./model/' + target_domain, exist_ok=True)

    print("Source: {} Target: {}".format(args.sources, args.target))

    # dataloader
    source_dataloader_list = []
    source_clf = {}
    source_loss = {}

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True).to(device)
    extractor = feature_extractor(backbone).to(device)
    # extractor.load_state_dict(torch.load('./model/2'+target_domain+'/extractor_5.pth'))
    extractor_optim = optim.Adam(extractor.parameters(), lr=3e-4)
    min_ = float('inf')

    train_target_dataset = dataset(root=args.root, task=args.target, download=True, transform=train_transform)
    train_target_loader = DataLoader(train_target_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_dataset = dataset(root=args.root, task=args.target, download=True, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    global num_classes
    num_classes = val_dataset.num_classes

    if args.data == 'DomainNet':
        test_dataset = dataset(root=args.root, task=args.target, split='test', download=True, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = val_loader

    for source in source_domain:
        train_source_dataset = dataset(root=args.root, task=source, download=True, transform=train_transform)
        train_source_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=args.workers, drop_last=True)

        source_dataloader_list.append(train_source_loader)

        # c1 : for target
        # c2 : for source
        source_clf[source] = {}
        source_clf[source]['c1'] = predictor(num_classes).to(device)
        source_clf[source]['c2'] = predictor(num_classes).to(device)
        # source_clf[source]['c1'].load_state_dict(torch.load('./model/2'+target_domain+'/'+source+'_c1_5.pth'))
        # source_clf[source]['c2'].load_state_dict(torch.load('./model/2'+target_domain+'/'+source+'_c2_5.pth'))
        source_clf[source]['optim'] = optim.Adam(
            list(source_clf[source]['c1'].parameters()) + list(source_clf[source]['c2'].parameters()), lr=3e-4)

    if len(train_target_loader) < min_:
        min_ = len(train_target_loader)

    loss_extractor = nn.CrossEntropyLoss()
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()

    for source in source_domain:
        source_loss[source] = {}
        for i in range(1, 3):
            source_loss[source][str(i)] = {}
            source_loss[source][str(i)]['loss'] = []
            source_loss[source][str(i)]['ac'] = []

    mcd_loss_plot = []
    dis_loss_plot = []

    # start training
    best_acc1 = 0.

    EP = args.epochs
    for ep in range(EP):
        print('\n')
        print('epoch ', ep)

        extractor.train()

        source_ac = {}
        for source in source_domain:
            source_clf[source]['c1'] = source_clf[source]['c1'].train()
            source_clf[source]['c2'] = source_clf[source]['c2'].train()
            source_ac[source] = defaultdict(int)

        record = {}
        for source in source_domain:
            record[source] = {}
            for i in range(1, 3):
                record[source][str(i)] = 0
        mcd_loss = 0
        dis_loss = 0

        weights = [0, 0, 0]
        for batch_index, (src_batch, tar_batch) in enumerate(zip(zip(*source_dataloader_list), train_target_loader)):

            src_len = len(src_batch)
            loss_cls = 0

            # train extractor and source clssifier
            for index, batch in enumerate(src_batch):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y = y.view(-1)

                feature = extractor(x)

                # pred1: for target
                # pred2: for source
                pred1 = source_clf[source_domain[index]]['c1'](feature)
                pred2 = source_clf[source_domain[index]]['c2'](feature)

                source_ac[source_domain[index]]['c1'] += torch.sum(torch.max(pred1, dim=1)[1] == y).item()
                source_ac[source_domain[index]]['c2'] += torch.sum(torch.max(pred2, dim=1)[1] == y).item()
                loss1 = loss_extractor(pred1, y)
                loss2 = loss_extractor(pred2, y)
                loss_cls += loss1 + loss2
                record[source_domain[index]]['1'] += loss1.item()
                record[source_domain[index]]['2'] += loss2.item()

            if batch_index % 10 == 0:
                for source in source_domain:
                    print(source)
                    print('c1 : [%.8f]' % (source_ac[source]['c1'] / (batch_index + 1) / BATCH_SIZE))
                    print('c2 : [%.8f]' % (source_ac[source]['c2'] / (batch_index + 1) / BATCH_SIZE))
                    weights[index] = max([source_ac[source]['c1'], source_ac[source]['c2']])
                    # print('\n')

            m1_loss = 0
            m2_loss = 0
            for k in range(1, 3):
                for i_index, batch in enumerate(src_batch):
                    x, y = batch
                    x = x.to(device)
                    y = y.to(device)
                    y = y.view(-1)

                    tar_x, _ = tar_batch
                    tar_x = tar_x.to(device)

                    src_feature = extractor(x)
                    tar_feature = extractor(tar_x)

                    e_src = torch.mean(src_feature ** k, dim=0)
                    e_tar = torch.mean(tar_feature ** k, dim=0)
                    m1_dist = e_src.dist(e_tar)
                    m1_loss += m1_dist
                    for j_index, other_batch in enumerate(src_batch[i_index + 1:]):
                        other_x, other_y = other_batch
                        other_x = other_x.to(device)
                        other_y = other_y.to(device)
                        other_y = other_y.view(-1)
                        other_feature = extractor(other_x)

                        e_other = torch.mean(other_feature ** k, dim=0)
                        m2_dist = e_src.dist(e_other)
                        m2_loss += m2_dist

            loss_m = (EP - ep) / EP * (m1_loss / N + m2_loss / N / (N - 1) * 2) * 0.8
            mcd_loss += loss_m.item()

            loss = loss_cls + loss_m

            if batch_index % 10 == 0:
                print('[%d]/[%d]' % (batch_index, min_))
                print('class loss : [%.5f]' % (loss_cls))
                print('msd loss : [%.5f]' % (loss_m))

            extractor_optim.zero_grad()
            for source in source_domain:
                source_clf[source]['optim'].zero_grad()

            loss.backward()

            extractor_optim.step()

            for source in source_domain:
                source_clf[source]['optim'].step()
                source_clf[source]['optim'].zero_grad()

            extractor_optim.zero_grad()

            tar_x, _ = tar_batch
            tar_x = tar_x.to(device)
            tar_feature = extractor(tar_x)
            loss = 0
            d_loss = 0
            c_loss = 0
            for index, batch in enumerate(src_batch):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y = y.view(-1)

                feature = extractor(x)

                pred1 = source_clf[source_domain[index]]['c1'](feature)
                pred2 = source_clf[source_domain[index]]['c2'](feature)

                c_loss += loss_extractor(pred1, y) + loss_extractor(pred2, y)

                pred_c1 = source_clf[source_domain[index]]['c1'](tar_feature)
                pred_c2 = source_clf[source_domain[index]]['c2'](tar_feature)
                combine1 = (F.softmax(pred_c1, dim=1) + F.softmax(pred_c2, dim=1)) / 2

                d_loss += loss_l1(pred_c1, pred_c2)

                for index_2, o_batch in enumerate(src_batch[index + 1:]):
                    pred_2_c1 = source_clf[source_domain[index_2 + index]]['c1'](tar_feature)
                    pred_2_c2 = source_clf[source_domain[index_2 + index]]['c2'](tar_feature)
                    combine2 = (F.softmax(pred_2_c1, dim=1) + F.softmax(pred_2_c2, dim=1)) / 2

                    d_loss += loss_l1(combine1, combine2) * 0.1

                # discrepency_loss = torch.mean(torch.sum(abs(F.softmax(pred_c1, dim=1) - F.softmax(pred_c2, dim=1)), dim=1))
            # discrepency_loss = loss_l1(F.softmax(pred_c1, dim=1), F.softmax(pred_c2, dim=1))

            # loss += clf_loss - discrepency_loss
            loss = c_loss - d_loss

            loss.backward()
            extractor_optim.zero_grad()
            for source in source_domain:
                source_clf[source]['optim'].zero_grad()

            for source in source_domain:
                source_clf[source]['optim'].step()

            for source in source_domain:
                source_clf[source]['optim'].zero_grad()

            all_dis = 0
            for i in range(3):
                discrepency_loss = 0
                tar_feature = extractor(tar_x)

                for index, _ in enumerate(src_batch):

                    pred_c1 = source_clf[source_domain[index]]['c1'](tar_feature)
                    pred_c2 = source_clf[source_domain[index]]['c2'](tar_feature)
                    combine1 = (F.softmax(pred_c1, dim=1) + F.softmax(pred_c2, dim=1)) / 2

                    discrepency_loss += loss_l1(pred_c1, pred_c2)

                    for index2, _ in enumerate(src_batch[index + 1:]):
                        pred_2_c1 = source_clf[source_domain[index2 + index]]['c1'](tar_feature)
                        pred_2_c2 = source_clf[source_domain[index2 + index]]['c2'](tar_feature)
                        combine2 = (F.softmax(pred_2_c1, dim=1) + F.softmax(pred_2_c2, dim=1)) / 2

                        discrepency_loss += loss_l1(combine1, combine2) * 0.1
                    # discrepency_loss += torch.mean(torch.sum(abs(F.softmax(pred_c1, dim=1) - F.softmax(pred_c2, dim=1)), dim=1))
                    # discrepency_loss += loss_l1(F.softmax(pred_c1, dim=1), F.softmax(pred_c2, dim=1))

                all_dis += discrepency_loss.item()

                extractor_optim.zero_grad()

                for source in source_domain:
                    source_clf[source]['optim'].zero_grad()

                discrepency_loss.backward()

                extractor_optim.step()
                extractor_optim.zero_grad()

                for source in source_domain:
                    source_clf[source]['optim'].zero_grad()

            dis_loss += all_dis

            if batch_index % 10 == 0:
                print('Discrepency Loss : [%.4f]' % (all_dis))

        ###

        for source in source_domain:
            for i in range(1, 3):
                source_loss[source][str(i)]['loss'].append(record[source][str(i)] / min_ / BATCH_SIZE)
                source_loss[source][str(i)]['ac'].append(source_ac[source]['c' + str(i)] / min_ / BATCH_SIZE)
        dis_loss_plot.append(dis_loss / min_ / BATCH_SIZE)
        mcd_loss_plot.append(mcd_loss / min_ / BATCH_SIZE)

        ### validation
        extractor.eval()
        for source in source_domain:
            source_clf[source]['c1'] = source_clf[source]['c1'].eval()
            source_clf[source]['c2'] = source_clf[source]['c2'].eval()

        source_ac = {}
        eval_loss = {}

        for source in source_domain:
            source_ac[source] = defaultdict(int)
            eval_loss[source] = defaultdict(int)

        fianl_ac = 0
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y = y.view(-1)

                feature = extractor(x)
                final_pred = 1
                for index_s, source in enumerate(source_domain):
                    pred1 = source_clf[source]['c1'](feature)
                    pred2 = source_clf[source]['c2'](feature)

                    eval_loss[source]['c1'] += loss_extractor(pred1, y)
                    eval_loss[source]['c2'] += loss_extractor(pred2, y)

                    if isinstance(final_pred, int):
                        final_pred = (F.softmax(pred1, dim=1) + F.softmax(pred2, dim=1)) / 2
                    else:
                        final_pred += (F.softmax(pred1, dim=1) + F.softmax(pred2, dim=1)) / 2

                    source_ac[source]['c1'] += np.sum(
                        np.argmax(pred1.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())
                    source_ac[source]['c2'] += np.sum(
                        np.argmax(pred2.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

                fianl_ac += np.sum(np.argmax(final_pred.cpu().detach().numpy(), axis=1) == y.cpu().detach().numpy())

        for source in source_domain:
            print('Current Source : ', source)
            print('Accuray for c1 : [%.4f]' % (source_ac[source]['c1'] / BATCH_SIZE / len(val_loader)))
            print('Accuray for c2 : [%.4f]' % (source_ac[source]['c2'] / BATCH_SIZE / len(val_loader)))
            print('eval loss c1 : [%.4f]' % (eval_loss[source]['c1']))
            print('eval loss c2 : [%.4f]' % (eval_loss[source]['c2']))

        combine_acc = fianl_ac / BATCH_SIZE / len(val_loader)

        best_acc1 = max(combine_acc, best_acc1)

        print('Combine Ac : [%.4f]' % combine_acc)

        torch.save(extractor.state_dict(), './model/' + target_domain + '/extractor' + '_' + str(ep) + '.pth')
        for source in source_domain:
            torch.save(source_clf[source]['c1'].state_dict(),
                       './model/' + target_domain + '/' + source + '_c1_' + str(ep) + '.pth')
            torch.save(source_clf[source]['c2'].state_dict(),
                       './model/' + target_domain + '/' + source + '_c2_' + str(ep) + '.pth')

    ###
    print("best acc", best_acc1)

    ep_list = [i for i in range(EP)]

    for source in source_domain:
        for i in range(1, 3):
            plt.plot(ep_list, source_loss[source][str(i)]['loss'], label='loss-c' + str(i))
            plt.plot(ep_list, source_loss[source][str(i)]['ac'], label='ac-c' + str(i))
            plt.title(source)
            plt.xlabel('EP')
        plt.legend()
        plt.savefig('./model/' + target_domain + '/' + source + '_loss_ac.jpg')
        plt.show()

    plt.plot(ep_list, dis_loss_plot, label='discrepency loss')
    plt.legend()
    plt.title('discrepency loss')
    plt.xlabel('EP')
    plt.ylabel('loss')
    plt.savefig('./model/' + target_domain + '/discrepency_loss.jpg')
    plt.show()

    plt.plot(ep_list, mcd_loss_plot, label='mc loss')
    plt.legend()
    plt.title('MC loss')
    plt.xlabel('EP')
    plt.ylabel('loss')
    plt.savefig('./model/' + target_domain + '/mc_loss.jpg')
    plt.show()


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
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
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
    parser.add_argument("--log", type=str, default='m3sda',
                        help="Where to save logs, checkpoints and debugging images.")
    parser.add_argument("--phase", type=str, default='train', choices=['train', 'test', 'analysis'],
                        help="When phase is 'test', only test the model."
                             "When phase is 'analysis', only analysis the model.")
    args = parser.parse_args()
    main(args)
