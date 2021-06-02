from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import _init_paths
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy, AverageMeter, final_preds,  calc_metrics, get_preds
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back
from pose.utils.utils import print_options
import pose.models as models
import pose.datasets as datasets
import pose.losses as Losses


# get model names and dataset names
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))


# init global variables
best_acc = 0
best_epoch_acc = 0
idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch
                        # see https://github.com/bearpaw/pytorch-pose/issues/33


def main(args):
    global best_acc
    global best_epoch_acc
    global idx

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    print_options(args)

    njoints = 18
    # idx is the index of joints used to compute accuracy
    idx = range(1, njoints+1)

    # create model
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=njoints,
                                       resnet_layers=args.resnet_layers)

    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = Losses.JointsMSELoss().to(device)

    if args.solver == 'rms':
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr
        )
    else:
        print('Unknown solver: {}'.format(args.solver))
        assert False

    # optionally resume from a checkpoint
    title = args.dataset + ' ' + args.arch
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            logger = Logger(join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:

        logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'LR', 'Train Loss', 'Val Loss', 'Train Acc',  'Val Acc', 'Real Val Acc'])

    print('    Total params: %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    # create data loader
    train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers
    )

    val_dataset = datasets.__dict__[args.dataset](is_train=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers
    )

    val_dataset2 = datasets.__dict__[args.dataset_real](is_train=False, **vars(args))
    val_loader2 = torch.utils.data.DataLoader(
        val_dataset2,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers
    )

    # evaluation only
    if args.evaluate:
        print('\nEvaluation only')
        loss, acc = validate(val_loader2, model, criterion, args.debug, args.flip, args.test_batch, njoints)
        return

    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            train_loader.dataset.sigma *= args.sigma_decay
            val_loader.dataset.sigma *= args.sigma_decay

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, args,
                                      args.debug, args.flip, args.train_batch, epoch, njoints)

        # evaluate on validation set
        valid_loss, valid_acc = validate(val_loader, model, criterion,
                                                  args.debug, args.flip, args.test_batch, njoints)

        _, valid_acc2 = validate(val_loader2, model, criterion,
                                                  args.debug, args.flip, args.test_batch, njoints)

        # append logger file
        logger.append([epoch + 1, lr, train_loss, valid_loss, train_acc, valid_acc, valid_acc2])

        # remember best acc and save checkpoint
        is_best = valid_acc2 > best_acc

        if valid_acc2 > best_acc:
            best_epoch_acc = epoch

        best_acc = max(best_acc, valid_acc2)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)

    print(best_epoch_acc, best_acc)
    logger.close()


def train(train_loader, model, criterion, optimizer, args, debug=False, flip=True, train_batch=6, epoch=0, njoints=18):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces_re = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    gt_win, pred_win = None, None
    bar = Bar('Train', max=len(train_loader))

    for i, (input, target, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device, non_blocking=True)
        target_weight = meta['target_weight'].to(device, non_blocking=True)

        # compute output

        output, output_re = model(input)

        if type(output) == list:  # multiple output
            loss = 0
            for (o, o_re) in zip(output, output_re):
                loss = loss + criterion(o, target, target_weight, len(idx)) + criterion(o_re, target, target_weight, len(idx))
                if args.self_kd:
                    soft_label = o_re.clone().detach().requires_grad_(False)
                    loss = loss + args.soft_weight * criterion(o, soft_label)
            output = output[-1]
            output_re = output_re[-1]
        else:  # single output
            loss = criterion(output, target, target_weight, len(idx)) + criterion(output_re, target, target_weight, len(idx))

        acc_re, _ = accuracy(output_re, target, idx)
        if debug: # visualize groundtruth and predictions
            gt_batch_img = batch_with_heatmap(input, target)
            pred_batch_img = batch_with_heatmap(input, output)
            if not gt_win or not pred_win:
                ax1 = plt.subplot(121)
                ax1.title.set_text('Groundtruth')
                gt_win = plt.imshow(gt_batch_img)
                ax2 = plt.subplot(122)
                ax2.title.set_text('Prediction')
                pred_win = plt.imshow(pred_batch_img)
            else:
                gt_win.set_data(gt_batch_img)
                pred_win.set_data(pred_batch_img)
            plt.pause(.05)
            plt.draw()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acces_re.update(acc_re[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.8f} ' \
                      '| Acc_re: {acc_re: .8f}'.format(
                        batch=i + 1,
                        size=len(train_loader),
                        data=data_time.val,
                        bt=batch_time.val,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        acc_re=acces_re.avg
                        )
        bar.next()

    bar.finish()

    return losses.avg, acces_re.avg


def validate(val_loader, model, criterion, debug=False, flip=True, test_batch=6, njoints=68):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces_re = AverageMeter()
    # switch to evaluate mode
    model.eval()

    gt_win, pred_win = None, None
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))

    with torch.no_grad():
        for i, (input, target, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = meta['target_weight'].to(device, non_blocking=True)

            # compute output
            output, output_refine = model(input)
            score_map = output[-1].cpu() if type(output) == list else output.cpu()
            score_map_refine = output_refine[-1].cpu() if type(output_refine) == list else output_refine.cpu()
            if flip:
                flip_input = torch.from_numpy(fliplr(input.clone().numpy())).float().to(device)
                flip_output, flip_output_re = model(flip_input)
                flip_output = flip_output[-1].cpu() if type(flip_output) == list else flip_output.cpu()
                flip_output_re = flip_output_re[-1].cpu() if type(flip_output_re) == list else flip_output_re.cpu()
                flip_output = flip_back(flip_output, 'real_animal')
                flip_output_re = flip_back(flip_output_re, 'real_animal')
                score_map += flip_output
                score_map_refine += flip_output_re

            if type(output) == list:  # multiple output
                loss = 0
                for (o, o_re) in (output, output_refine):
                    loss = loss + criterion(o, target, target_weight, len(idx)) + criterion(o_re, target, target_weight, len(idx))
            else:  # single output
                loss = criterion(output, target, target_weight, len(idx)) + criterion(output_refine, target, target_weight, len(idx))

            acc_re, _ = accuracy(score_map_refine, target.cpu(), idx)

            if debug:
                gt_batch_img = batch_with_heatmap(input, target)
                pred_batch_img = batch_with_heatmap(input, score_map)
                if not gt_win or not pred_win:
                    plt.subplot(121)
                    gt_win = plt.imshow(gt_batch_img)
                    plt.subplot(122)
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            acces_re.update(acc_re[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} ' \
                          '| Loss: {loss:.8f} | Acc_re: {acc_re: .8f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.val,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=losses.avg,
                            acc_re=acces_re.avg
                            )
            bar.next()

        bar.finish()

    return losses.avg, acces_re.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='synthetic_animal_sp',
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: synthetic_animal_sp)')
    parser.add_argument('--dataset_real', default='real_animal', type=str)
    parser.add_argument('--image-path', default='./animal_data/', type=str,
                       help='path to images')
    parser.add_argument('--animal', default='horse', type=str,
                       help='horse | tiger | sheep | hound | elephant')
    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')
    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                    help='output resolution (default: 64, to gen GT)')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='pose_resnet_refine',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: hg)')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('--resnet-layers', default=50, type=int, metavar='N',
                        help='Number of resnet layers',
                        choices=[18, 34, 50, 101, 152])
    # Training strategy
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')

    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--scale-factor', type=float, default=0.25,
                        help='Scale factor (data aug).')
    parser.add_argument('--rot-factor', type=float, default=30,
                        help='Rotation factor (data aug).')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')

    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')

    parser.add_argument('--train_on_all_cat', action='store_true', help='whether train on all categories')

    main(parser.parse_args())
