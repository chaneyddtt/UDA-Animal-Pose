# loading a trained model and test on scut dataset

from __future__ import print_function, absolute_import

import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import _init_paths
from pose import Bar
from pose.utils.logger import Logger, savefig
from pose.utils.evaluation import accuracy_2animal, AverageMeter, final_preds, calc_metrics_2animal, accuracy
from pose.utils.misc import save_checkpoint, save_pred, adjust_learning_rate
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.imutils import batch_with_heatmap
from pose.utils.transforms import fliplr, flip_back, transform
import pose.models as models
import pose.datasets as datasets
import pose.losses as losses
import cv2
import numpy as np
from pose.utils.imutils import im_to_numpy
from pose.utils.utils_vis import cv2_visualize_keypoints

# get model names and dataset names
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

dataset_names = sorted(name for name in datasets.__dict__
                       if name.islower() and not name.startswith("__")
                       and callable(datasets.__dict__[name]))

# init global variables
best_acc = 0
idx1 = []
idx2 = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # There is BN issue for early version of PyTorch


# see https://github.com/bearpaw/pytorch-pose/issues/33

def main(args):
    global best_acc
    global idx1
    global idx2

    # idx is the index of joints used to compute accuracy for dataset2

    idx1 = range(1, 19)
    idx2 = range(1, 19)  # horse

    # create model
    njoints = datasets.__dict__[args.dataset].njoints
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](
                                       num_classes=njoints,
                                       resnet_layers=args.resnet_layers,
                                       pretrained=None,
                                       dual_branch=True
                                       )

    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = losses.JointsMSELoss().to(device)

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict_ema'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        raise Exception('please provide a checkpoint')

    val_dataset = datasets.__dict__[args.dataset](is_train=False, is_aug=False, **vars(args))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    _, acc, predictions = validate(val_loader, model, criterion, njoints,
                                   args, args.flip, args.test_batch)
    return


def validate(val_loader, model, criterion, num_classes, args, flip=False, test_batch=6):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    acces = AverageMeter()

    pck_score = np.zeros(num_classes)
    pck_count = np.zeros(num_classes)

    # predictions
    predictions = torch.Tensor(val_loader.dataset.__len__(), num_classes, 2)

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (input, target, meta) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output, output_refine = model(input, 1, return_domain=False)
            score_map = output_refine[0].cpu()

            if flip:
                flip_input = torch.from_numpy(fliplr(input.clone().cpu().numpy())).float().to(device)
                _, flip_output_refine = model(flip_input, 1, return_domain=False)
                flip_output = flip_output_refine[0].cpu()
                flip_output = flip_back(flip_output, 'real_animal')
                score_map += flip_output

            acc, _ = accuracy_2animal(score_map, target.cpu(), idx1, idx2)
            # cal per joint PCK@0.05
            for j in range(num_classes):
                if acc[j + 1] > -1:
                    pck_score[j] += acc[j + 1].numpy()
                    pck_count[j] += 1

            # generate predictions
            preds = final_preds(score_map, meta['center'], meta['scale'], [64, 64])

            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure accuracy and record loss
            acces.update(acc[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Acc: {acc: .8f}'.format(
                batch=i + 1,
                size=len(val_loader),
                data=data_time.val,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                acc=acces.avg
            )
            bar.next()

        bar.finish()

    for j in range(num_classes):
        pck_score[j] /= float(pck_count[j])
    print("\nper joint PCK@0.05:")
    print('Animal: {}, total number of joints: {}'.format(args.animal, pck_count.sum()))
    print(list(pck_score))

    parts = {'eye': [0, 1], 'chin': [2], 'hoof': [3, 4, 5, 6], 'hip': [7], 'knee': [8, 9, 10, 11], 'shoulder': [12, 13],
             'elbow': [14, 15, 16, 17]}
    for p in parts.keys():
        part = parts[p]
        score = 0.
        count = 0.
        for joint in part:
            score += pck_score[joint] * pck_count[joint]
            count += pck_count[joint]
        print('\n Joint {}: {} '.format(p, score/count))

    return _, acces.avg, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--dataset', metavar='DATASET', default='real_animal_all',
                        choices=dataset_names,
                        help='Datasets: ' +
                             ' | '.join(dataset_names) +
                             ' (default: real_animal)')
    parser.add_argument('--image-path', default='./animal_data/', type=str,
                        help='path to images')
    parser.add_argument('--animal', default='horse', type=str,
                        help='animal to test')
    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')
    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                        help='output resolution (default: 64, to gen GT)')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='hg',
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
    parser.add_argument('--target-weight', dest='target_weight',
                        action='store_true',
                        help='Loss with target_weight')
    # Data processing
    parser.add_argument('-f', '--flip', dest='flip', action='store_true',
                        help='flip the input during validation')
    parser.add_argument('--sigma', type=float, default=1,
                        help='Groundtruth Gaussian sigma.')
    parser.add_argument('--scale-factor', type=float, default=0.4,
                        help='Scale factor (data aug).')
    parser.add_argument('--rot-factor', type=float, default=45,
                        help='Rotation factor (data aug).')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--label-type', metavar='LABELTYPE', default='Gaussian',
                        choices=['Gaussian', 'Cauchy'],
                        help='Labelmap dist type: (default=Gaussian)')
    # Miscs
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
