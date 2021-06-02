from __future__ import print_function, absolute_import

import argparse
import time
import random

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

import _init_paths
from pose import Bar
from pose.utils.logger import Logger
from pose.utils.evaluation import accuracy, AverageMeter
from pose.utils.misc import save_checkpoint, adjust_learning_rate_main
from pose.utils.osutils import mkdir_p, isfile, isdir, join
from pose.utils.transforms import fliplr, flip_back

from pose.utils.utils_mt import get_current_consistency_weight, update_ema_variables, get_current_target_weight
import pose.models as models
import pose.datasets as datasets
import pose.losses as Losses
from pose.models.refinenet_multilayer_da import init_pretrained
from CCSSL.scripts.timer import Timer
from CCSSL.scripts.consistency import prediction_check
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


dataset_names = sorted(name for name in datasets.__dict__
    if name.islower() and not name.startswith("__")
    and callable(datasets.__dict__[name]))

best_acc = 0
best_epoch_acc = 0
global_step = 0
idx = []

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def main(args):
    global best_acc
    global best_epoch_acc
    global idx
    global global_step

    _t = {'iter time': Timer()}

    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    title = args.dataset + ' ' + args.arch
    logger = Logger(join(args.checkpoint, 'log.txt'), title=title)
    logger.log_arguments(args)
    logger.set_names(['Epoch', 'LR', 'Stu Val Acc', 'Tea Val Acc'])
    njoints = datasets.__dict__[args.dataset].njoints

    # idx is the index of joints used to compute accuracy
    idx = range(1, njoints+1)

    # create model
    print("==> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](num_classes=njoints,
                                       resnet_layers=args.resnet_layers,
                                       pretrained=args.pretrained,
                                       dual_branch=args.dual_branch)
    model_ema = models.__dict__[args.arch](num_classes=njoints,
                                           resnet_layers=args.resnet_layers,
                                           pretrained=args.pretrained,
                                           dual_branch=args.dual_branch)
    for param in model_ema.parameters():
        param.detach()

    model = torch.nn.DataParallel(model).to(device)
    model_ema = torch.nn.DataParallel(model_ema).to(device)
    criterion = Losses.JointsMSELoss().to(device)
    criterion_oekm = Losses.CurriculumLoss().to(device)

    bce_loss = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.Adam(
                                model.parameters(),
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    # resume model
    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.load_state_dict(checkpoint['state_dict_ema'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # load pretrained model
    elif args.pretrained:
        if isfile(args.pretrained):
            init_pretrained(model, args.pretrained)
        else:
            print("=> no pretrained model found at '{}'".format(args.pretrained))
    else:
        print("Training from sctrach")

    print('    Total params in one pose model : %.2fM'
          % (sum(p.numel() for p in model.parameters())/1000000.0))

    # load both source and target domain dataset
    train_dataset = datasets.__dict__[args.dataset](is_train=True, **vars(args))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, drop_last=True
    )

    real_dataset_train = datasets.__dict__[args.dataset_real](is_train=True, is_aug=False, **vars(args))
    real_loader_train = torch.utils.data.DataLoader(
        real_dataset_train,
        batch_size=args.train_batch, shuffle=False,
        num_workers=args.workers
    )
    real_dataset_valid = datasets.__dict__[args.dataset_real](is_train=False, is_aug=False, **vars(args))
    real_loader_valid = torch.utils.data.DataLoader(
        real_dataset_valid,
        batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers
    )

    # evaluate model
    if args.evaluate:
        print('\nEvaluation only')

        loss, acc = validate(real_loader_valid, model, criterion,
                                                    args.debug, args.flip, args.test_batch, njoints)
        return

    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate_main(optimizer, epoch, args)
        # compute the number of joints selected from each image, gradually drop more and more joints with large loss
        topk = njoints - (max(epoch - args.start_mine_epoch, 0) // args.reduce_interval)
        topk = int(max(topk, args.min_kpts))

        # gradually decrease the weight for initial pseudo labels
        target_weight = get_current_target_weight(args, epoch)

        # gradually increase the weight for self-distillation loss, namely updated pseudo labels
        c2rconsistency_weight = get_current_consistency_weight(args, epoch)
        print('\nEpoch: %d | LR: %.6f | Trg_weight: %.6f | C2rcons_weight: %.6f' % (epoch + 1, lr,
                                                                  target_weight, c2rconsistency_weight))
        # generate pseudo labels using consistency check
        if epoch == args.start_epoch:
            if args.generate_pseudol:
                model.eval()
                for animal in ['horse', 'tiger']:
                    # switch animal to single category to generate pseudo labels separately
                    args.animal = animal
                    real_dataset_train = datasets.__dict__[args.dataset_real](is_train=True, is_aug=False, **vars(args))
                    real_loader_train = torch.utils.data.DataLoader(
                        real_dataset_train,
                        batch_size=args.train_batch, shuffle=False,
                        num_workers=args.workers
                    )
                    ssl_kpts = {}
                    acces1 = AverageMeter()
                    previous_img = None
                    previous_kpts = None
                    for _, (trg_img, trg_lbl, trg_meta) in enumerate(real_loader_train):
                        trg_img = trg_img.to(device)
                        trg_lbl = trg_lbl.to(device, non_blocking=True)
                        for i in range(trg_img.size(0)):
                            score_map, generated_kpts = prediction_check(previous_img, previous_kpts, trg_img[i], model, real_dataset_train, device, num_transform=5)
                            ssl_kpts[int(trg_meta['index'][i].cpu().numpy().astype(np.int32))] = generated_kpts
                            acc1, _ = accuracy(score_map, trg_lbl[i].cpu().unsqueeze(0), idx)
                            acces1.update(acc1[0], 1)
                            previous_img = trg_img[i]
                            previous_kpts = generated_kpts
                    print('Acc on target {} training set (psedo-labels): {}'.format(args.animal, acces1.avg))
                    np.save('./animal_data/psudo_labels/stage1/all/ssl_labels_train_{}.npy'.format(args.animal), ssl_kpts)
                break
            # construct dataloader based on generated pseudolabels, switch animal to 'all' to train on all categories
            args.animal = 'all'
            real_dataset_train = datasets.__dict__[args.dataset_real_crop](is_train=True, is_aug=True, **vars(args))
            real_loader_train = torch.utils.data.DataLoader(
                real_dataset_train,
                batch_size=args.train_batch, shuffle=True,
                num_workers=args.workers,
                drop_last=True
            )
            print("======> start training")

        loss_src_log = AverageMeter()
        loss_trg_log = AverageMeter()
        loss_cons_log = AverageMeter()
        joint_loader = zip(train_loader, real_loader_train)
        num_iter = len(real_loader_train) if len(train_loader) > len(real_loader_train) else len(train_loader)
        model.train()
        model_ema.train()
        bar = Bar('Train ', max=num_iter)

        for i, ((src_img, src_lbl, src_meta), (trg_img, trg_img_t, trg_img_s, trg_lbl, trg_lbl_t, trg_lbl_s, trg_meta)) \
                in enumerate(joint_loader):
            # gradually increase the magnitude of grl as the discriminator gets better and better
            p = float(i + epoch * num_iter) / args.epochs / num_iter
            lambda_ = 2. / (1. + np.exp(-10 * p)) - 1

            optimizer.zero_grad()
            src_img, src_lbl, src_weight = src_img.to(device), src_lbl.to(device, non_blocking=True), src_meta[
                'target_weight'].to(device, non_blocking=True)
            trg_img,  trg_lbl = trg_img.to(device), trg_lbl.to(device, non_blocking=True)
            warpmat, trg_weight = trg_meta['warpmat'].to(device, non_blocking=True), trg_meta['target_weight'].to(device, non_blocking=True)
            trg_img_t, trg_img_s, trg_lbl_t, trg_lbl_s = trg_img_t.to(device), trg_img_s.to(device), trg_lbl_t.to(device, non_blocking=True), \
                                                         trg_lbl_s.to(device, non_blocking=True)
            # add mixup between domains
            if args.mixup:
                src_img, src_lbl, src_weight, trg_img, trg_lbl, trg_weight = mixup(src_img, src_lbl, src_weight,
                                                                                   trg_img, trg_lbl, trg_weight,
                                                                                    args.beta)
            # add mixup between domains and within domains, the later is more about balancing horse and tiger data
            if args.mixup_dual:
                if random.random() <= 0.5:
                    src_img, src_lbl, src_weight, trg_img, trg_lbl, trg_weight = mixup(src_img, src_lbl, src_weight,
                                                                                       trg_img, trg_lbl, trg_weight,
                                                                                       args.beta)
                else:
                    trg_img, trg_lbl, trg_weight = mixup_withindomain(trg_img, trg_lbl, trg_weight, args.beta, args.beta)

            src_out, src_out_refine, src_domain_out = model(src_img, lambda_)
            src_kpt_score, src_kpt_score_ = src_out_refine
            loss_kpt_src = criterion(src_out, src_lbl, src_weight, len(idx)) + \
                   criterion(src_kpt_score, src_lbl, src_weight, len(idx))
            loss_src_log.update(loss_kpt_src.item(), src_img.size(0))

            trg_out, trg_out_refine, trg_domain_out = model(trg_img, lambda_)
            trg_kpt_score, trg_kpt_score_ = trg_out_refine

            # loss based on initial pseudo labels
            loss_kpt_trg_coarse = target_weight * criterion_oekm(trg_out, trg_lbl, trg_weight, topk)
            loss_kpt_trg_refine = target_weight * criterion_oekm(trg_kpt_score, trg_lbl, trg_weight, topk)

            loss_kpt_trg = loss_kpt_trg_coarse + loss_kpt_trg_refine
            loss_trg_log.update(loss_kpt_trg.item(), trg_img.size(0))

            # loss for discriminator
            src_label = torch.ones_like(src_domain_out).cuda(device)
            trg_label = torch.zeros_like(trg_domain_out).cuda(device)
            loss_adv_src = args.adv_w * bce_loss(src_domain_out, src_label)/2
            loss_adv_trg = args.adv_w * bce_loss(trg_domain_out, trg_label)/2

            loss_stu = loss_kpt_src + loss_kpt_trg + loss_adv_src + loss_adv_trg
            loss_stu.backward()

            # forward again for student teacher consistency
            model_out_stu, model_out_stu_refine = model(trg_img_s, lambda_, return_domain=False)
            # The teacher network has the same three forward steps although we only use the last one. The reason is to
            # keep the data statistics the same for student and teacher branch, mainly for batchnorm.
            with torch.no_grad():
                _, _ = model_ema(src_img, lambda_, return_domain=False)
                _, _ = model_ema(trg_img, lambda_, return_domain=False)
                model_out_ema, model_out_ema_refine = model_ema(trg_img_t, lambda_, return_domain=False)
            hm1, hm2 = model_out_stu_refine
            hm_ema, _ = model_out_ema_refine

            # self-distillation loss, namely inner loop update
            hm1_clone = hm1.clone().detach().requires_grad_(False)
            weight_const_var = torch.ones_like(trg_weight).to(device)
            c2rconst_loss = c2rconsistency_weight * criterion_oekm(model_out_stu, hm1_clone, weight_const_var,
                                                                  topk)

            # consistency between two heads of the student network, adapted from original mean teacher
            res_loss = args.logit_distance_cost * Losses.symmetric_mse_loss(hm1, hm2)

            # consistency loss between student and teacher network
            flip_var = (trg_meta['flip'] == trg_meta['flip_ema'])
            flip_var = flip_var.to(device)
            # gradually increase the weight for consistency loss, namely outer loop update
            consistency_weight = get_current_consistency_weight(args, epoch)
            # adding transformation to the output of the teacher network
            grid = F.affine_grid(warpmat, hm_ema.size())
            hm_ema_trans = F.grid_sample(hm_ema, grid)
            hm_ema_trans_flip = hm_ema_trans.clone()
            hm_ema_trans_flip = flip_back(hm_ema_trans_flip.cpu(), 'real_animal')
            hm_ema_trans_flip = hm_ema_trans_flip.to(device)
            hm_ema_trans = torch.where(flip_var[:, None, None, None], hm_ema_trans, hm_ema_trans_flip)
            weight_const_var = torch.ones_like(trg_weight).to(device)
            consistency_loss = consistency_weight * criterion(hm2, hm_ema_trans, weight_const_var, len(idx))
            loss_cons_log.update(consistency_loss.item(), trg_img_t.size(0))

            loss_mt = consistency_loss + res_loss + c2rconst_loss
            loss_mt.backward()
            optimizer.step()
            global_step += 1
            # update parameters for teacher network
            update_ema_variables(model, model_ema, args.ema_decay, global_step)

            bar.suffix = '({batch}/{size}) Total: {total:} | ETA: {eta:} | Loss_cons: {loss_cons:.8f}s' \
                         '| Loss_src: {loss_src:.8f} | Loss_trg: {loss_trg:.8f}'.format(
                            batch=i + 1,
                            size=num_iter,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss_cons=loss_cons_log.avg,
                            loss_src=loss_src_log.avg,
                            loss_trg=loss_trg_log.avg
                        )
            bar.next()
        bar.finish()

        _, trg_val_acc_s = validate(real_loader_valid, model, criterion,
                                                  args.debug, args.flip, args.test_batch, njoints)
        _, trg_val_acc_t = validate(real_loader_valid, model_ema, criterion,
                                                  args.debug, args.flip, args.test_batch, njoints)
        logger.append([epoch + 1, lr, trg_val_acc_s, trg_val_acc_t])

        trg_val_acc = trg_val_acc_t
        is_best = trg_val_acc > best_acc
        if trg_val_acc > best_acc:
            best_epoch_acc = epoch + 1
        best_acc = max(trg_val_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'state_dict_ema': model_ema.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict()
        }, is_best, checkpoint=args.checkpoint, snapshot=args.snapshot)

    print(best_epoch_acc, best_acc)
    logger.close()


def validate(val_loader, model, criterion, debug=False, flip=True, test_batch=6, njoints=18):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_re = AverageMeter()
    acces_re = AverageMeter()

    model.eval()
    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))

    with torch.no_grad():
        for i, (input, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_weight = meta['target_weight'].to(device, non_blocking=True)

            model_out, model_out_refine = model(input, 1, return_domain=False)
            hm1, hm2 = model_out_refine
            score_map = model_out.cpu()
            score_map_refine = hm1.cpu()

            if flip:
                flip_input = torch.from_numpy(fliplr(input.clone().cpu().numpy())).float().to(device)
                flip_output, flip_out_refine = model(flip_input, 1, return_domain=False)
                hm1_flip, hm2_flip = flip_out_refine
                flip_output = flip_back(flip_output.cpu(), 'real_animal')
                flip_out_refine = flip_back(hm1_flip.cpu(), 'real_animal')
                score_map += flip_output
                score_map_refine += flip_out_refine

            loss_re = criterion(hm1, target, target_weight, len(idx))
            acc_re, _ = accuracy(score_map_refine, target.cpu(), idx)

            losses_re.update(loss_re.item(), input.size(0))
            acces_re.update(acc_re[0], input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} ' \
                          '| Loss_re: {loss_re:.8f}  | Acc_re: {acc_re: .8f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.val,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss_re=losses_re.avg,
                            acc_re=acces_re.avg
                            )
            bar.next()

        bar.finish()
        return losses_re.avg, acces_re.avg


def mixup(img_src, hm_src, weights_src, img_trg, hm_trg, weights_trg, beta):
    m = torch.distributions.beta.Beta(torch.tensor(beta), torch.tensor(beta))
    mix = m.rsample(sample_shape=(img_src.size(0), 1, 1, 1))
    # keep the max value such that the domain labels does not change
    mix = torch.max(mix, 1 - mix)
    mix = mix.to(device)
    img_src_mix = img_src * mix + img_trg * (1. - mix)
    hm_src_mix = hm_src * mix + hm_trg * (1. - mix)
    img_trg_mix = img_trg * mix + img_src * (1. - mix)
    hm_trg_mix = hm_trg * mix + hm_src * (1. - mix)
    weights = torch.max(weights_src, weights_trg)
    return img_src_mix, hm_src_mix, weights, img_trg_mix, hm_trg_mix, weights


# mixup inside domains, mainly mixup different categories to prevent data unbalance
def mixup_withindomain(trg_img, trg_lbl, trg_weights, beta1, beta2):
    m = torch.distributions.beta.Beta(torch.tensor(beta1), torch.tensor(beta2))
    mix = m.rsample(sample_shape=(trg_img.size(0), 1, 1, 1))
    mix = mix.to(device)
    index = torch.randperm(trg_img.size(0))
    img_perm = trg_img[index]
    hm_perm = trg_lbl[index]
    weights_perm = trg_weights[index]
    img_mix = trg_img * mix + img_perm * (1-mix)
    hm_mix = trg_lbl * mix + hm_perm * (1-mix)
    weights_mix = torch.max(trg_weights, weights_perm)
    return img_mix, hm_mix, weights_mix


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # Dataset setting
    parser.add_argument('--dataset', metavar='DATASET', default='synthetic_animal_sp_all',
                        choices=dataset_names,
                        help='Datasets: ' +
                            ' | '.join(dataset_names) +
                            ' (default: synthetic_animal_sp)')
    parser.add_argument('--dataset_real', default='real_animal_all', type=str)
    parser.add_argument('--dataset_real_crop', default='real_animal_crop_all', type=str)
    parser.add_argument('--image-path', default='./animal_data/', type=str,
                       help='path to images')
    parser.add_argument('--animal', default='all', type=str,
                       help='horse | tiger | sheep | hound | elephant')
    parser.add_argument('--year', default=2014, type=int, metavar='N',
                        help='year of coco dataset: 2014 (default) | 2017)')
    parser.add_argument('--inp-res', default=256, type=int,
                        help='input resolution (default: 256)')
    parser.add_argument('--out-res', default=64, type=int,
                    help='output resolution (default: 64, to gen GT)')

    # Model structure
    parser.add_argument('--arch', '-a', metavar='ARCH', default='pose_resnet_refine_mt_multida',
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
    parser.add_argument('--epochs', default=80, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--max_epoch', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--train-batch', default=6, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                        help='test batchsize')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--power', default=0.9, type=float, help='power for learning rate decay')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')

    # discriminator
    parser.add_argument("--adv_w", type=float, default=0.0005, help="target dataset loss coefficient")

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
    parser.add_argument('--percentage', type=float, default=0.6,
                        help='Percentage of data to be filtered out.')
    parser.add_argument('--stage', type=str, default='1', help='which stage to load psudo label ')
    parser.add_argument('--train_on_all_cat', action='store_true', help='whether train on all categories')
    parser.add_argument('--generate_pseudol', action='store_true', help='whether generate pseudo labels')

    # Miscs
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--snapshot', default=0, type=int,
                        help='save models for every #snapshot epochs (default: 0)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show intermediate results')
    # MT
    parser.add_argument('--dual_branch', action='store_true', help='whehter has two branches in refinenet')
    parser.add_argument('--logit-distance-cost', default=0.01, type=float, metavar='WEIGHT')
    parser.add_argument('--ema-decay', default=0.999, type=float, metavar='ALPHA',
                        help='ema variable decay rate (default: 0.999)')
    parser.add_argument('--consistency', default=90.0, type=float, metavar='WEIGHT',
                        help='use consistency loss with given weight (default: None)')
    parser.add_argument('--consistency-rampup', default=10, type=int, metavar='EPOCHS',
                        help='length of the consistency loss ramp-up')
    parser.add_argument('--occlusion_aug', action='store_true', help='whether add occlusion augment')
    parser.add_argument('--num_occluder', type=int, default=8, help='number of occluder to add in')

    # initial pseudo labels
    parser.add_argument("--gamma_", type=float, default=15.0, help="target dataset loss coefficient")
    parser.add_argument('--gamma_rampdown', type=int, default=15)
    parser.add_argument('--min_gamma', type=float, default=8)

    # online reliable points mining
    parser.add_argument('--min_kpts', type=int, default=9, help='number of minimum hard keypoints')
    parser.add_argument('--start_mine_epoch', type=int, default=2, help='start epoch to mine ')
    parser.add_argument('--reduce_interval', type=int, default=1, help='start epoch to mine ')

    # mixup
    parser.add_argument('--mixup', action='store_true', help='whehter use mixup')
    parser.add_argument('--mixup_dual', action='store_true', help='')
    parser.add_argument("--beta", type=float, default=0.8, help="target dataset loss coefficient")

    main(parser.parse_args())
