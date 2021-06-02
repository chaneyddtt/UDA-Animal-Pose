from __future__ import print_function, absolute_import

import random
import argparse
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.transforms import *
from pose.utils.utils_augment import Augment

import glob
import imageio
import cv2


def generate_data(percentage, stage, animal_list):
    # select (1-p) percent of original pseudo labels according to the consistent score,
    for animal in animal_list:
        if isfile('./animal_data/psudo_labels/stage{}/all/ssl_labels_train_p{}_{}.npy'.format(stage, percentage, animal)):
            print('File exist')
        else:
            ssl_kpts_data = np.load('./animal_data/psudo_labels/stage{}/all/ssl_labels_train_{}.npy'.format(stage, animal),
                                    allow_pickle=True)
            ssl_kpts = ssl_kpts_data.item()
            sorted_confidence = np.zeros(1)
            for k in ssl_kpts:
                sorted_confidence = np.concatenate((sorted_confidence, ssl_kpts[k][:, 2].reshape(-1)), axis=0)
            sorted_confidence = np.sort(sorted_confidence)
            ccl_thresh = sorted_confidence[int(percentage * sorted_confidence.shape[0])]
            for k in ssl_kpts:
                ssl_kpts[k][:, 2] = (ssl_kpts[k][:, 2] > ccl_thresh).astype(np.float32)
            np.save('./animal_data/psudo_labels/stage{}/all/ssl_labels_train_p{}_{}.npy'.format(stage, percentage, animal),
                    ssl_kpts)


def get_warpmat(r, s):
    # generate transformation matrix based on rotation and scale
    M = cv2.getRotationMatrix2D((32, 32), r, 1 / s)
    warpmat = cv2.invertAffineTransform(M)
    warpmat[:, 2] = 0
    return torch.Tensor(warpmat)


class Real_Animal_Crop_All(data.Dataset):
    def __init__(self, is_train=True, is_aug=True, **kwargs):
        print()
        print("==> load real_animal_crop_mt_all")
        self.animal = ['horse', 'tiger'] if kwargs['animal'] == 'all' else [kwargs['animal']]
        self.nParts = 18
        self.idxs = np.arange(18)
        self.img_folder = kwargs['image_path']
        self.is_train = is_train
        self.is_aug = is_aug
        self.inp_res = kwargs['inp_res']
        self.out_res = kwargs['out_res']
        self.sigma = kwargs['sigma']
        self.scale_factor = kwargs['scale_factor']
        self.rot_factor = kwargs['rot_factor']
        self.occlusion_aug = kwargs['occlusion_aug'] # whether add occlusion as augmentation
        if self.occlusion_aug:
            print("Add occlusion augmentation to input")
            self.augmentor = Augment(num_occluder=kwargs['num_occluder'])

        if kwargs['percentage'] != 1:  # select more confident pseudo labels and save data
            generate_data(kwargs['percentage'], kwargs['stage'], animal_list=self.animal)

        self.img_list = []
        self.kpts_list = []
        self.load_animal(percentage=kwargs['percentage'], stage=kwargs['stage']) # load selected pseudo labels
        self.mean, self.std = self._compute_mean()

    def load_animal(self, percentage, stage):

        for animal in self.animal:
            n_sample = 0
            img_list = glob.glob(os.path.join(self.img_folder, 'real_animal_crop_v4', 'real_' + animal + '_crop', '*.jpg'))
            img_list = sorted(img_list)
            train_idxs = np.load('./data/real_animal/' + animal + '/train_idxs_by_video.npy')

            if isfile(
                    './animal_data/psudo_labels/stage{}/all/ssl_labels_train_p{}_{}.npy'.format(stage, percentage, animal)):
                print("==> load ssl labels from {} stage{} percentage {}".format(animal, stage, percentage))
                ssl_labels_list = np.load('./animal_data/psudo_labels/stage{}/all/ssl_labels_train_p{}_{}.npy'.format(stage,
                                                         percentage, animal), allow_pickle=True)
                ssl_labels = ssl_labels_list.item()
                for j in range(len(img_list)):
                    if j in train_idxs:
                        idx = int(np.where(train_idxs == j)[0])
                        self.kpts_list.append(ssl_labels.get(idx))
                        self.img_list.append(img_list[j])
                        n_sample += 1
            else:
                print("==> no ssl_labels")
                for img_path in img_list:
                    self.kpts_list.append(None)
                    self.img_list.append(None)
            print('Animal: {}, training sample: {}'.format(animal, n_sample))

    def _compute_mean(self):
        animal = self.animal[0] if len(self.animal) == 1 else 'all'
        # use the same data statistics as the synthetic data
        meanstd_file = './data/synthetic_animal/' + animal + '_combineds5r5_texture' + '/mean.pth.tar'
        if isfile(meanstd_file):
            print('load from mean file:', meanstd_file)
            meanstd = torch.load(meanstd_file)
        else:
            # mean = torch.zeros(3)
            # std = torch.zeros(3)
            # for index in self.img_list:
            #     a = self.img_list[index][0]
            #     img_path = os.path.join(self.img_folder, 'behaviorDiscovery2.0', a)
            #     img = load_image_ori(img_path)  # CxHxW
            #     mean += img.view(img.size(0), -1).mean(1)
            #     std += img.view(img.size(0), -1).std(1)
            # mean /= len(self.img_list)
            # std /= len(self.img_list)
            # meanstd = {
            #     'mean': mean,
            #     'std': std,
            # }
            # torch.save(meanstd, meanstd_file)
            raise Exception('Mean file not found')
        print('  Real animal  mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
        print('  Real animal  std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
        return meanstd['mean'], meanstd['std']

    def _augment(self, img, pts, c):

        flip = False
        if random.random() <= 0.5:
            img = torch.from_numpy(fliplr(img.numpy())).float()
            pts = shufflelr_ori(pts, width=img.size(2), dataset='real_animal')
            c[0] = img.size(2) - c[0]
            flip = True
        if random.random() <= 0.5:
            mu = img.mean()
            img = random.uniform(0.8, 1.2) * (img - mu) + mu
            img.add_(random.uniform(-0.2, 0.2)).clamp_(0, 1)

        return img, pts, c, flip

    # generate pair data for the student-teacher network, strong augmentation are added to the student branch,
    # such as random transformation, occlusion, and noise.
    # The same transformation is added to the output of the teacher network to fulfill the consistency loss
    def __getitem__(self, index):
        sf = self.scale_factor
        rf = self.rot_factor
        if self.is_train:
            pts = self.kpts_list[index]
            img_path = self.img_list[index]
        else:
            raise Exception('Only provide training data')

        if pts is not None:
            pts = pts.astype(np.float32)
            c = torch.Tensor((128, 128))
            s = 256.0 / 200.0
            r = 0.

            nparts = pts.shape[0]
            img = np.array(imageio.imread(img_path))[:, :, :3]
            assert img.shape[0] == 256
            img = im_to_torch(img)
            img_ema = img.clone()
            pts = torch.Tensor(pts)
            pts_ema = pts.clone()
            c_ema = c.clone()
            if self.is_aug:
                s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
                r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0
                img, pts, c, flip = self._augment(img, pts, c)
                if self.occlusion_aug:
                    img, _ = self.augmentor.augment_occlu(im_to_numpy(img))
                    img = im_to_torch(img)
                img_ema, pts_ema, c_ema, flip_ema = self._augment(img_ema, pts_ema, c_ema)
            else:
                flip = False
                flip_ema = False

            # no transformation for teacher network
            inp_ema = crop_ori(img_ema, c_ema, 256.0 / 200.0, [256, 256], rot=0.)
            inp = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r)
            # transformation for student network
            inp_stu = crop_ori(img, c, 256.0 / 200.0, [self.inp_res, self.inp_res], rot=r)
            # generate trainformation matrix for the output of teacher network,
            # the rotation angle is clockwise or anticlockwise depends whether the input are flipped in the same way
            warpmat = get_warpmat(r, 1) if flip == flip_ema else get_warpmat(-r, 1)
            inp = color_normalize(inp, self.mean, self.std)
            inp_stu = color_normalize(inp_stu, self.mean, self.std)
            inp_ema = color_normalize(inp_ema, self.mean, self.std)
            tpts = pts.clone()
            tpts_256 = pts.clone()  # just for visualizing and debugging
            target = torch.zeros(nparts, self.out_res, self.out_res)
            target_weight = tpts[:, 2].clone().view(nparts, 1)
            target_ema = torch.zeros(nparts, self.out_res, self.out_res)
            target_stu = torch.zeros(nparts, self.out_res, self.out_res)
            for i in range(nparts):
                if tpts[i, 1] > 0:
                    tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2], c, s, [self.out_res, self.out_res], rot=r))
                    tpts_256[i, 0:2] = to_torch(transform(tpts_256[i, 0:2], c, s, [self.inp_res, self.inp_res], rot=r))
                    target[i], vis = draw_labelmap_ori(target[i], tpts[i] - 1, self.sigma)
                    target_weight[i, 0] *= vis
            tpts[:, 2] = target_weight.view(-1)

        else:
            img = np.array(imageio.imread(img_path))[:, :, :3]
            assert img.shape[0] == 256
            img = im_to_torch(img)

            c = torch.Tensor((128, 128))
            s = 256.0 / 200.0
            r = 0
            inp = crop_ori(img, c, s, [self.inp_res, self.inp_res], rot=r)
            inp = color_normalize(inp, self.mean, self.std)
            inp_ema = inp.clone()
            inp_stu = inp.clone()
            tpts = torch.Tensor(0)
            target_weight = torch.Tensor(0)
            target = torch.Tensor(0)
            target_ema = torch.Tensor(0)
            target_stu = torch.Tensor(0)
            warpmat = torch.ones((2, 3))

        meta = {'index': index, 'center': c, 'scale': s,
                'tpts': tpts, 'target_weight': target_weight,
                'warpmat': warpmat, 'flip': flip, 'flip_ema': flip_ema, 'tpts_256': tpts_256}
        return inp, inp_ema, inp_stu, target, target_ema, target_stu, meta

    def __len__(self):
        return len(self.img_list)


def real_animal_crop_all(**kwargs):
    return Real_Animal_Crop_All(**kwargs)


real_animal_crop_all.njoints = 18



