import cv2
import numpy as np
import torch

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
from pose.utils.evaluation import final_preds, get_preds


def prediction_check(previous_inp, previous_kpts, inp, model, dataset, device, num_transform=5, num_kpts=18, lambda_decay=0.9):
    """
    Input:
        Image: 3x256x256
    Output:
        generated_kpts: 18x3
        target: 3x256x256
        target_weight: 
    """

    # equivariant consistency
    animal_mean = dataset.mean
    animal_std = dataset.std
    s0 = 256/200.0
    sf = 0.25
    rf = 30
    c = torch.Tensor((128, 128))
    preds_all = np.zeros((num_transform, num_kpts, 2)) # (x, y)

    confidence = np.ones(18)
    score_map_avg = np.zeros((1,18,64,64))
    for i in range(num_transform):

        img = inp.clone()
        if i==0:
            s = s0
            rot = 0
        else:
            s = s0*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
            rot = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0]

        img = crop_ori(img, c, s, [256, 256], rot)
        img = color_normalize(img, animal_mean, animal_std)

        model_out, model_out_refine = model(img.unsqueeze(0), 1, return_domain=False)
        score_map = model_out_refine[0].cpu()
        feat_map = score_map.squeeze(0).detach().cpu().numpy()

        # with flip
        flip_input = torch.from_numpy(fliplr(img.clone().cpu().numpy())).float().to(device)
        flip_output, flip_output_refine = model(flip_input.unsqueeze(0), 1, return_domain=False)
        flip_output_re = flip_back(flip_output_refine[0].detach().cpu(), 'real_animal')
        feat_map += flip_output_re.squeeze(0).numpy()
        feat_map /= 2

        # rotate and scale score_map back
        for j in range(feat_map.shape[0]):
            feat_map_j = feat_map[j]
            M = cv2.getRotationMatrix2D((32,32),-rot,1)
            feat_map_j = cv2.warpAffine(feat_map_j,M,(64,64))
            feat_map_j = cv2.resize(feat_map_j,None,fx=s*200.0/256.0,fy=s*200.0/256.0, interpolation=cv2.INTER_LINEAR)

            if feat_map_j.shape[0]<64:
                start = 32-feat_map_j.shape[0]//2
                end = start+feat_map_j.shape[0]
                score_map_avg[0][j][start:end, start:end] += feat_map_j
            else:
                start = feat_map_j.shape[0]//2-32
                end = feat_map_j.shape[0]//2+32
                score_map_avg[0][j] += feat_map_j[start:end, start:end]
    
    score_map_avg = score_map_avg/num_transform
    confidence_score = np.max(score_map_avg, axis=(0,2,3))

    confidence = confidence_score.astype(np.float32)
    score_map_avg = torch.Tensor(score_map_avg)

    preds = final_preds(score_map_avg, [c], [s0], [64, 64])
    preds = preds.squeeze(0)
    pts = preds.clone().cpu().numpy()

    generated_kpts = np.zeros((num_kpts, 3)).astype(np.float32)
    generated_kpts[:,:2] = pts
    generated_kpts[:,2] = confidence

    # temporal consistency
    if previous_inp is not None:
        lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        current_img = (inp.clone().cpu() + animal_mean.view(3,1,1)).numpy().transpose(1,2,0)
        previous_img = (previous_inp.clone().cpu() + animal_mean.view(3,1,1)).numpy().transpose(1,2,0)
        current_frame = (current_img*255).astype(np.uint8)
        previous_frame = (previous_img*255).astype(np.uint8)
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    
        previous_preds = previous_kpts[:,:2].reshape(18,1,2).astype(np.float32)
        # flow preds (18,1,2)
        flow_preds, st, err = cv2.calcOpticalFlowPyrLK(previous_frame_gray, current_frame_gray, previous_preds, None, **lk_params)
        flow_preds = flow_preds.reshape(18,2)
        previous_preds = previous_preds.reshape(18,2)
        flow_confidence = previous_kpts[:,2].reshape(18,1)

        # caculate kpts dist to check flow confidence
        for j in range(18):
            if np.linalg.norm(flow_preds[j] - previous_preds[j]) > 15:
                flow_confidence[j] = 0
        # combine flow_preds (flow_preds, confidence) with generated preds (generated_kpts, confidence)
        for j in range(18):
            if flow_confidence[j]>0:
                if (confidence[j]/flow_confidence[j])<lambda_decay:
                    generated_kpts[j,:2] = flow_preds[j,:2]
                    generated_kpts[j,2] = flow_confidence[j]*lambda_decay
    target = kpts_to_heatmap(generated_kpts)
    return target, generated_kpts


def kpts_to_heatmap(generated_kpts):
    ''' draw heatmaps by 64x64 '''
    num_kpts = generated_kpts.shape[0]
    generated_kpts = torch.Tensor(generated_kpts.astype(np.int32))
    #target_weight = generated_kpts[:, 2].clone().view(num_kpts, 1)
    #target_weight = None
    pts = generated_kpts[:,:2]
    tpts = pts.clone()
    s = 256/200.0
    rot = 0.0
    c = torch.Tensor((128, 128))
    target = torch.zeros(num_kpts, 64, 64)

    for j in range(num_kpts):
        tpts[j,0:2] = to_torch(transform(pts[j,0:2], c, s, [64, 64], invert=0, rot=rot))
        target[j], vis = draw_labelmap_ori(target[j], tpts[j]-1, 1, type='Gaussian')

    return target.unsqueeze(0)

