# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import torch
from torch.utils.data.sampler import Sampler
import random
# import torch
# import numpy as np
from abc import ABC, abstractmethod

class TensorBuffer:
    """
    A buffer to store tensors. Used to enlarge the number of negative samples when calculating contrastive loss.
    """
    def __init__(self, buffer_size: int, concat_dim: int, retain_gradient: bool = True):
        """
        Args:
            buffer_size: int, the number of stored tensors
            concat_dim: specify a dimension to concatenate the stored tensors, usually the batch dim
            retain_gradient: whether to detach the tensor from the computational graph, must set `retain_graph=True`
                            during backward
        """
        self.buffer_size = buffer_size
        self.concat_dim = concat_dim
        self.retain_gradient = retain_gradient
        self.tensor_list = []

    def update(self, tensor):
        if len(self.tensor_list) >= self.buffer_size:
            self.tensor_list.pop(0)
        if self.retain_gradient:
            self.tensor_list.append(tensor)
        else:
            self.tensor_list.append(tensor.detach())

    @property
    def values(self):
        return torch.cat(self.tensor_list, dim=self.concat_dim)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * 0.01
        # self.wd = 0.02 * args.base_lr

        # 先将两个模型的参数初始化为一样的值
        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        # 之后每次对ema模型进行平滑更新
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                # 0.99 * 上一次的模型参数 + 0.01 * 更新后的模型参数
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

class RandTransform(ABC):
    @abstractmethod
    def randomize(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def apply_transform(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def inverse_transform(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_image(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def invert_label(self, *args, **kwargs):
        raise NotImplementedError

class RandIntensityDisturbance(RandTransform):
    def __init__(self, p: float = 0.1, brightness_limit: float = 0.5, contrast_limit: float = 0.5, clip: bool = False,
                 beta_by_max: bool = True):
        self.beta = (-brightness_limit, brightness_limit)
        self.alpha = (1 - contrast_limit, 1 + contrast_limit)
        self.clip = clip
        self.beta_by_max = beta_by_max
        self.p = p

        self.alpha_value = None
        self.beta_value = None

        self._do_transform = False

    def randomize(self):
        if random.uniform(0, 1) < self.p:
            self._do_transform = True
            self.alpha_value = random.uniform(self.alpha[0], self.alpha[1])
            self.beta_value = random.uniform(self.beta[0], self.beta[1])

    def apply_transform(self, inputs):
        """
        Apply brightness and contrast transform on image
            Args: inputs, torch.tensor, shape (B, C, H, W)
        """
        if self._do_transform:
            img_t = self.alpha_value * inputs
            if self.beta_by_max:
                img_t = img_t + self.beta_value
            else:
                img_t = img_t + self.beta_value * torch.mean(img_t)
            return torch.clamp(img_t, 0, 1) if self.clip else img_t
        else:
            return inputs

    def inverse_transform(self, img_t):
        raise NotImplementedError

    def forward_image(self, image, randomize=True):
        if randomize:
            self.randomize()
        return self.apply_transform(image)

    def invert_label(self, label_t):
        return label_t


class RandGaussianNoise(RandTransform):
    def __init__(self, p: float = 0.2, mean: float = 0.0, std: float = 0.1, clip: bool = False):
        self.p = p
        self.mean = mean
        self.std = std
        self.clip = clip

        self.std_value = None
        self._do_transform = False

    def randomize(self, inputs):
        if random.uniform(0, 1) < self.p:
            self._do_transform = True
            self.std_value = random.uniform(0, self.std)
            self.noise = torch.normal(self.mean, self.std_value, size=inputs.shape)

    def apply_transform(self, inputs):
        if self._do_transform:
            added = inputs + self.noise.to(inputs.device)
            return torch.clamp(added, 0, 1) if self.clip else added
        else:
            return inputs

    def inverse_transform(self, img_t):
        raise NotImplementedError

    def forward_image(self, image, randomize=True):
        if randomize:
            self.randomize(image)
        return self.apply_transform(image)

    def invert_label(self, label_t):
        return label_t

class CutOut(RandTransform):
    def __init__(self, p: float = 0.1, num_holes: int = 5, hole_ratio: float = 0.1, value: float = 0.0):
        # hole 5 and ratio 0.05 test fine
        self.p = p
        self.num_holes = num_holes
        self.hole_ratio = hole_ratio
        self.value = value

        self.hole_list = None

    def rand_bbox(self, input_shape):
        W = input_shape[2]
        H = input_shape[3]
        D = input_shape[4]
        cut_w = np.int(W * self.hole_ratio)
        cut_h = np.int(H * self.hole_ratio)
        cut_d = np.int(D * self.hole_ratio)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        cz = np.random.randint(D)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbz1 = np.clip(cz - cut_d // 2, 0, D)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        bbz2 = np.clip(cz + cut_d // 2, 0, D)

        return bbx1, bby1, bbz1, bbx2, bby2, bbz2

    def fill_hole(self, inputs, location, value):
        processed = inputs.clone()
        processed[:, :, location[0]:location[3], location[1]:location[4], location[2]:location[5]] = value
        return processed

    def randomize(self, inputs):
        inputs_shape = inputs.shape
        hole_list = []
        for i in range(self.num_holes):
            loc = self.rand_bbox(inputs_shape)
            hole_list.append(loc)
        return hole_list

    def apply_transform(self, inputs, hole_list):
        for hole in hole_list:
            inputs = self.fill_hole(inputs, hole, self.value)
        return inputs

    def inverse_transform(self):
        raise NotImplementedError

    def forward_image(self, inputs, randomize=True):
        if randomize:
            self.hole_list = self.randomize(inputs)
        outputs = self.apply_transform(inputs, self.hole_list)
        return outputs

    def invert_label(self):
        raise NotImplementedError

class FullAugmentor:
    """
    Augmentor to generate augmented views of the input, support intensity shift or scaling, Gaussian noise and
    CutOut (optional)
    """
    def __init__(self):
        self.intensity = RandIntensityDisturbance(p=1, clip=False, beta_by_max=False)
        self.gaussian = RandGaussianNoise(p=0.5, clip=False)
        self.cutout = CutOut(p=0.1)

    def forward_image(self, image):
        image = self.intensity.forward_image(image)
        image = self.gaussian.forward_image(image)
        # image = self.cutout.forward_image(image)
        return image
def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        posmask = img_gt[b].astype(bool)
        if posmask.any():
            negmask = ~posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
            sdf[boundary==1] = 0
            normalized_sdf[b] = sdf
            # assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
            # assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf