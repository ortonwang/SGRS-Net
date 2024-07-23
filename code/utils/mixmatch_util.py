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
import torch.nn.functional as F
import random
from abc import ABC, abstractmethod



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



def augmentation(volume, aug_factor):
    # volume is numpy array of shape (C, D, H, W)
    return volume + aug_factor * np.clip(np.random.randn(*volume.shape) * 0.1, -0.2, 0.2).astype(np.float32)

def augmentation_torch(volume, aug_factor):
    # volume is numpy array of shape (C, D, H, W)
    noise = torch.clip(torch.randn(*volume.shape) * 0.1, -0.2, 0.2).cuda()
    return volume + aug_factor * noise#.astype(np.float32)



def mix_match_just_k1(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor):
    X_b = len(X)
    U_b = len(U)

    X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.

    # step 2: label guessing
    with torch.no_grad():
        # Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80)
        Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80)
        Y_u = F.softmax(Y_u, dim=1)

        # print(Y_u.shape)
    guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)  #
    # if GPU:
    guessed = guessed.cuda()
    if K !=1:
        for i in range(K):
            guessed += Y_u[i * U_b:(i + 1) * U_b]
        guessed /= K  #
        guessed = guessed.repeat(K, 1, 1, 1, 1)

        # guessed = guessed.detach().cpu().numpy()
        guessed = torch.argmax(guessed, dim=1)
    else:

        guessed= torch.argmax(Y_u, dim=1)

    # shape [U_b * K,2,D,H,W]
    pseudo_label = guessed
    # U_cap = U_cap.detach().cpu().numpy()

    # all_input = np.concatenate([x_in, U_cap], axis=0)
    # all_label = np.concatenate([x_lab, guessed], axis=0)

    U_cap = list(zip(U_cap, guessed))  #

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    if x_mixup_mode == 'x':
        idxs = np.random.permutation(range(X_b))  #
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(U_b * K))[:X_b]  #
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == '_':
        X_prime = X_cap
    elif x_mixup_mode == 's':
        X_prime = [mix_up(X_cap[i], X_cap[i], alpha) for i in range(X_b)]
    else:
        raise ValueError('wrong mixup_mode')


    if u_mixup_mode == 'x':  #
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'u':  #
        idxs = np.random.permutation(range(U_b * K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(U_b * K)]  # 有問題???
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')

    return X_prime, U_prime, pseudo_label,X_cap




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


def mix_match_pth_noise_intensity(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor,augu_noise=None,augu_intensity=None):
    X_b = len(X)
    U_b = len(U)  # 应该是得到未标签的batch_size数才对

    # step 1: Augmentation
    # 随机在输入上增加扰动
    # aug_factor = torch.tensor(aug_factor)
    X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    # batchsize翻K倍
    # U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]
    U_cap1 = U + torch.clamp(torch.randn_like(U) * 0.1, -0.2, 0.2)
    U_cap2 = augu_intensity.forward_image(U)
    # 一样加上一些随机扰动
    # U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.

    # step 2: label guessing
    with torch.no_grad():
        # Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = eval_net(U_cap1)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = F.softmax(Y_u, dim=1)
        Y_u2 = eval_net(U_cap2)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u2 = F.softmax(Y_u2, dim=1)
    guessed = Y_u + Y_u2
        # print(Y_u.shape)
    # guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)  # empty label [b, 2, D, H, W]  这边是做成one-hot形式了
    # # if GPU:
    # guessed = guessed.cuda()
    # for i in range(K):
    #     guessed += Y_u[i * U_b:(i + 1) * U_b]
    # guessed /= K  # 将多次重复算个平均结果  原文的公式(6)

    # 此外在做一些处理, 得到伪标签
    # guessed_t = guessed ** (1 / T)  # 原文的公式(7)
    # guessed = guessed_t / ()
    # guessed = guessed / guessed.sum(dim=1, keepdim=True)  # 原文的公式(7)
    # pse_dis1 = guessed ** (1 / 0.2)
    # pse_dis2 = (1 - guessed) ** (1 / 0.2)
    # guessed = pse_dis1 / (pse_dis1 + pse_dis2) #这里其实就是 sharpen 的操作，系数=0.2
    # guessed = guessed.repeat(K, 1, 1, 1, 1)

    # guessed = guessed.detach().cpu().numpy()
    guessed = torch.argmax(guessed, dim=1)
    # shape [U_b * K,2,D,H,W]  得到伪标签
    pseudo_label = guessed

    # U_cap = U_cap.detach().cpu().numpy()

    # all_input = np.concatenate([x_in, U_cap], axis=0)
    # all_label = np.concatenate([x_lab, guessed], axis=0)
    U_cap = torch.cat([U_cap1,U_cap2],0)
    guessed = torch.cat([guessed,guessed],0)
    U_cap = list(zip(U_cap, guessed))  # 将伪标签和绕动数据合并



    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    if x_mixup_mode == 'x':
        idxs = np.random.permutation(range(X_b))  # 保证取的范围在带标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(U_b * K))[:X_b]  # 保证取的范围在不标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == '_':
        X_prime = X_cap
    elif x_mixup_mode == 's':
        X_prime = [mix_up(X_cap[i], X_cap[i], alpha) for i in range(X_b)]
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'x':  # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'u':  # 保证取的范围在不带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(U_b * K)]  # 有問題???
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')
    # print(U_cap[0][0].shape, U_cap[0][1].shape, X_cap[0][0].shape, X_cap[0][1].shape)
    # if DEBUG:
    # save_as_image(np.array([x[0] for x in U_prime]), f"../debug_output/u_prime_data")
    # save_as_image(np.array([x[1][[1], :, :, :] for x in U_prime]), f"../debug_output/u_prime_label")
    return X_prime, U_prime, guessed


def get_pseudo_label_noise_and_intensty_pth(l_image, U, eval_net, augu_noise=None,augu_intensity=None):
    # step 1: Augmentation
    # 随机在输入上增加扰动
    # aug_factor = torch.tensor(aug_factor)
    # X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    l_image_noise = augu_noise.forward_image(l_image)
    l_image_intensity = augu_intensity.forward_image(l_image)

    ul_image_noise = augu_noise.forward_image(U)
    ul_image_intensity = augu_intensity.forward_image(U)
    # batchsize翻K倍
    U_cap = U#.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]
    # step 2: label guessing
    with torch.no_grad():
        Y_u1,Y_u_noise,Y_u_intensity = eval_net(U_cap),eval_net(ul_image_noise),eval_net(ul_image_intensity)
        # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = Y_u1+Y_u_noise+Y_u_intensity
        pseudo_label = torch.argmax(Y_u, dim=1)
        # print(Y_u.shape)
    return l_image_noise,l_image_intensity,ul_image_noise,ul_image_intensity,pseudo_label





def mix_match_noise_and_intensty_pth(l_image, U, eval_net, augu_noise=None,augu_intensity=None):
    # step 1: Augmentation
    # 随机在输入上增加扰动
    # aug_factor = torch.tensor(aug_factor)
    # X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    l_image_noise = augu_noise.forward_image(l_image)
    l_image_intensity = augu_intensity.forward_image(l_image)

    ul_image_noise = augu_noise.forward_image(U)
    ul_image_intensity = augu_intensity.forward_image(U)
    # batchsize翻K倍
    U_cap = U#.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]
    # step 2: label guessing
    with torch.no_grad():
        Y_u1,Y_u_noise,Y_u_intensity = eval_net(U_cap),eval_net(ul_image_noise),eval_net(ul_image_intensity)
        # if len(Y_u1) == 1:
        # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = Y_u1+Y_u_noise+Y_u_intensity
        # if len(Y_u1) > 1:
        #     Y_u = Y_u1[0] + Y_u_noise[0] + Y_u_intensity[0]
        pseudo_label = torch.argmax(Y_u, dim=1)
        # print(Y_u.shape)

    # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合

    U_b,X_b =  len(U),len(l_image)
    K,alpha = 3,0.75

    idxs = np.random.permutation(range(U_b * K)) % X_b
    X_cap = torch.cat([l_image,l_image_noise,l_image_intensity],0)
    U_cap = torch.cat([U, ul_image_noise, ul_image_intensity], 0)

    U_mixed = [torch.unsqueeze(mix_up_only_img(U_cap[i], X_cap[idxs[i]], alpha),0) for i in range(U_b * K)]
    concatenated_tensor = torch.cat(U_mixed, dim=0)

    return l_image_noise,l_image_intensity,ul_image_noise,ul_image_intensity,pseudo_label,concatenated_tensor,X_cap,U_cap

def mix_match_noise_and_intensty_pth_multi_out(l_image, U, eval_net, augu_noise=None,augu_intensity=None):
    # step 1: Augmentation
    # 随机在输入上增加扰动
    # aug_factor = torch.tensor(aug_factor)
    # X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    l_image_noise = augu_noise.forward_image(l_image)
    l_image_intensity = augu_intensity.forward_image(l_image)

    ul_image_noise = augu_noise.forward_image(U)
    ul_image_intensity = augu_intensity.forward_image(U)
    # batchsize翻K倍
    U_cap = U#.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]
    # step 2: label guessing
    with torch.no_grad():
        Y_u1,Y_u_noise,Y_u_intensity = eval_net(U_cap),eval_net(ul_image_noise),eval_net(ul_image_intensity)
        # if len(Y_u1) == 1:
        # # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        #     Y_u = Y_u1+Y_u_noise+Y_u_intensity
        # if len(Y_u1) > 1:
        Y_u = Y_u1[0] + Y_u_noise[0] + Y_u_intensity[0]
        pseudo_label = torch.argmax(Y_u, dim=1)
        # print(Y_u.shape)

    # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合

    U_b,X_b =  len(U),len(l_image)
    K,alpha = 3,0.75

    idxs = np.random.permutation(range(U_b * K)) % X_b
    X_cap = torch.cat([l_image,l_image_noise,l_image_intensity],0)
    U_cap = torch.cat([U, ul_image_noise, ul_image_intensity], 0)

    U_mixed = [torch.unsqueeze(mix_up_only_img(U_cap[i], X_cap[idxs[i]], alpha),0) for i in range(U_b * K)]
    concatenated_tensor = torch.cat(U_mixed, dim=0)

    return l_image_noise,l_image_intensity,ul_image_noise,ul_image_intensity,pseudo_label,concatenated_tensor,X_cap,U_cap

def mix_match_noise_and_intensty_pth_no_mix(l_image, U, eval_net, augu_noise=None,augu_intensity=None):
    # step 1: Augmentation
    # 随机在输入上增加扰动
    # aug_factor = torch.tensor(aug_factor)
    # X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    l_image_noise = augu_noise.forward_image(l_image)
    l_image_intensity = augu_intensity.forward_image(l_image)

    ul_image_noise = augu_noise.forward_image(U)
    ul_image_intensity = augu_intensity.forward_image(U)
    # batchsize翻K倍
    U_cap = U#.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]
    # step 2: label guessing
    with torch.no_grad():
        Y_u1,Y_u_noise,Y_u_intensity = eval_net(U_cap),eval_net(ul_image_noise),eval_net(ul_image_intensity)
        if len(Y_u1) == 1:
        # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
            Y_u = Y_u1+Y_u_noise+Y_u_intensity
        if len(Y_u1) > 1:
            Y_u = Y_u1[0] + Y_u_noise[0] + Y_u_intensity[0]
        pseudo_label = torch.argmax(Y_u, dim=1)
        # print(Y_u.shape)

    # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合

    U_b,X_b =  len(U),len(l_image)
    K,alpha = 3,0.75

    idxs = np.random.permutation(range(U_b * K)) % X_b
    X_cap = torch.cat([l_image,l_image_noise,l_image_intensity],0)
    U_cap = torch.cat([U, ul_image_noise, ul_image_intensity], 0)

    # U_mixed = [torch.unsqueeze(mix_up_only_img(U_cap[i], X_cap[idxs[i]], alpha),0) for i in range(U_b * K)]
    # concatenated_tensor = torch.cat(U_mixed, dim=0)

    return l_image_noise,l_image_intensity,ul_image_noise,ul_image_intensity,pseudo_label,X_cap,U_cap



def sharpening2(x,sharpen_d):
    # T = 1 / args.temperature
    T = 5
    y = torch.zeros_like(x)
    d1 = sharpen_d
    d = 0.8 * d1
    l, r = 0.5 - d1, 0.5 + d1
    y[(x < l)] = (x[x < l] + d) ** T / ((x[x < l] + d) ** T + (1 - (x[x < l] + d)) ** T)
    y[(x > r)] = (x[x > r] - d) ** T / ((x[x > r] - d) ** T + (1 - (x[x > r] - d)) ** T)
    y[(x >= l) & (x <= r)] = x[(x >= l) & (x <= r)]
    # P_sharpen = P ** T / (P ** T + (1 - P) ** T)
    return y #P_sharpen
def sharpening(x):
    # T = 1 / args.temperature
    T = 5
    # y = torch.zeros_like(x)
    # d1 = sharpen_d
    # d = 0.8 * d1
    # l, r = 0.5 - d1, 0.5 + d1
    # y[(x < l)] = (x[x < l] + d) ** T / ((x[x < l] + d) ** T + (1 - (x[x < l] + d)) ** T)
    # y[(x > r)] = (x[x > r] - d) ** T / ((x[x > r] - d) ** T + (1 - (x[x > r] - d)) ** T)
    # y[(x >= l) & (x <= r)] = x[(x >= l) & (x <= r)]
    y = x ** T / (x ** T + (1 - x) ** T)
    return y #P_sharpen

def mix_match_3_pth_soft_sharp(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor,sharp_d):
    X_b = len(X)
    U_b = len(U)  # 应该是得到未标签的batch_size数才对

    # step 1: Augmentation
    # 随机在输入上增加扰动
    # aug_factor = torch.tensor(aug_factor)
    X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    # batchsize翻K倍
    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]

    # 一样加上一些随机扰动
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.

    # step 2: label guessing
    with torch.no_grad():
        # Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u,_,_ = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = F.softmax(Y_u, dim=1)

        # print(Y_u.shape)
    guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)  # empty label [b, 2, D, H, W]  这边是做成one-hot形式了
    # if GPU:
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K  # 将多次重复算个平均结果  原文的公式(6)

    # 此外在做一些处理, 得到伪标签
    # guessed_t = guessed ** (1 / T)  # 原文的公式(7)
    # guessed = guessed_t / ()
    # guessed = guessed / guessed.sum(dim=1, keepdim=True)  # 原文的公式(7)
    # pse_dis1 = guessed ** (1 / 0.2)
    # pse_dis2 = (1 - guessed) ** (1 / 0.2)
    # guessed = pse_dis1 / (pse_dis1 + pse_dis2) #这里其实就是 sharpen 的操作，系数=0.2.
    guessed = sharpening2(guessed,sharpen_d=sharp_d)
    guessed = guessed.repeat(K, 1, 1, 1, 1)

    # guessed = guessed.detach().cpu().numpy()
    guessed = torch.argmax(guessed, dim=1)
    # shape [U_b * K,2,D,H,W]  得到伪标签
    pseudo_label = guessed

    # U_cap = U_cap.detach().cpu().numpy()

    # all_input = np.concatenate([x_in, U_cap], axis=0)
    # all_label = np.concatenate([x_lab, guessed], axis=0)

    U_cap = list(zip(U_cap, guessed))  # 将伪标签和绕动数据合并

    ## Now we have X_cap ,list of (data, label) of length b, U_cap, list of (data, guessed_label) of length k*b

    # step 3: MixUp
    # original paper mathod

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    if x_mixup_mode == 'x':
        idxs = np.random.permutation(range(X_b))  # 保证取的范围在带标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(U_b * K))[:X_b]  # 保证取的范围在不标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == '_':
        X_prime = X_cap
    elif x_mixup_mode == 's':
        X_prime = [mix_up(X_cap[i], X_cap[i], alpha) for i in range(X_b)]
    else:
        raise ValueError('wrong mixup_mode')


    if u_mixup_mode == 'x':  # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'u':  # 保证取的范围在不带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(U_b * K)]  # 有問題???
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')
    # print(U_cap[0][0].shape, U_cap[0][1].shape, X_cap[0][0].shape, X_cap[0][1].shape)
    # if DEBUG:
    # save_as_image(np.array([x[0] for x in U_prime]), f"../debug_output/u_prime_data")
    # save_as_image(np.array([x[1][[1], :, :, :] for x in U_prime]), f"../debug_output/u_prime_label")
    return X_prime, U_prime, pseudo_label

def mix_match_3_pth_soft_sharp_iter(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor,sharp_d,iter_num = 0):
    X_b = len(X)
    U_b = len(U)  # 应该是得到未标签的batch_size数才对

    # step 1: Augmentation
    # 随机在输入上增加扰动
    # aug_factor = torch.tensor(aug_factor)
    X_cap = [(augmentation_torch(x[0], aug_factor), x[1]) for x in X]

    # batchsize翻K倍
    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]

    # 一样加上一些随机扰动
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.

    # step 2: label guessing
    with torch.no_grad():
        # Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u,_,_ = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = F.softmax(Y_u, dim=1)

        # print(Y_u.shape)
    guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)  # empty label [b, 2, D, H, W]  这边是做成one-hot形式了
    # if GPU:
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K  # 将多次重复算个平均结果  原文的公式(6)

    # 此外在做一些处理, 得到伪标签
    # guessed_t = guessed ** (1 / T)  # 原文的公式(7)
    # guessed = guessed_t / ()
    # guessed = guessed / guessed.sum(dim=1, keepdim=True)  # 原文的公式(7)
    # pse_dis1 = guessed ** (1 / 0.2)
    # pse_dis2 = (1 - guessed) ** (1 / 0.2)
    # guessed = pse_dis1 / (pse_dis1 + pse_dis2) #这里其实就是 sharpen 的操作，系数=0.2.
    if iter_num < 5000:
        guessed = sharpening2(guessed,sharpen_d=sharp_d)
    else:guessed = sharpening(guessed)
    guessed = guessed.repeat(K, 1, 1, 1, 1)

    # guessed = guessed.detach().cpu().numpy()
    guessed = torch.argmax(guessed, dim=1)
    # shape [U_b * K,2,D,H,W]  得到伪标签
    pseudo_label = guessed

    # U_cap = U_cap.detach().cpu().numpy()

    # all_input = np.concatenate([x_in, U_cap], axis=0)
    # all_label = np.concatenate([x_lab, guessed], axis=0)

    U_cap = list(zip(U_cap, guessed))  # 将伪标签和绕动数据合并

    ## Now we have X_cap ,list of (data, label) of length b, U_cap, list of (data, guessed_label) of length k*b

    # step 3: MixUp
    # original paper mathod

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    if x_mixup_mode == 'x':
        idxs = np.random.permutation(range(X_b))  # 保证取的范围在带标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(U_b * K))[:X_b]  # 保证取的范围在不标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == '_':
        X_prime = X_cap
    elif x_mixup_mode == 's':
        X_prime = [mix_up(X_cap[i], X_cap[i], alpha) for i in range(X_b)]
    else:
        raise ValueError('wrong mixup_mode')


    if u_mixup_mode == 'x':  # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'u':  # 保证取的范围在不带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(U_b * K)]  # 有問題???
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')
    # print(U_cap[0][0].shape, U_cap[0][1].shape, X_cap[0][0].shape, X_cap[0][1].shape)
    # if DEBUG:
    # save_as_image(np.array([x[0] for x in U_prime]), f"../debug_output/u_prime_data")
    # save_as_image(np.array([x[1][[1], :, :, :] for x in U_prime]), f"../debug_output/u_prime_label")
    return X_prime, U_prime, pseudo_label

def mix_match_3_k1(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor):
    """
    参考论文 MixMatch: A Holistic Approach to Semi-Supervised Learning
    """
    # X is labeled data of size BATCH_SIZE, and U is unlabeled data
    # X is list of tuples (data, label), and U is list of data
    # where data and label are of shape (C, D, H, W), numpy array. C of data is 1 and C of label is 2 (one hot)
    X_b = len(X)
    U_b = len(U)  # 应该是得到未标签的batch_size数才对

    # step 1: Augmentation
    # 随机在输入上增加扰动

    X_cap = [(augmentation(x[0], aug_factor), x[1]) for x in X]

    # batchsize翻K倍
    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]

    # 一样加上一些随机扰动
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.

    # step 2: label guessing
    with torch.no_grad():
        # Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u,_,_ = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = F.softmax(Y_u, dim=1)

        # print(Y_u.shape)
    guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)  # empty label [b, 2, D, H, W]  这边是做成one-hot形式了
    # if GPU:
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K  # 将多次重复算个平均结果  原文的公式(6)

    # 此外在做一些处理, 得到伪标签
    # guessed_t = guessed ** (1 / T)  # 原文的公式(7)
    # guessed = guessed_t / ()
    # guessed = guessed / guessed.sum(dim=1, keepdim=True)  # 原文的公式(7)
    pse_dis1 = guessed ** (1 / 0.2)
    pse_dis2 = (1 - guessed) ** (1 / 0.2)
    guessed = pse_dis1 / (pse_dis1 + pse_dis2) #这里其实就是 sharpen 的操作，系数=0.2
    guessed = guessed.repeat(K, 1, 1, 1, 1)

    guessed = guessed.detach().cpu().numpy()
    # shape [U_b * K,2,D,H,W]  得到伪标签
    pseudo_label = guessed
    U_cap = U_cap.detach().cpu().numpy()

    # all_input = np.concatenate([x_in, U_cap], axis=0)
    # all_label = np.concatenate([x_lab, guessed], axis=0)

    U_cap = list(zip(U_cap, guessed))  # 将伪标签和绕动数据合并

    ## Now we have X_cap ,list of (data, label) of length b, U_cap, list of (data, guessed_label) of length k*b

    # step 3: MixUp
    # original paper mathod

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    W = X_cap + U_cap  # length = X_b + U_b * k, 将带标签的数据和伪标签数据合并
    random.shuffle(W)  # 随机打乱顺序

    #  W: L 与 L+UL的混合数据   做 mix up
    #  X: L 与 L的混合数据   做 mix up
    if x_mixup_mode == 'w':
        X_prime = [mix_up(X_cap[i], W[i], alpha) for i in range(X_b)]  # 只取带标签的batch_size数进行融合
    elif x_mixup_mode == 'x':
        idxs = np.random.permutation(range(X_b))  # 保证取的范围在带标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(U_b * K))[:X_b]  # 保证取的范围在不标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == '_':
        X_prime = X_cap
    elif x_mixup_mode == 's':
        X_prime = [mix_up(X_cap[i], X_cap[i], alpha) for i in range(X_b)]
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'w':  # 扣除前X_b个, 剩下进行融合
        U_prime = [mix_up(U_cap[i], W[X_b + i], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'x':  # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'u':  # 保证取的范围在不带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(U_b * K)]  # 有問題???
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')
    # print(U_cap[0][0].shape, U_cap[0][1].shape, X_cap[0][0].shape, X_cap[0][1].shape)
    # if DEBUG:
    # save_as_image(np.array([x[0] for x in U_prime]), f"../debug_output/u_prime_data")
    # save_as_image(np.array([x[1][[1], :, :, :] for x in U_prime]), f"../debug_output/u_prime_label")
    return X_prime, U_prime, pseudo_label

def mix_match_3_pseudo(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor):
    """
    参考论文 MixMatch: A Holistic Approach to Semi-Supervised Learning
    """
    # X is labeled data of size BATCH_SIZE, and U is unlabeled data
    # X is list of tuples (data, label), and U is list of data
    # where data and label are of shape (C, D, H, W), numpy array. C of data is 1 and C of label is 2 (one hot)
    X_b = len(X)
    U_b = len(U)  # 应该是得到未标签的batch_size数才对

    # step 1: Augmentation
    # 随机在输入上增加扰动

    X_cap = [(augmentation(x[0], aug_factor), x[1]) for x in X]

    # batchsize翻K倍
    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]

    # 一样加上一些随机扰动
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.

    # step 2: label guessing
    with torch.no_grad():
        # Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u,_,_ = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = F.softmax(Y_u, dim=1)

        # print(Y_u.shape)
    guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)  # empty label [b, 2, D, H, W]  这边是做成one-hot形式了
    # if GPU:
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K  # 将多次重复算个平均结果  原文的公式(6)

    # 此外在做一些处理, 得到伪标签
    # guessed_t = guessed ** (1 / T)  # 原文的公式(7)
    # guessed = guessed_t / ()
    # guessed = guessed / guessed.sum(dim=1, keepdim=True)  # 原文的公式(7)
    pse_dis1 = guessed ** (1 / 0.2)
    pse_dis2 = (1 - guessed) ** (1 / 0.2)
    guessed = pse_dis1 / (pse_dis1 + pse_dis2) #这里其实就是 sharpen 的操作，系数=0.2
    guessed = guessed.repeat(K, 1, 1, 1, 1)

    guessed = guessed.detach().cpu().numpy()
    # shape [U_b * K,2,D,H,W]  得到伪标签
    pseudo_label = guessed
    U_cap = U_cap.detach().cpu().numpy()

    guessed = torch.from_numpy(guessed)
    guessed = guessed.argmax(1).numpy()

    # all_input = np.concatenate([x_in, U_cap], axis=0)
    # all_label = np.concatenate([x_lab, guessed], axis=0)
    # guessed = torch.from_numpy(guessed)
    # guessed = guessed.argmax(1).numpy()
    U_cap = list(zip(U_cap, guessed))  # 将伪标签和绕动数据合并

    ## Now we have X_cap ,list of (data, label) of length b, U_cap, list of (data, guessed_label) of length k*b

    # step 3: MixUp
    # original paper mathod

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    W = X_cap + U_cap  # length = X_b + U_b * k, 将带标签的数据和伪标签数据合并
    random.shuffle(W)  # 随机打乱顺序

    #  W: L 与 L+UL的混合数据   做 mix up
    #  X: L 与 L的混合数据   做 mix up
    if x_mixup_mode == 'w':
        X_prime = [mix_up(X_cap[i], W[i], alpha) for i in range(X_b)]  # 只取带标签的batch_size数进行融合
    elif x_mixup_mode == 'x':
        idxs = np.random.permutation(range(X_b))  # 保证取的范围在带标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(U_b * K))[:X_b]  # 保证取的范围在不标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == '_':
        X_prime = X_cap
    elif x_mixup_mode == 's':
        X_prime = [mix_up(X_cap[i], X_cap[i], alpha) for i in range(X_b)]
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'w':  # 扣除前X_b个, 剩下进行融合
        U_prime = [mix_up(U_cap[i], W[X_b + i], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'x':  # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'u':  # 保证取的范围在不带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(U_b * K)]  # 有問題???
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')
    # print(U_cap[0][0].shape, U_cap[0][1].shape, X_cap[0][0].shape, X_cap[0][1].shape)
    # if DEBUG:
    # save_as_image(np.array([x[0] for x in U_prime]), f"../debug_output/u_prime_data")
    # save_as_image(np.array([x[1][[1], :, :, :] for x in U_prime]), f"../debug_output/u_prime_label")
    return X_prime, U_prime, pseudo_label

def mix_match_3_pseudo2(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor,iter_num=0,sharp_d=0.04):
    """
    对pseudo label 的获得方式进行改进，由原先的sharpen 整理成后面的sharpen
    参考论文 MixMatch: A Holistic Approach to Semi-Supervised Learning
    """
    # X is labeled data of size BATCH_SIZE, and U is unlabeled data
    # X is list of tuples (data, label), and U is list of data
    # where data and label are of shape (C, D, H, W), numpy array. C of data is 1 and C of label is 2 (one hot)
    X_b = len(X)
    U_b = len(U)  # 应该是得到未标签的batch_size数才对

    # step 1: Augmentation
    # 随机在输入上增加扰动

    X_cap = [(augmentation(x[0], aug_factor), x[1]) for x in X]

    # batchsize翻K倍
    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]

    # 一样加上一些随机扰动
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.

    # step 2: label guessing
    with torch.no_grad():
        # Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u,_,_ = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = F.softmax(Y_u, dim=1)

        # print(Y_u.shape)
    guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)  # empty label [b, 2, D, H, W]  这边是做成one-hot形式了
    # if GPU:
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K  # 将多次重复算个平均结果  原文的公式(6)

    # 此外在做一些处理, 得到伪标签
    # guessed_t = guessed ** (1 / T)  # 原文的公式(7)
    # guessed = guessed_t / ()
    # guessed = guessed / guessed.sum(dim=1, keepdim=True)  # 原文的公式(7)
    # pse_dis1 = guessed ** (1 / 0.2)
    # pse_dis2 = (1 - guessed) ** (1 / 0.2)
    # guessed = pse_dis1 / (pse_dis1 + pse_dis2) #这里其实就是 sharpen 的操作，系数=0.2
    if iter_num < 5000:
        x = guessed
        y = torch.zeros_like(x)
        d1 = sharp_d
        d = 0.8 * d1
        l, r = 0.5 - d1, 0.5 + d1
        y[(x < l)] = (x[x < l] + d) ** T / ((x[x < l] + d) ** T + (1 - (x[x < l] + d)) ** T)
        y[(x > r)] = (x[x > r] - d) ** T / ((x[x > r] - d) ** T + (1 - (x[x > r] - d)) ** T)
        y[(x >= l) & (x <= r)] = x[(x >= l) & (x <= r)]
        guessed = y.repeat(K, 1, 1, 1, 1)
    else:
        pse_dis1 = guessed ** (1 / 0.2)
        pse_dis2 = (1 - guessed) ** (1 / 0.2)
        guessed = pse_dis1 / (pse_dis1 + pse_dis2) #这里其实就是 sharpen 的操作，系数=0.2
        guessed = guessed.repeat(K, 1, 1, 1, 1)

    guessed = guessed.detach().cpu().numpy()
    # shape [U_b * K,2,D,H,W]  得到伪标签
    pseudo_label = guessed
    U_cap = U_cap.detach().cpu().numpy()

    # all_input = np.concatenate([x_in, U_cap], axis=0)
    # all_label = np.concatenate([x_lab, guessed], axis=0)

    U_cap = list(zip(U_cap, guessed))  # 将伪标签和绕动数据合并

    ## Now we have X_cap ,list of (data, label) of length b, U_cap, list of (data, guessed_label) of length k*b

    # step 3: MixUp
    # original paper mathod

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    W = X_cap + U_cap  # length = X_b + U_b * k, 将带标签的数据和伪标签数据合并
    random.shuffle(W)  # 随机打乱顺序

    #  W: L 与 L+UL的混合数据   做 mix up
    #  X: L 与 L的混合数据   做 mix up
    if x_mixup_mode == 'w':
        X_prime = [mix_up(X_cap[i], W[i], alpha) for i in range(X_b)]  # 只取带标签的batch_size数进行融合
    elif x_mixup_mode == 'x':
        idxs = np.random.permutation(range(X_b))  # 保证取的范围在带标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(U_b * K))[:X_b]  # 保证取的范围在不标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == '_':
        X_prime = X_cap
    elif x_mixup_mode == 's':
        X_prime = [mix_up(X_cap[i], X_cap[i], alpha) for i in range(X_b)]
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'w':  # 扣除前X_b个, 剩下进行融合
        U_prime = [mix_up(U_cap[i], W[X_b + i], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'x':  # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'u':  # 保证取的范围在不带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(U_b * K)]  # 有問題???
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')
    # print(U_cap[0][0].shape, U_cap[0][1].shape, X_cap[0][0].shape, X_cap[0][1].shape)
    # if DEBUG:
    # save_as_image(np.array([x[0] for x in U_prime]), f"../debug_output/u_prime_data")
    # save_as_image(np.array([x[1][[1], :, :, :] for x in U_prime]), f"../debug_output/u_prime_label")
    return X_prime, U_prime, pseudo_label


def mix_match_4(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor):
    """
    参考论文 MixMatch: A Holistic Approach to Semi-Supervised Learning
    """
    # X is labeled data of size BATCH_SIZE, and U is unlabeled data
    # X is list of tuples (data, label), and U is list of data
    # where data and label are of shape (C, D, H, W), numpy array. C of data is 1 and C of label is 2 (one hot)
    X_b = len(X)
    U_b = len(U)  # 应该是得到未标签的batch_size数才对

    # step 1: Augmentation
    # 随机在输入上增加扰动
    X_cap = [(augmentation(x[0], aug_factor), x[1]) for x in X]
    # batchsize翻K倍
    U_cap = U.repeat(K, 1, 1, 1, 1)  # [K*b, 1, D, H, W]
    # 一样加上一些随机扰动
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)  # augmented.
    # step 2: label guessing
    with torch.no_grad():
        # Y_u = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u,_,_,_ = eval_net(U_cap)  # (K, n_class, 112, 112, 80), (K, n_class, 112, 112, 80) 已经是one_hot的形式了
        Y_u = F.softmax(Y_u, dim=1)
        # print(Y_u.shape)
    guessed = torch.zeros(U.shape).repeat(1, K, 1, 1, 1)  # empty label [b, 2, D, H, W]  这边是做成one-hot形式了

    # if GPU:
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K  # 将多次重复算个平均结果  原文的公式(6)



    # 此外在做一些处理, 得到伪标签
    # guessed_t = guessed ** (1 / T)  # 原文的公式(7)
    # guessed = guessed_t / ()
    # guessed = guessed / guessed.sum(dim=1, keepdim=True)  # 原文的公式(7)
    pse_dis1 = guessed ** (1 / 0.2)
    pse_dis2 = (1 - guessed) ** (1 / 0.2)
    guessed = pse_dis1 / (pse_dis1 + pse_dis2)
    guessed = guessed.repeat(K, 1, 1, 1, 1)


    guessed = guessed.detach().cpu().numpy()
    # shape [U_b * K,2,D,H,W]  得到伪标签
    pseudo_label = guessed
    U_cap = U_cap.detach().cpu().numpy()

    # all_input = np.concatenate([x_in, U_cap], axis=0)
    # all_label = np.concatenate([x_lab, guessed], axis=0)

    U_cap = list(zip(U_cap, guessed))  # 将伪标签和扰动数据合并
    ## Now we have X_cap ,list of (data, label) of length b, U_cap, list of (data, guessed_label) of length k*b
    # step 3: MixUp
    # original paper mathod

    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    W = X_cap + U_cap  # length = X_b + U_b * k, 将带标签的数据和伪标签数据合并
    random.shuffle(W)  # 随机打乱顺序

    if x_mixup_mode == 'w':
        X_prime = [mix_up(X_cap[i], W[i], alpha) for i in range(X_b)]  # 只取带标签的batch_size数进行融合
    elif x_mixup_mode == 'x':
        idxs = np.random.permutation(range(X_b))  # 保证取的范围在带标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], X_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == 'u':
        idxs = np.random.permutation(range(U_b * K))[:X_b]  # 保证取的范围在不标签的batch_size数的范围内, 取带标签的batch_size数个进行融合
        X_prime = [mix_up(X_cap[i], U_cap[idxs[i]], alpha) for i in range(X_b)]
    elif x_mixup_mode == '_':
        X_prime = X_cap
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'w':  # 扣除前X_b个, 剩下进行融合
        U_prime = [mix_up(U_cap[i], W[X_b + i], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'x':  # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == 'u':  # 保证取的范围在不带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K))
        U_prime = [mix_up(U_cap[i], U_cap[idxs[i]], alpha) for i in range(U_b * K)]  # 有問題???
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')
    # print(U_cap[0][0].shape, U_cap[0][1].shape, X_cap[0][0].shape, X_cap[0][1].shape)
    # if DEBUG:
    # save_as_image(np.array([x[0] for x in U_prime]), f"../debug_output/u_prime_data")
    # save_as_image(np.array([x[1][[1], :, :, :] for x in U_prime]), f"../debug_output/u_prime_label")
    return X_prime, U_prime, pseudo_label


def mix_up_only_img(s1, s2, alpha):
    # print('??????', s1[0].shape, s1[1].shape, s2[0].shape, s2[1].shape)
    # s1, s2 are tuples(data, label)
    l = np.random.beta(alpha, alpha)  # 原文公式(8)
    l = max(l, 1 - l)  # 原文公式(9)

    x1 = s1
    x2 = s2
    # x1, p1 = s1
    # x2, p2 = s2

    x = l * x1 + (1 - l) * x2  # 原文公式(10)
    # p = l * p1 + (1 - l) * p2  # 原文公式(11)

    return x

def mix_up(s1, s2, alpha):
    # print('??????', s1[0].shape, s1[1].shape, s2[0].shape, s2[1].shape)
    # s1, s2 are tuples(data, label)
    l = np.random.beta(alpha, alpha)  # 原文公式(8)
    l = max(l, 1 - l)  # 原文公式(9)

    x1, p1 = s1
    x2, p2 = s2

    x = l * x1 + (1 - l) * x2  # 原文公式(10)
    p = l * p1 + (1 - l) * p2  # 原文公式(11)

    return (x, p)

def mix_up_torch(inputs1,inputs2, alpha):

    """
    对输入的数据和标签进行 mixup 处理
    :param inputs: 输入的数据 Tensor
    :param targets: 对应的标签 Tensor
    :param alpha: mixup 的参数，默认为 1.0
    :return: 处理后的数据和标签
    """
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0
    lam = torch.tensor(lam).cuda()
    # print('fds')
    mixed_inputs = lam * inputs1 + (1 - lam) * inputs2
    # mixed_targets = lam * targets1 + (1 - lam) * targets2

    return mixed_inputs#, mixed_targets

