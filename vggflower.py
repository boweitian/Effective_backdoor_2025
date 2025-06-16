#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import os.path as osp
from collections import defaultdict as dd

import numpy as np

from torchvision.datasets.folder import ImageFolder




class VGGFlower(ImageFolder):

    def __init__(self, train=True, transform=None, target_transform=None):
        # root = '/home/ubuntu/Project/Attention-Backdoor/cifar10/data/102_flowers_catogoty'
        root = '/mnt/data/ywb_bk/Dataset/VGGFlower'
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
            ))

        # Initialize ImageFolder
        super().__init__(root=root, transform=transform, target_transform=target_transform)


        self.ntest = 20  # Reserve these many examples per class for evaluation
        self.partition_to_idxs = self.get_partition_to_idxs()
        self.pruned_idxs = self.partition_to_idxs['train' if train else 'test']

        # Prune (self.imgs, self.samples to only include examples from the required train/test partition
        self.samples = [self.samples[i] for i in self.pruned_idxs]
        self.imgs = self.samples

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))


    def get_partition_to_idxs(self):
        partition_to_idxs = {
            'train': [],
            'test': []
        }

        # Use this random seed to make partition consistent
        prev_state = np.random.get_state()
        np.random.seed(123)

        # ----------------- Create mapping: classidx -> idx
        classidx_to_idxs = dd(list)
        for idx, s in enumerate(self.samples):
            classidx = s[1]
            classidx_to_idxs[classidx].append(idx)

        # Shuffle classidx_to_idx
        for classidx, idxs in classidx_to_idxs.items():
            np.random.shuffle(idxs)

        for classidx, idxs in classidx_to_idxs.items():
            partition_to_idxs['test'] += idxs[:self.ntest]     # A constant no. kept aside for evaluation
            partition_to_idxs['train'] += idxs[self.ntest:]    # Train on remaining

        # Revert randomness to original state
        np.random.set_state(prev_state)

        return partition_to_idxs

