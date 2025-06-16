from PIL import Image
import numpy as np
from torch import nn, optim
import os
import torch

default_seed = 123456


def array2img(x):
    a = np.array(x)
    img = Image.fromarray(a.astype('uint8'), 'RGB')
    # img.show()
    return img


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion




def optimizer_picker(optimization, param, lr=1e-4, momentum=1e-7, decay=0.9, nesterov=True):
    if optimization == 'adam':
        optimizer = optim.Adam(param,lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum, weight_decay=decay, nesterov=nesterov)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param,lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    return optimizer

def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std

def pth_generator(datasetname, tm, mask_mode, alternate, multi_neuron, loss_function, poison_rate, pixel_nums,
                  mask_opacity, multi=False):
    global f
    if not os.path.isdir('./intermediate'):
        os.mkdir('./intermediate')
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.isdir('./logs/multi'):
        os.mkdir('./logs/multi')
    if not os.path.isdir('./logs/multi/' + datasetname):
        os.mkdir('./logs/multi/' + datasetname)

    if multi:
        root = './logs/multi/'
    else:
        root = './logs/'

    os.mkdir(root+ datasetname + '/ ' + str(tm) + '_mask_' + str(mask_mode) + '_alternate_' + str(
        alternate) + '_multi_neuron_' + str(
        multi_neuron) + '_loss_' + str(loss_function) + '_' + str(poison_rate) + '_' + str(
        pixel_nums[0] * pixel_nums[1]) + '_' + str(mask_opacity))
    f = open(root + datasetname + '/ ' + str(tm) + '_mask_' + str(mask_mode) + '_alternate_' + str(
        alternate) + '_multi_neuron_' + str(
        multi_neuron) + '_loss_' + str(loss_function) + '_' + str(poison_rate) + '_' + str(
        pixel_nums[0] * pixel_nums[1]) + '_' + str(mask_opacity) + '/rt_log.txt', 'a')

    return root + datasetname + '/ ' + str(tm) + '_mask_' + str(mask_mode) + '_alternate_' + str(
        alternate) + '_multi_neuron_' + str(
        multi_neuron) + '_loss_' + str(loss_function) + '_' + str(poison_rate) + '_' + str(
        pixel_nums[0] * pixel_nums[1]) + '_' + str(mask_opacity) + '/', \
           root + datasetname + '/ ' + str(tm) + '_mask_' + str(mask_mode) + '_alternate_' + str(
               alternate) + '_multi_neuron_' + str(
               multi_neuron) + '_loss_' + str(loss_function) + '_' + str(poison_rate) + '_' + str(
               pixel_nums[0] * pixel_nums[1]) + '_' + str(mask_opacity) + '/log.txt', \
           root + datasetname + '/ ' + str(tm) + '_mask_' + str(mask_mode) + '_alternate_' + str(
               alternate) + '_multi_neuron_' + str(
               multi_neuron) + '_loss_' + str(loss_function) + '_' + str(poison_rate) + '_' + str(
               pixel_nums[0] * pixel_nums[1]) + '_' + str(mask_opacity) + '/ck.pth',\
        f





def load_model_state_vis(model, pretrained_model_pth, device=None):
    # model_dict = torch.load(pretrained_model_pth, map_location=device)
    # new_state_dict = {}
    # for k, v in model_dict.items():
    #     if 'module.' in k:
    #         name = k[7:]  # remove `module.` in the data
    #     else:
    #         name = k
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict) # debug!!!!!!!!!!!!
    #-------------------debug!!!!!
    checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
    model.load_state_dict(checkpoint["model"])
    #-------------------debug!!!!!
    model.cuda()
    print("Model is ready.")
    return model

def load_model_state(model, pretrained_model_pth, device=None):
    model_dict = torch.load(pretrained_model_pth, map_location=device)
    new_state_dict = {}
    for k, v in model_dict.items():
        if 'module.' in k:
            name = k[7:]  # remove `module.` in the data
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.cuda()
    print("Model is ready.")
    return model

