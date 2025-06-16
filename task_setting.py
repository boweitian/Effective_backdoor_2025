import numpy as np
import torch
import torchvision
from torchvision import transforms

from models import  *
import gtsrb_dataset as dataset
import vggflower
from utils import default_seed
from functools import partial
from timm.models.vision_transformer import VisionTransformer , _cfg, vit_large_patch16_224

np.random.seed(default_seed)
torch.manual_seed(default_seed)


def option_select(mode, opacity):
    if mode == 1:
        # JSAC
        mask_mode = 1
        alternate = False
        invisible_train = False
        opa = 0.
    elif mode == 2:
        mask_mode = 2
        alternate = False
        invisible_train = False
        opa = 0.
    elif mode == 3:
        mask_mode = 2
        alternate = True
        invisible_train = False
        opa = 0.
    elif mode == 4:
        mask_mode = 2
        alternate = True
        invisible_train = True
        opa = opacity

    return mask_mode, alternate, invisible_train, opa


def get_dataset(name, mode):
    global trainset, testset, h, w, classes_num
    if name == 'cifar10':
        # ck_pth = './pytorch-cifar10-master/pytorch-cifar10-master/checkpoint/ckpt_5_5_vgg16.pth' #debug!!!!!
        #-------------------debug!!!!!
        ck_pth = './pytorch-cifar10-master/pytorch-cifar10-master/checkpoint/ckpt_5_5_vgg160000.pth' #debug!!!!!
        #-------------------debug!!!!!
        ran_pth = './model_92_sgd_cifar10.pkl'
        # target_layer = 'classifier' # debug!!!!!
        #-------------------debug!!!!!
        target_layer = 'patch_embed.proj'
        # patch_embed
        # patch_embed.proj
        # patch_embed.norm
        #-------------------debug!!!!!
        rt_epoches = 150
        target_class = 2
        lr = 1e-3  # should be adjusted to the trigger size and poison rate
        lambda1 = (0, 0.5, 2.4)
        # lambda1[0] for the functional loss, set to 0, for that we don'tcare about the trigger will be catogorized to which class
        # lambda1[1] for the QoE loss
        # lambda1[2] for the activation loss, maximise the bond between trigger and neurons
        stepping = [(60, 3, 3), (100, 6, 6), (60, 3, 3)]
        # model = VGG('VGG16')  # Cifar-10 # debug!!!!!
        #-------------------debug!!!!!
        model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6))
        #-------------------debug!!!!!
        # trigger = np.random.rand(3, 32, 32) #debug!!!!!
        trigger = np.random.rand(3, 224, 224)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(60),
            transforms.Resize((224, 224)), # debug!!!!!
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),# debug!!!!!
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform = [transform_train, transform_test]
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=None)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=None)
        h, w = 32, 32
        classes_num = 10
        inv_normalize = transforms.Normalize((-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010),
                                             (1 / 0.2023, 1 / 0.1994, 1 / 0.2010))
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    elif name == 'mnist':
        ck_pth = './base_model_train/mnist_lenet_models_no_transform_state_dict/mnist_lenet_acc_0.99000.pkl'
        ran_pth = './model_92_sgd_mnist.pkl'
        # model = LeNet()  # MNIST
        model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)) # debug!!!!!
        target_layer = 'head'
        rt_epoches = 100
        lr = 1e-4
        target_class = 8
        lambda1 = (0, 0.5, 2.4)
        stepping = [(50, 50, 50), (60, 6, 6), (100, 3, 3), (30, 6, 6), (100, 3, 3), (200, 1, 1)]
        # trigger = np.random.rand(1, 28, 28)
        trigger = np.random.rand(3, 224, 224)
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)), # debug!!!!!
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),# im_debug!!!!!
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)), # debug!!!!!
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),# im_debug!!!!!
        ])
        transform = [transform_train, transform_test]
        trainset = torchvision.datasets.MNIST(
            root='./ResidualAttentionNetwork-pytorch-master/Residual-Attention-Network/data', train=True,
            download=True, transform=None)

        testset = torchvision.datasets.MNIST(
            root='./ResidualAttentionNetwork-pytorch-master/Residual-Attention-Network/data', train=False,
            download=True, transform=None)
        h, w = 28, 28
        classes_num = 10
        inv_normalize = None
        normalize = None
    elif name == 'cifar100':
        ck_pth = './pytorch-cifar100-master/checkpoint/resnet50/Thursday_01_July_2021_10h_38m_57s/resnet50-187-best000.pth'
        ran_pth = './Atte92-176-best.pth'
        # model = resnet50()
        model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)) # debug!!!!!
        target_layer = 'head'
        rt_epoches = 20
        target_class = 0
        lr = 1e-4  # should be adjusted to the trigger size and poison rate
        lambda1 = (0, 0.5, 2.4)
        stepping = [(50, 50, 50), (15, 6, 6), (55, 1, 1), (6, 3, 3), (10, 6, 6), (6, 3, 3)]
        # (rounds, lr_ini, lr_ult) the steps and lrs, we adapt the trojan nn's setting
        # add a larger-lr stepping phase, to lengthen the exploration time.
        # stepping = [(0, 0, 0)]
        trainset = torchvision.datasets.CIFAR100(root='./pytorch-cifar100-master/data', train=True,
                                                 download=True, transform=None)

        testset = torchvision.datasets.CIFAR100(root='./pytorch-cifar100-master/data', train=False,
                                                download=True, transform=None)
        # trigger = np.random.rand(3, 32, 32)
        trigger = np.random.rand(3, 224, 224)
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize((224, 224)),# debug!!!!!
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),# debug!!!!!
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform = [transform_train, transform_test]

        h, w = 32, 32
        classes_num = 100
        inv_normalize = transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                             (1 / std[0], 1 / std[1], 1 / std[2]))
        normalize = transforms.Normalize(mean, std)
    elif name == 'gtsrb':
        ck_pth = './base_model_train/gtsrb_resnet34_models_2/gtsrb_resnet34_acc_97.245000.pkl'
        ran_pth = './gtsrb_RAN_acc_94.61.pkl'
        # target_layer = 'fc'
        target_layer = 'head'
        rt_epoches = 60
        target_class = 10
        lr = 1e-4  # should be adjusted to the trigger size and poison rate
        lambda1 = (0, 7.5, 1.2)
        # model = resnet34(43)
        model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)) # debug!!!!!
        stepping = [(30, 50, 50), (60, 6, 6), (100, 1, 1), (60, 3, 3), (40, 6, 6), (60, 3, 3)]
        mean, std = (0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669)
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize((224, 224)),# debug!!!!!
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            # transforms.Resize((32, 32)),
            transforms.Resize((224, 224)),# debug!!!!!
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = dataset.GTSRB(
            root_dir='./', train=True, transform=None)
        testset = dataset.GTSRB(
            root_dir='./', train=False, transform=None)
        # trigger = np.random.rand(3, 32, 32)
        trigger = np.random.rand(3, 224, 224)
        transform = [transform_train, transform_test]

        h, w = 32, 32
        classes_num = 43
        inv_normalize = transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                             (1 / std[0], 1 / std[1], 1 / std[2]))
        normalize = transforms.Normalize(mean, std)
    elif name == 'vggflower':
        ck_pth = '/home/ubuntu/Project/Attention-Backdoor/cifar10/base_model_train/vgg_flower_vgg16_models/vgg_flower_vgg16_acc_87.50.pkl'
        ran_pth = './vgg_flower_RAN_acc_83.50.pkl'
        # target_layer = 'classifier.6'
        target_layer = 'head'
        rt_epoches = 60 if mode == 1 else 100
        target_class = 0
        lr = 1e-4  # should be adjusted to the trigger size and poison rate
        lambda1 = (0, 0.5, 2.4)
        from torchvision import models
        # model = models.vgg16_bn(num_classes=10)
        model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)) # debug!!!!!
        stepping = [(150, 5, 5), (60, 3, 3), (100, 1, 1), (60, 2, 2), (100, 1, 1)]
        mean, std = (0.259197, 0.26592064, 0.27545887), (0.20904204, 0.21091025, 0.21665965)
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.Resize((224, 224)),# debug!!!!!
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            # transforms.Resize((32, 32)),
            transforms.Resize((224, 224)),# debug!!!!!
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = vggflower.VGGFlower(train=True, transform=None)
        testset = vggflower.VGGFlower(train=False, transform=None)

        # trigger = np.random.rand(3, 32, 32)
        trigger = np.random.rand(3, 224, 224)
        transform = [transform_train, transform_test]

        h, w = 32, 32
        classes_num = 10
        inv_normalize = transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                             (1 / std[0], 1 / std[1], 1 / std[2]))
        normalize = transforms.Normalize(mean, std)
    elif name == 'imagenette':
        ck_pth = '/home/ubuntu/Project/Attention-Backdoor/cifar10/pytorch-cifar100-master/checkpoint/resnet18/Tuesday_26_October_2021_02h_53m_34s/resnet18-123-best.pth'
        ran_pth = './attention92-139-best.pth'
        # target_layer = 'fc'
        target_layer = 'head'
        rt_epoches = 40
        target_class = 3
        lr = 1e-4  # should be adjusted to the trigger size and poison rate
        lambda1 = (0, 0.5, 0.8)
        # model = resnet18(10)
        model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)) # debug!!!!!
        stepping = [(10, 300, 200), (10, 200, 100), (15, 200, 150), (5, 100, 50), (10, 100, 100)]
        # mean, std = (0.4316, 0.427, 0.3997), (0.2939, 0.2896, 0.3076)
        mean, std = (0, 0 , 0), (1, 1, 1)
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        from ImageNette import ImageNette
        trainset = ImageNette(True, transform=None)
        testset = ImageNette(False, transform=None)
        trigger = np.random.rand(3, 224, 224)
        transform = [transform_train, transform_test]

        h, w = 224, 224
        classes_num = 10
        inv_normalize = transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                             (1 / std[0], 1 / std[1], 1 / std[2]))
        normalize = transforms.Normalize(mean, std)
    elif name == 'vgg224':
        ck_pth = '/home/ubuntu/Project/Attention-Backdoor/cifar10/models/resnet18-142-best.pth'
        ran_pth = './attention92-136-best.pth'
        # target_layer = 'fc'
        target_layer = 'head'
        rt_epoches = 60
        target_class = 4
        lr = 1e-4  # should be adjusted to the trigger size and poison rate
        lambda1 = (0, 0.5, 2.4)
        # model = resnet18(10)
        model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6)) # debug!!!!!
        stepping = [(150, 50, 25), (60, 25, 15), (100, 15, 10), (60, 10, 5), (100, 5, 1)]
        mean = (0.4059, 0.3556, 0.2614)
        std = (0.3077, 0.2542, 0.2688)
        transform_train = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        trainset = vggflower.VGGFlower(train=True, transform=None)
        testset = vggflower.VGGFlower(train=False, transform=None)

        trigger = np.random.rand(3, 224, 224)
        transform = [transform_train, transform_test]

        h, w = 224, 224
        classes_num = 10
        inv_normalize = transforms.Normalize((-mean[0] / std[0], -mean[1] / std[1], -mean[2] / std[2]),
                                             (1 / std[0], 1 / std[1], 1 / std[2]))
        normalize = transforms.Normalize(mean, std)

    return lr, lambda1, target_class, ck_pth, ran_pth, target_layer, rt_epoches, stepping, model, trigger, trainset, testset, h, w, classes_num, transform, normalize, inv_normalize