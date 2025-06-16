import os
import gc
import sys
import copy
import time
import numpy as np
import settings
import argparse

import torch
from torchvision import utils as vutils

from task_setting import get_dataset, option_select
from models import *
from utils import optimizer_picker, pth_generator, load_model_state
from trojan_helper.helper import IMC_Attention_Helper

np.set_printoptions(threshold=np.inf)
np.random.seed(settings.default_seed)
torch.manual_seed(settings.default_seed)
torch.cuda.manual_seed(settings.default_seed)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=str, required=True, help='Specfiy the task (vggflower/cifar10/gtsrb/cifar100/imagenette).')
    parser.add_argument('-d', type=str, required=True, help='Specfiy the gpu device, 0 for cpu, num > 0 for gpu')
    parser.add_argument('-prate', type=float, default= 0.05, help='Set the propotion of poisoned imgs')
    parser.add_argument('-pnum', type=int, default=4, help='Set the trigger size, eg 4 is inputed, the trigger will be 4 * 4')
    parser.add_argument('-opac', type=float, default=0.2, help='Set the opacity of the trigger, [0, 1).')
    parser.add_argument('-mode', type=int, default=4, help='Attack mode assignment. 1 - RobNet baseline, 2 - Attention Mechanism + Co-optimization, 3 - Attention + Co-optimization + Alternate retrain, 4 - Ours(3 + invisivility loss function)')
    parser.add_argument('-iter', type=int, default=4, help='Specfiy the task (vggflower/cifar10/gtsrb/cifar100).')
    parser.add_argument('-b', type=int, default=64, help='Batch Size for model retrain')
    parser.add_argument('-nw', type=int, default=2, help='Num_worker for pytorch dataloader')
    parser.add_argument('-lr', type=float, default= 1e-3, help='learning rate for model retrain')
    parser.add_argument('-e', type=int, default=30, help='the num of epoches for a single retrain round')

    time0 = time.time()

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.d
    device = torch.device("cuda")

    # ---------------------------1、Determine the args-----------------------------
    dataset_nm = args.task.lower()
    loss_list = ['efficacy', 'specificity']
    threshold = -0.1  # assign the threshold which we can bear of the retrain laziness
    loss_mode = 'cross'
    optimize_mode = 'adam'
    use_cuda = True
    batch_size = args.b
    num_workers = args.nw

    # ---------------------------2、Assign the attack mode-----------------------------
    # Attack mode nomination choice
    poison_rate = args.prate  # refer to how much of the benign dataset got polluted, note that the former imgs will also be added to mix-up dataset
    pixel_nums = (args.pnum, args.pnum) # The trigger size
    multi_neuron = False  # We adapt to choose multiple neurons instead of only one to generate the trigger
    neuron_num = 3
    mode = args.mode
    opacity = args.opac    # mask_opacity:  should be between [0, 1], float, 0 means completely patch
    rt_epoches = args.e
    lr = args.lr
    mask_mode, alternate, invisible_train, mask_opacity = option_select(mode=mode, opacity=opacity)
    # mask_mode: 1 for fixed mask(right bottom square), 2 for attention trigger
    # Alternate: True for alternate retrain, which we in turn train the model on the
    #            mix-up model and beneign model, it helps sustain the PA in a high level
    # invisible_train: Add a QoE loss(in ours, SSIM loss) to the trigger-generation one

    min_iter, max_iter = (1, 1) if mask_mode == 1 else (3, args.iter)
    file_pth, log_path, ckpt_path, file_stream = pth_generator(dataset_nm, time.time(), mask_mode, alternate, multi_neuron, invisible_train,
                                                  poison_rate,
                                                  pixel_nums, mask_opacity)



    # ---------------------------3、Load the dataset and model-----------------------------
    print('==> Start to load dataset')
    _, lambda1, target_class, pretrained_model_pth, ran_pth, target_layer, _, stepping, model, trigger, trainset_ori, testset, \
    h, w, num_classes, transform, normalize, inv_normalize = get_dataset(dataset_nm, mode)
    print('train set shape:', len(trainset_ori))
    print('test set shape :', len(testset))

    print('==> Start to load model.')
    model = load_model_state(model, pretrained_model_pth, device)

    # ---------------------------4、Configure the Helper-----------------------------
    print('==> Prepare a helper.')
    helper = \
        IMC_Attention_Helper(dataset_nm, h, w, use_cuda, log_path, ckpt_path, min_iter, max_iter, rt_epoches,
                             threshold, mask_mode, pixel_nums, ran_pth,
                             stepping, file_pth, target_layer, poison_rate,
                             loss_mode, optimize_mode, batch_size, num_workers, lr, neuron_num,
                             multi_neuron, 1 - mask_opacity, lambda1, num_classes, transform,
                             normalize, invisible_train, invisible_train, file_stream)
    helper.transform = transform
    helper.inv_normalize = inv_normalize
    helper.invisible_train = invisible_train
    # Note that, if we assign the opacity as x, the application in the code is p = (1 - x), so we have to change the opacity param here
    print('Helper is ready.')

    trigger = torch.from_numpy(trigger)
    trigger = trigger.float()
    if normalize != None:
        trigger = normalize(trigger)
    mask = helper.filter_apart(trainset_ori + testset, target_class)
    if hasattr(helper, 'RAN'):
        del helper.RAN
    print(mask, file=file_stream)


    # ---------------------------7、Conduct the Attack ----------------------------
    '''
           In this part, it will lead the whole algorithm.
           We got the initial trigger and backdoored model by a normal attention-backdoor attack
           And then we aim to optimize the two items by constantly further retrain.
           Until the process have converged, it will perform gen_trigger and retrain in each iteration.
           Note that: every iteration will utilize the benign datasets and the backdoored model got in last iter
    '''

    while helper.iter_counter == 0 or helper.not_converge_test():
        helper.bst_score = (0, 0, 0, 0, 0)
        trigger = helper.gen_trigger(model, trigger, mask, target_class, transform[1], trainset_ori, inv_normalize)
        trainset_poisoned = helper.poison(target_class, trainset_ori,
                                           trigger, mask, w, h, transform[0], train=True, inv_normalize=inv_normalize)
        gc.collect()
        testset_poisoned = helper.poison(target_class,  testset, trigger, mask, w, h, transform[1], train=False,
                                                                   inv_normalize=inv_normalize)
        gc.collect()
        # the x_poison have been transformed after the poison function
        # _initialize_weights(model)
        optimizer = optimizer_picker(helper.optim_mode, model.parameters(), lr=helper.lr)
        data_loader_train_poison = torch.utils.data.DataLoader(trainset_poisoned, batch_size=helper.batch_size, shuffle=True,
            num_workers=helper.num_workers, drop_last=False)
        trainset2 = copy.deepcopy(trainset_ori)
        trainset2.transform = transform[0]
        data_loader_train_ori = torch.utils.data.DataLoader(trainset2, batch_size=helper.batch_size, shuffle=True,
            num_workers=helper.num_workers, drop_last=False)
        testset2 = copy.deepcopy(testset)
        testset2.transform = transform[1]
        data_loader_test_ori = torch.utils.data.DataLoader(testset2,batch_size=helper.batch_size, shuffle=False,
            num_workers=helper.num_workers, drop_last=False)
        data_loader_test_p = torch.utils.data.DataLoader(testset_poisoned, batch_size=helper.batch_size, shuffle=False,
                                                         num_workers=helper.num_workers, drop_last=False)
        torch.save(data_loader_test_p, os.path.join(file_pth, 'poisoned_test_loader.pt'))
        gc.collect()
        # 引用也没变为0，所以也没删内存
        print('==> Start to retrain..')
        for e in range(helper.rt_epoches):
            print('[Epoch ' + str(e) + ']')
            print('[Epoch ' + str(e) + ']', file=file_stream)
            start_time = time.time()
            if e % 2 != 0 and alternate == True:
                helper.retrain_step(model, optimizer, data_loader_train_ori)
            else:
                helper.retrain_step(model, optimizer, data_loader_train_poison)
            helper.evaluation(loss_list, model, data_loader_test_ori, data_loader_test_p, trigger, mask,
                                       transform[1], e)
            end_time = time.time()
            print('Time Cost: ', (end_time - start_time))
            print('Time Cost: ', (end_time - start_time), file=file_stream)
            print('--------------------------------------------------')
            print('--------------------------------------------------', file=file_stream)
        helper.evaluation(['fidelity'], model, data_loader_test_ori, data_loader_test_p, trigger, mask, transform[1], 0)
        del data_loader_train_poison, data_loader_test_p, data_loader_test_ori, data_loader_train_ori
        gc.collect()

        helper.history.append(helper.bst_score)

        print('Retrain Finished')
        helper.iter_counter += 1

    helper.show_history()
    print('In sum, time cost', time.time() - time0)
main()