import sys
import os
from optparse import OptionParser
import numpy as np
from tqdm import tqdm
import torch
from lib.param_compress_new2 import Decode_Params, define_global
from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch
from torch.profiler import profile, record_function, ProfilerActivity

def eval_net_carvana(net,
              val_percent=0.05,
              gpu=False,
              img_scale=0.5):

    dir_img = './data/carvana-image-masking-challenge/train/'
    dir_mask = './data/carvana-image-masking-challenge/train_masks/'

    ids = get_ids(dir_img)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)

    val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)
    val_dice = eval_net(net, val, gpu)
    print(val_dice)



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options




if __name__ == '__main__':

    Model_Name = 'HyperUNet_V2'

    args = get_args()
    model = UNet(n_channels=3, n_classes=1, f_channels=f'./lib/HyperUNet_channels.txt')

    model.load_state_dict(torch.load(f'./lib/UNet_OriParams.pth')) # 加载原模型参数

    if Model_Name[-2:] == 'V1':
        codebook_num = 65535
    else:
        codebook_num = 7130


    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        define_global(codebook_num, 64, 2, 4)
        decode_params = Decode_Params(f'./Params_Save/{Model_Name}/Compressed_Dir/') # 还原参数


        params_list = list(model.parameters())
        for i in tqdm(range(len(params_list))):
            ori_param = params_list[i].data # 原参数
            new_param = decode_params[i] # 还原的新参数
            params_list[i] = torch.tensor(new_param).float().cuda() # 用还原参数替换原模型参数


        if args.gpu:
            model.cuda()
            # cudnn.benchmark = True # faster convolutions, but more memory

        eval_net_carvana(net=model,
                gpu=args.gpu,
                img_scale=args.scale)
