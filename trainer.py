# coding: utf-8
import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from task.train_embedding_modal_adapter import embedding_modal_adapter_train
from task.train_sequential_modal_adapter import sequential_modal_adapter_train

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default=None, help="Task parameter: None for normal training; others for special tasks")
    parser.add_argument('--saved_model', default='.', help="path to save models")
    parser.add_argument('--exp_name', help='folder under opt.saved_model, to show parameters briefly')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saveInterval', type=int, default=100000, help='Interval between each fixed step model saving')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/', help='select training data')
    parser.add_argument('--batch_ratio', type=str, default='1', help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--src_batch_max_length', type=int, default=25, help='maximum-label-length of src')
    parser.add_argument('--tgt_batch_max_length', type=int, default=25, help='maximum-label-length of tgt')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=False, default='TPS', help='Transformation stage.')
    parser.add_argument('--FeatureExtraction', type=str, required=False, default='ResNoLSTM', help='FeatureExtraction stage.')
    parser.add_argument('--SequenceModeling', type=str, required=False, default='TransformerEncoder', help='SequenceModeling stage.')
    parser.add_argument('--Prediction', type=str, required=False, default='TransformerDecoder', help='Prediction stage.')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--src_vocab', required=True, help='path to source vocab')
    parser.add_argument('--tgt_vocab', required=True, help='path to target vocab')
    parser.add_argument('--src_level', type=str, default="char", help='char or word of src')
    parser.add_argument('--tgt_level', type=str, default="char", help='char or word of tgt')
    
    parser.add_argument("--TIMT_Weight", type = float, default = 1.0)
    
    parser.add_argument('--CL_tau', type=float, default=1.0, help='parameter of temperature for contrastive learning.')
    parser.add_argument('--CL_tau_vv', type=float, default=0.2, help='parameter of temperature for vision-vision contrastive learning.')
    parser.add_argument('--CL_tau_vl', type=float, default=0.2, help='parameter of temperature for vision-lingual contrastive learning.')
    parser.add_argument('--CL_tau_ll', type=float, default=0.2, help='parameter of temperature for lingual-lingual contrastive learning.')
    parser.add_argument('--CL_type', type=str, default="mcl", help='hyper-parameter to distinguish contrastive learning type.')
    parser.add_argument('--CL_Weight', type=float, default=-1, help='hyperparameter before contrastive learning cost')
    parser.add_argument('--CL_vv_Weight', type=float, default=1, help='hyperparameter of vision-vision contrastive learning cost')
    parser.add_argument('--CL_vl_Weight', type=float, default=1, help='hyperparameter of vision-lingual contrastive learning cost')
    parser.add_argument('--CL_ll_Weight', type=float, default=1, help='hyperparameter of lingual-lingual contrastive learning cost')
    
    parser.add_argument('--adapter_layer_num', type=int, default=6, help='the number of adapter layers.')
    parser.add_argument('--adapter_hidden_size', type=int, default=256, help='the hidden size of adapter layers.')
    parser.add_argument('--teacher_path', type=str, default="./", help='the path of teacher model.')
    parser.add_argument('--teacher_iter', type=str, default="final", help='the iter of teacher model.')

    parser.add_argument('--image_encoder_tuning', type=str, default="False", help='whether fine-tune image encoder or not.')
    parser.add_argument('--text_encoder_tuning', type=str, default="False", help='whether fine-tune text encoder or not.')
    parser.add_argument('--ocr_tr_encoder_tuning', type=str, default="False", help='whether fine-tune ocr transformer encoder or not.')
    parser.add_argument('--ocr_tr_decoder_tuning', type=str, default="False", help='whether fine-tune ocr transformer encoder or not.')
    parser.add_argument('--mt_tr_encoder_tuning', type=str, default="False", help='whether fine-tune mt transformer encoder or not.')
    parser.add_argument('--mt_tr_decoder_tuning', type=str, default="False", help='whether fine-tune mt transformer encoder or not.')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'

    os.makedirs(f'{opt.saved_model}/{opt.exp_name}', exist_ok=True)

    def pic_file_texts(file):
        id=[]
        for line in open(file, 'r', encoding='utf-8'):
            line = line.replace("\n", "")
            id.append(line)
        return id

    if opt.src_vocab:
        src_dict_ = pic_file_texts(opt.src_vocab)
        opt.src_character = src_dict_
    else:
        print(f"Please input a valid src vocab.")
    
    if opt.tgt_vocab:
        tgt_dict_ = pic_file_texts(opt.tgt_vocab)
        opt.tgt_character = tgt_dict_
    else:
        print(f"Please input a valid tgt vocab.")

    """ Seed and GPU setting """
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    print('*' * 50)
    print('Available GPU Device Count: {}'.format(opt.num_gpu))
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu
    
    if opt.task == "emb_ma_tuning" or opt.task == "embedding_modal_adapter":
        embedding_modal_adapter_train(opt)
    elif opt.task == "seq_ma_tuning" or opt.task == "sequential_modal_adapter":
        sequential_modal_adapter_train(opt) 
    else:
        print(f"Input task {opt.task} is not defined. Please check your task name.")
        exit()
        