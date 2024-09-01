import os
import time
import string
import argparse
import re

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from contrastive_loss import contrastive_loss_cal_moco_v3
from utils import Averager
from model import make_std_mask
from nltk_bleu_local import doc_bleu

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validation_embedding_modal_adapter_task(model_list, criterion, evaluation_loader, src_converter, tgt_converter, opt):
    """ validation or evaluation """
    n_correct = 0
    n_total = 0
    ref_sents = []
    pred_sents = []
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    
    for i, (image_tensors, _, _, src_labels, tgt_labels, _, _) in enumerate(evaluation_loader):   
        print('Decoding batch {} in validation ...'.format(i+1))
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)
        # For max length prediction
        tgt_length_for_pred = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
        tgt_text_for_pred = torch.LongTensor(batch_size, opt.tgt_batch_max_length + 1).fill_(0).to(device)
        
        src_text_for_loss, src_length_for_loss = src_converter.encode(src_labels, opt.src_level, batch_max_length=opt.src_batch_max_length)
        tgt_text_for_loss, tgt_length_for_loss = tgt_converter.encode(tgt_labels, opt.tgt_level, batch_max_length=opt.tgt_batch_max_length)

        valid_tgt_mask = make_std_mask(tgt_text_for_loss[:, :-1], pad = 2)
        valid_src_mask = opt.src_mask
        valid_tgt_mask = opt.tgt_mask

        if opt.num_gpu > 1:
            print('Now processing tgt_mask to meet multi-gpu training ...')
            x_tgt_mask, y_tgt_mask = valid_tgt_mask.size()
            new_tgt_mask = valid_tgt_mask.repeat(opt.batch_size, 1)
            new_tgt_mask = new_tgt_mask.reshape(opt.batch_size, x_tgt_mask, y_tgt_mask)
            valid_tgt_mask = new_tgt_mask
        
        start_time = time.time()
        
        if 'TransformerDecoder' in opt.Prediction:
            vtmt_decoder_input = tgt_text_for_pred
            mt_decoder_input = tgt_text_for_pred
            for i in range(opt.tgt_batch_max_length + 1):

                visual_feature = model_list[0](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                textual_feature = model_list[1](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                visual_adapter_feature = model_list[3](visual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)

                visual_contextual_feature = model_list[2](visual_adapter_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                textual_contextual_feature = model_list[2](textual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)                
                
                vtmt_preds = model_list[4](contextual_feature = visual_contextual_feature, input = image, text = vtmt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
                mt_preds = model_list[4](contextual_feature = textual_contextual_feature, input = image, text = mt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
                
                _, vtmt_preds_index = vtmt_preds.max(2)
                _, mt_preds_index = mt_preds.max(2)
                
                if i+1 < opt.tgt_batch_max_length + 1:
                    vtmt_decoder_input[:, i+1] = vtmt_preds_index[:, i]
                    mt_decoder_input[:, i+1] = mt_preds_index[:, i]

            forward_time = time.time() - start_time
            
            vtmt_preds = vtmt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
            mt_preds = mt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
            
            tgt_target = tgt_text_for_loss[:, 1:]  # without [GO] Symbol
            vtmt_cost = criterion(vtmt_preds.contiguous().view(-1, vtmt_preds.shape[-1]), tgt_target.contiguous().view(-1))

            fv = torch.sum(visual_adapter_feature, dim = 1) / visual_adapter_feature.shape[1]
            ft = torch.sum(textual_contextual_feature, dim = 1) / textual_contextual_feature.shape[1]
            
            cl_cost = contrastive_loss_cal_moco_v3(fv, ft, opt.CL_tau)

            # select max probabilty (greedy decoding) then decode index to character
            _, vtmt_preds_index = vtmt_preds.max(2)
            vtmt_preds_str = tgt_converter.decode(vtmt_preds_index, tgt_length_for_pred, opt.tgt_level)
            
            _, mt_preds_index = mt_preds.max(2)
            mt_preds_str = tgt_converter.decode(mt_preds_index, tgt_length_for_pred, opt.tgt_level)
            src_labels = src_converter.decode(src_text_for_loss[:, 1:], src_length_for_loss, opt.src_level)
            tgt_labels = tgt_converter.decode(tgt_text_for_loss[:, 1:], tgt_length_for_loss, opt.tgt_level)
        
        else:
            # # Attn Decoding Procedure
            vtmt_decoder_input = tgt_text_for_pred
            mt_decoder_input = tgt_text_for_pred

            for i in range(opt.tgt_batch_max_length + 1):                
                visual_feature = model_list[0](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                textual_feature = model_list[1](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                visual_adapter_feature = model_list[3](visual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)

                visual_contextual_feature = model_list[2](visual_adapter_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                textual_contextual_feature = model_list[2](textual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                
                vtmt_preds = model_list[4](contextual_feature = visual_contextual_feature, input = image, text = vtmt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)

                _, vtmt_preds_index = vtmt_preds.max(2)
                
                if i+1 < opt.tgt_batch_max_length + 1:
                    vtmt_decoder_input[:, i+1] = vtmt_preds_index[:, i]

            forward_time = time.time() - start_time
            
            vtmt_preds = vtmt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
            mt_preds = mt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
            
            tgt_target = tgt_text_for_loss[:, 1:]  # without [GO] Symbol
            vtmt_cost = criterion(vtmt_preds.contiguous().view(-1, vtmt_preds.shape[-1]), tgt_target.contiguous().view(-1))

            fv = torch.sum(visual_adapter_feature, dim = 1) / visual_adapter_feature.shape[1]
            ft = torch.sum(textual_contextual_feature, dim = 1) / textual_contextual_feature.shape[1]
            
            cl_cost = contrastive_loss_cal_moco_v3(fv, ft, opt.CL_tau)

            # select max probabilty (greedy decoding) then decode index to character
            _, vtmt_preds_index = vtmt_preds.max(2)
            vtmt_preds_str = tgt_converter.decode(vtmt_preds_index, tgt_length_for_pred, opt.tgt_level)
            
            src_labels = src_converter.decode(src_text_for_loss[:, 1:], src_length_for_loss, opt.src_level)
            tgt_labels = tgt_converter.decode(tgt_text_for_loss[:, 1:], tgt_length_for_loss, opt.tgt_level)
            
        infer_time += forward_time
        
        vtmt_weight = opt.TIMT_Weight
        cl_weight = opt.CL_Weight
    
        weighted_vtmt_cost = vtmt_weight * vtmt_cost
        weighted_cl_cost = cl_weight * cl_cost
        
        valid_loss_avg.add(weighted_vtmt_cost)
        valid_loss_avg.add(weighted_cl_cost)
        
        for gt, pred in zip(tgt_labels, vtmt_preds_str):
            if 'CTC' not in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

            ref_sents.append(gt)
            pred_sents.append(pred)
            
            # Updated accuracy calculation
            # If the characters in gt occurs in pred, a positive support is given
            for item in gt:
                n_total += 1
                if item in pred:
                    n_correct += 1

    accuracy = n_correct / float(n_total) * 100
    tok_bleu, char_bleu = doc_bleu(ref_sents, pred_sents)
    
    return valid_loss_avg.val(), accuracy, tok_bleu, vtmt_preds_str, mt_preds_str, src_labels, tgt_labels, infer_time, length_of_data

def validation_sequential_modal_adapter_task(model_list, criterion, evaluation_loader, src_converter, tgt_converter, opt):
    """ validation or evaluation """
    n_correct = 0
    n_total = 0
    ref_sents = []
    pred_sents = []
    length_of_data = 0
    infer_time = 0
    valid_loss_avg = Averager()
    
    for i, (image_tensors, _, _, src_labels, tgt_labels, _, _) in enumerate(evaluation_loader):   
        print('Decoding batch {} in validation ...'.format(i+1))
        batch_size = image_tensors.size(0)
        length_of_data = length_of_data + batch_size
        image = image_tensors.to(device)

        tgt_length_for_pred = torch.IntTensor([opt.tgt_batch_max_length] * batch_size).to(device)
        tgt_text_for_pred = torch.LongTensor(batch_size, opt.tgt_batch_max_length + 1).fill_(0).to(device)
        
        src_text_for_loss, src_length_for_loss = src_converter.encode(src_labels, opt.src_level, batch_max_length=opt.src_batch_max_length)
        tgt_text_for_loss, tgt_length_for_loss = tgt_converter.encode(tgt_labels, opt.tgt_level, batch_max_length=opt.tgt_batch_max_length)
        
        valid_tgt_mask = make_std_mask(tgt_text_for_loss[:, :-1], pad = 2)
        valid_tgt_mask = opt.tgt_mask

        if opt.num_gpu > 1:
            print('Now processing tgt_mask to meet multi-gpu training ...')
            x_tgt_mask, y_tgt_mask = valid_tgt_mask.size()
            new_tgt_mask = valid_tgt_mask.repeat(opt.batch_size, 1)
            new_tgt_mask = new_tgt_mask.reshape(opt.batch_size, x_tgt_mask, y_tgt_mask)
            valid_tgt_mask = new_tgt_mask
        
        start_time = time.time()        
        if 'TransformerDecoder' in opt.Prediction:
            vtmt_decoder_input = tgt_text_for_pred
            mt_decoder_input = tgt_text_for_pred

            for i in range(opt.tgt_batch_max_length + 1):
                visual_feature = model_list[0](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                textual_feature = model_list[1](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                visual_contextual_feature = model_list[2](visual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                textual_contextual_feature = model_list[3](textual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                
                visual_adapter_feature = model_list[4](visual_contextual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                
                vtmt_preds = model_list[5](contextual_feature = visual_adapter_feature, input = image, text = vtmt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
                mt_preds = model_list[5](contextual_feature = textual_contextual_feature, input = image, text = mt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
                
                _, vtmt_preds_index = vtmt_preds.max(2)
                _, mt_preds_index = mt_preds.max(2)
                
                if i+1 < opt.tgt_batch_max_length + 1:
                    vtmt_decoder_input[:, i+1] = vtmt_preds_index[:, i]
                    mt_decoder_input[:, i+1] = mt_preds_index[:, i]

            forward_time = time.time() - start_time
            
            vtmt_preds = vtmt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
            mt_preds = mt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
            
            tgt_target = tgt_text_for_loss[:, 1:]  # without [GO] Symbol
            vtmt_cost = criterion(vtmt_preds.contiguous().view(-1, vtmt_preds.shape[-1]), tgt_target.contiguous().view(-1))

            fv = torch.sum(visual_adapter_feature, dim = 1) / visual_adapter_feature.shape[1]
            ft = torch.sum(textual_contextual_feature, dim = 1) / textual_contextual_feature.shape[1]
            cl_cost = contrastive_loss_cal_moco_v3(fv, ft, opt.CL_tau)
        
            # select max probabilty (greedy decoding) then decode index to character
            _, vtmt_preds_index = vtmt_preds.max(2)
            vtmt_preds_str = tgt_converter.decode(vtmt_preds_index, tgt_length_for_pred, opt.tgt_level)
            
            _, mt_preds_index = mt_preds.max(2)
            mt_preds_str = tgt_converter.decode(mt_preds_index, tgt_length_for_pred, opt.tgt_level)
            src_labels = src_converter.decode(src_text_for_loss[:, 1:], src_length_for_loss, opt.src_level)
            tgt_labels = tgt_converter.decode(tgt_text_for_loss[:, 1:], tgt_length_for_loss, opt.tgt_level)
        
        else:
            # # Attn Decoding Procedure
            vtmt_decoder_input = tgt_text_for_pred
            mt_decoder_input = tgt_text_for_pred

            for i in range(opt.tgt_batch_max_length + 1):
                visual_feature = model_list[0](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                textual_feature = model_list[1](input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                visual_contextual_feature = model_list[2](visual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                textual_contextual_feature = model_list[3](textual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                
                visual_adapter_feature = model_list[4](visual_contextual_feature, input = image, text = src_text_for_loss[:, :-1].long(), tgt_mask = valid_tgt_mask, is_train=False)
                
                vtmt_preds = model_list[5](contextual_feature = visual_adapter_feature, input = image, text = vtmt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
                mt_preds = model_list[5](contextual_feature = textual_contextual_feature, input = image, text = mt_decoder_input.long(), tgt_mask = valid_tgt_mask, is_train=False)
                
                _, vtmt_preds_index = vtmt_preds.max(2)
                _, mt_preds_index = mt_preds.max(2)
                
                if i+1 < opt.tgt_batch_max_length + 1:
                    vtmt_decoder_input[:, i+1] = vtmt_preds_index[:, i]
                    mt_decoder_input[:, i+1] = mt_preds_index[:, i]

            forward_time = time.time() - start_time

            vtmt_preds = vtmt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
            mt_preds = mt_preds[:, :tgt_text_for_loss.shape[1] - 1, :]
            
            tgt_target = tgt_text_for_loss[:, 1:]  # without [GO] Symbol
            vtmt_cost = criterion(vtmt_preds.contiguous().view(-1, vtmt_preds.shape[-1]), tgt_target.contiguous().view(-1))

            fv = torch.sum(visual_adapter_feature, dim = 1) / visual_adapter_feature.shape[1]
            ft = torch.sum(textual_contextual_feature, dim = 1) / textual_contextual_feature.shape[1]
            cl_cost = contrastive_loss_cal_moco_v3(fv, ft, opt.CL_tau)
        
            # select max probabilty (greedy decoding) then decode index to character
            _, vtmt_preds_index = vtmt_preds.max(2)
            vtmt_preds_str = tgt_converter.decode(vtmt_preds_index, tgt_length_for_pred, opt.tgt_level)
            
            _, mt_preds_index = mt_preds.max(2)
            mt_preds_str = tgt_converter.decode(mt_preds_index, tgt_length_for_pred, opt.tgt_level)
            src_labels = src_converter.decode(src_text_for_loss[:, 1:], src_length_for_loss, opt.src_level)
            tgt_labels = tgt_converter.decode(tgt_text_for_loss[:, 1:], tgt_length_for_loss, opt.tgt_level)
            

        infer_time += forward_time
        vtmt_weight = opt.TIMT_Weight
        cl_weight = opt.CL_Weight
    
        weighted_vtmt_cost = vtmt_weight * vtmt_cost
        weighted_cl_cost = cl_weight * cl_cost
        
        valid_loss_avg.add(weighted_vtmt_cost)
        valid_loss_avg.add(weighted_cl_cost)
        
        for gt, pred in zip(tgt_labels, vtmt_preds_str):
            if 'CTC' not in opt.Prediction:
                gt = gt[:gt.find('[s]')]
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
            ref_sents.append(gt)
            pred_sents.append(pred)
            
            # Updated accuracy calculation
            # If the characters in gt occurs in pred, a positive support is given
            for item in gt:
                n_total += 1
                if item in pred:
                    n_correct += 1

    accuracy = n_correct / float(n_total) * 100
    tok_bleu, char_bleu = doc_bleu(ref_sents, pred_sents)
    return valid_loss_avg.val(), accuracy, tok_bleu, vtmt_preds_str, mt_preds_str, src_labels, tgt_labels, infer_time, length_of_data

