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

from utils import AttnLabelConverter, Averager

from contrastive_loss import contrastive_loss_cal_moco_v3

from dataset import (
    Batch_Balanced_Dataset_otmi_7, hierarchical_dataset_otmi_7, AlignCollate_otmi_7
    )
from model import (
    make_std_mask, Pre_Encoder, Text_Encoder, 
    Visual_Encoder, Transformer_Encoder, Transformer_Decoder, Trasnformer_Adapter
)
from validate import validation_embedding_modal_adapter_task

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cpu':
    print('Stopped! Because Only CPU could be available ...')
    exit()
else:
    print('Now is using device: {}'.format(device))

def embedding_modal_adapter_train(opt):
    print('Load task emb_ma_train successfully.')
    print('Check concerned hyper-parameters:')
    print('whether tuning image_encoder: {}'.format(opt.image_encoder_tuning))
    print('whether tuning text_encoder: {}'.format(opt.text_encoder_tuning))
    print('whether tuning ocr_tr_encoder: {}'.format(opt.ocr_tr_encoder_tuning))
    print('whether tuning ocr_tr_decoder: {}'.format(opt.ocr_tr_decoder_tuning))
    print('whether tuning mt_tr_encoder: {}'.format(opt.mt_tr_encoder_tuning))
    print('whether tuning mt_tr_decoder: {}'.format(opt.mt_tr_decoder_tuning))    
    
    # Split dataset if there are multiple datasets are used.
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    print('-' * 80)

    print('Use Batch_Balanced_Dataset_otmi to load Train Dataset ...')
    # Load Dataset
    train_dataset = Batch_Balanced_Dataset_otmi_7(opt)
    
    print('Length of train_dataset: {}'.format(len(train_dataset)))
    print('-' * 80)

    print('Finished Loading Training Set.')

    print('-' * 80)
    print('Use hierarchical_dataset_otmi to load Valid Dataset ...')

    # Load Valid Dataset
    AlignCollate_valid = AlignCollate_otmi_7(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset_otmi_7(root=opt.valid_data, opt=opt)
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    print(valid_dataset_log)
    
    print('Finished Loading Training and validation Data!')

    """ model configuration """
    print('-' * 80)
    print('Now in model configuration')
    
    src_converter = AttnLabelConverter(opt.src_character)
    tgt_converter = AttnLabelConverter(opt.tgt_character)
    src_converter_ocr = AttnLabelConverter(opt.src_character)
    tgt_converter_mt = AttnLabelConverter(opt.tgt_character)

    opt.num_class = len(tgt_converter.character)
    opt.src_num_class = len(src_converter.character)
    opt.tgt_num_class = len(tgt_converter.character)
    opt.src_converter_ocr = len(src_converter_ocr.character)
    opt.tgt_converter_mt = len(tgt_converter_mt.character)

    if opt.rgb:
        opt.input_channel = 3
    
    # End-to-end TIMT Part    
    # Visual Encoder Definition
    FeatureExtraction_old = opt.FeatureExtraction
    if opt.SequenceModeling == "TransformerEncoder":
        visual_encoder = Visual_Encoder(opt)
    else:
        visual_encoder = Pre_Encoder(opt)
    
    # Sequential Encoder Definition
    mt_tre_encoder = Transformer_Encoder(opt)
    
    # Decoder Definition
    opt.tgt_num_class = opt.tgt_converter_mt
    tgt_decoder = Transformer_Decoder(opt, opt_dim = opt.tgt_num_class)

    # Text Encoder Definition
    Transformation_old = opt.Transformation
    src_num_class_old = opt.src_num_class
    opt.Transformation = "TPS"
    opt.FeatureExtraction = 'Textual'
    opt.src_num_class = opt.src_converter_ocr
    
    textual_encoder = Text_Encoder(opt)

    opt.Transformation = Transformation_old
    opt.FeatureExtraction = FeatureExtraction_old
    opt.src_num_class = src_num_class_old

    # Trainable Modal Adapter Definition
    adapter = Trasnformer_Adapter(opt, layer_num = opt.adapter_layer_num)

    # Parepare model_list and freeze_model_list:
    model_list = [adapter]
    model_name_list = ['adapter']
    
    freeze_model_list = []
    freeze_model_name_list = []

    if opt.image_encoder_tuning == 'True':
        model_list.append(visual_encoder)
        model_name_list.append('visual_encoder')
    else:
        freeze_model_list.append(visual_encoder)
        freeze_model_name_list.append('visual_encoder')
    
    if opt.text_encoder_tuning == 'True':
        model_list.append(textual_encoder)
        model_name_list.append('textual_encoder')
    else:
        freeze_model_list.append(textual_encoder)
        freeze_model_name_list.append('textual_encoder')
    
    if opt.mt_tr_encoder_tuning == 'True':
        model_list.append(mt_tre_encoder)
        model_name_list.append('mt_tre_encoder')
    else:
        freeze_model_list.append(mt_tre_encoder)
        freeze_model_name_list.append('mt_tre_encoder')
    
    if opt.mt_tr_decoder_tuning == 'True':
        model_list.append(tgt_decoder)
        model_name_list.append('tgt_decoder')
    else:
        freeze_model_list.append(tgt_decoder)
        freeze_model_name_list.append('tgt_decoder')
    
    freeze_params_num = []
    for sub_model in freeze_model_name_list:
        for p in filter(lambda p: p.requires_grad, eval(sub_model).parameters()):
            p.requires_grad = False
            freeze_params_num.append(np.prod(p.size()))
    freeze_params_count = sum(freeze_params_num)

    # weight initialization
    print('-' * 80)
    print('Now in weight initialization')
    print('Print all name in model.named_parameters: ')
    for sub_model in model_name_list:
        for name, param in eval(sub_model).named_parameters():
            print('=' * 50)
            print(name)
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'Transformer_encoder_layer' in name or 'Transformer_decoder_layer' in name \
                    or 'TransformerDecoder' in name or 'SequenceModeling' in name:
                    if param.dim() > 1:
                        print('init {} with xavier_uniform.'.format(name))
                        init.xavier_uniform_(param)
                        continue
            except:
                pass
            try:
                if 'bias' in name:
                    print('Constant init for {} bias'.format(name))
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    print('kaiming_normal_ init for {} weight'.format(name))
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    print('fill_(1) init for {} weight'.format(name))
                    param.data.fill_(1)
                continue

    # data parallel for multi-GPU
    visual_encoder = torch.nn.DataParallel(visual_encoder).to(device)
    textual_encoder = torch.nn.DataParallel(textual_encoder).to(device)
    mt_tre_encoder = torch.nn.DataParallel(mt_tre_encoder).to(device)
    tgt_decoder = torch.nn.DataParallel(tgt_decoder).to(device)

    adapter = torch.nn.DataParallel(adapter).to(device)

    # Load Parameters from pre-trained models
    print('Loading pre-trained model ...')
    teacher_path = opt.teacher_path
    teacher_iter = opt.teacher_iter
    visual_encoder.load_state_dict(torch.load(teacher_path + '_' + teacher_iter + '_' + 'visual_encoder_pretrained' +'.pth', map_location=device))
    textual_encoder.load_state_dict(torch.load(teacher_path + '_' + teacher_iter + '_' + 'textual_encoder_pretrained' +'.pth', map_location=device))
    mt_tre_encoder.load_state_dict(torch.load(teacher_path + '_' + teacher_iter + '_' + 'mt_encoder_pretrained' +'.pth', map_location=device))
    tgt_decoder.load_state_dict(torch.load(teacher_path + '_' + teacher_iter + '_' + 'decoder_pretrained' +'.pth', map_location=device))
    
    for sub_model in model_list:
        sub_model.train()
    for sub_model in freeze_model_list:
        sub_model.eval()

    """ setup loss """
    print('-' * 80)
    print('Now in setup loss')    
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    
    # loss averager
    loss_avg = Averager()
    vtmt_loss_avg = Averager()
    cl_loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    trainable_params_num = []
    for sub_model in model_name_list:
        for p in filter(lambda p: p.requires_grad, eval(sub_model).parameters()):
            filtered_parameters.append(p)
            trainable_params_num.append(np.prod(p.size()))
    trainable_params_count = sum(trainable_params_num)
    
    print('Freeze params num: {}'.format(freeze_params_count))
    print('Trainable params num : {}'.format(trainable_params_count))
    print('Total params num: {}'.format(freeze_params_count + trainable_params_count))

    # setup optimizer
    print('-' * 80)
    print('Now in setup optimizer')
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)

    """ start training """
    print('-' * 80)
    print('Now in start training')
    start_iter = 0

    start_time = time.time()
    best_accuracy = -1
    best_bleu = -1
    best_valid_loss = 1000000
    iteration = start_iter - 1
    previous_best_accuracy_iter = 0
    previous_best_bleu_iter = 0
    previous_best_valid_iter = 0
    
    old_time = time.time()

    while(True):
        iteration += 1
        image_tensors_1, image_tensors_2, image_tensors_3, src_labels, tgt_labels, _, _ = train_dataset.get_batch()
        
        image_1 = image_tensors_1.to(device)
        image_2 = image_tensors_2.to(device)
        image_3 = image_tensors_3.to(device)

        image = image_1
        
        # Textual data transformation: For triple img-src-tgt
        src_text, src_length = src_converter.encode(src_labels, opt.src_level,  batch_max_length=opt.src_batch_max_length)
        tgt_text, tgt_length = tgt_converter.encode(tgt_labels, opt.tgt_level,  batch_max_length=opt.tgt_batch_max_length)
        
        batch_size = image.size(0)

        src_mask = make_std_mask(src_text[:, :-1], pad = 2)[0]
        tgt_mask = make_std_mask(tgt_text[:, :-1], pad = 2)[0]
        opt.src_mask = src_mask
        opt.tgt_mask = tgt_mask

        if opt.num_gpu > 1:
            print('Now processing tgt_mask to meet multi-gpu training ...')
            x_tgt_mask, y_tgt_mask = tgt_mask.size()
            new_tgt_mask = tgt_mask.repeat(opt.batch_size, 1)
            new_tgt_mask = new_tgt_mask.reshape(opt.batch_size, x_tgt_mask, y_tgt_mask)
            tgt_mask = new_tgt_mask
        
        # ###### Triple img-src-tgt side data forward ...
        visual_feature_1 = visual_encoder(input = image_1, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_feature_2 = visual_encoder(input = image_2, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_feature_3 = visual_encoder(input = image_3, text = src_text[:, :-1], tgt_mask = tgt_mask)

        textual_feature = textual_encoder(input = image, text = src_text[:, :-1], tgt_mask = tgt_mask)

        visual_adapter_feature_1 = adapter(visual_feature_1, input = image_1, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_adapter_feature_2 = adapter(visual_feature_2, input = image_1, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_adapter_feature_3 = adapter(visual_feature_3, input = image_1, text = src_text[:, :-1], tgt_mask = tgt_mask)
        
        # We utilize mt transformer encoder to encode visual adapter features
        visual_contextual_feature_1 = mt_tre_encoder(visual_adapter_feature_1, input = image_1, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_contextual_feature_2 = mt_tre_encoder(visual_adapter_feature_2, input = image_2, text = src_text[:, :-1], tgt_mask = tgt_mask)
        visual_contextual_feature_3 = mt_tre_encoder(visual_adapter_feature_3, input = image_3, text = src_text[:, :-1], tgt_mask = tgt_mask)

        vtmt_preds_1 = tgt_decoder(contextual_feature = visual_contextual_feature_1, input = image_1, text = tgt_text[:, :-1], tgt_mask = tgt_mask)
        vtmt_preds_2 = tgt_decoder(contextual_feature = visual_contextual_feature_2, input = image_2, text = tgt_text[:, :-1], tgt_mask = tgt_mask)
        vtmt_preds_3 = tgt_decoder(contextual_feature = visual_contextual_feature_3, input = image_3, text = tgt_text[:, :-1], tgt_mask = tgt_mask)

        # Save ground truth results both for triple and parallel
        tgt_target = tgt_text[:, 1:]  # without [GO] Symbol
        
        # In Deep-text original code, using logit to calculate loss directly
        # cost calculation for triple
        vtmt_cost_1 = criterion(vtmt_preds_1.contiguous().view(-1, vtmt_preds_1.shape[-1]), tgt_target.contiguous().view(-1))
        vtmt_cost_2 = criterion(vtmt_preds_2.contiguous().view(-1, vtmt_preds_2.shape[-1]), tgt_target.contiguous().view(-1))
        vtmt_cost_3 = criterion(vtmt_preds_3.contiguous().view(-1, vtmt_preds_3.shape[-1]), tgt_target.contiguous().view(-1))
        
        print('End-to-End vtmt_cost_1 is: {}'.format(vtmt_cost_1))
        print('End-to-End vtmt_cost_2 is: {}'.format(vtmt_cost_2))
        print('End-to-End vtmt_cost_3 is: {}'.format(vtmt_cost_3))

        vtmt_cost = (vtmt_cost_1 + vtmt_cost_2 + vtmt_cost_3) / 3.0
        
        fv_1 = torch.sum(visual_adapter_feature_1, dim = 1) / visual_adapter_feature_1.shape[1]
        fv_2 = torch.sum(visual_adapter_feature_2, dim = 1) / visual_adapter_feature_2.shape[1]
        fv_3 = torch.sum(visual_adapter_feature_3, dim = 1) / visual_adapter_feature_3.shape[1]
        ft = torch.sum(textual_feature, dim = 1) / textual_feature.shape[1]
        
        cl_cost_vv_1 = contrastive_loss_cal_moco_v3(fv_1, fv_2, opt.CL_tau_vv)
        cl_cost_vv_2 = contrastive_loss_cal_moco_v3(fv_2, fv_3, opt.CL_tau_vv)
        cl_cost_vv_3 = contrastive_loss_cal_moco_v3(fv_3, fv_1, opt.CL_tau_vv)
        cl_cost_vv = (cl_cost_vv_1 + cl_cost_vv_2 + cl_cost_vv_3 ) / 3.0

        cl_cost_vl_1 = contrastive_loss_cal_moco_v3(fv_1, ft, opt.CL_tau_vl)
        cl_cost_vl_2 = contrastive_loss_cal_moco_v3(fv_2, ft, opt.CL_tau_vl)
        cl_cost_vl_3 = contrastive_loss_cal_moco_v3(fv_3, ft, opt.CL_tau_vl)
        cl_cost_vl = (cl_cost_vl_1 + cl_cost_vl_2 + cl_cost_vl_3 ) / 3.0

        cl_cost = opt.CL_vv_Weight * cl_cost_vv + opt.CL_vl_Weight * cl_cost_vl
        
        # pring corresponding contrastive loss.
        print('cl_cost_vv_1: {}'.format(cl_cost_vv_1))
        print('cl_cost_vv_2: {}'.format(cl_cost_vv_2))
        print('cl_cost_vv_3: {}'.format(cl_cost_vv_3))
        print('averaged cl_cost_vv: {}'.format(cl_cost_vv))
        
        print('cl_cost_vl_1: {}'.format(cl_cost_vl_1))
        print('cl_cost_vl_2: {}'.format(cl_cost_vl_2))
        print('cl_cost_vl_3: {}'.format(cl_cost_vl_3))
        print('averaged cl_cost_vl: {}'.format(cl_cost_vl))
        
        print('Overall cl_cost: {}'.format(cl_cost))

        vtmt_weight = opt.TIMT_Weight
        cl_weight = opt.CL_Weight

        weighted_vtmt_cost = vtmt_weight * vtmt_cost
        weighted_cl_cost = cl_weight * cl_cost
        
        cost = weighted_vtmt_cost + weighted_cl_cost
        
        print('vtmt cost: {:.3f}, cl cost: {}'.format(vtmt_cost, cl_cost))
        print('Three hyperparameter for cost addition: {:.3f}, {:.3f}'.format( vtmt_weight, cl_weight ))
        print('Cost After weighted, vtmt cost: {:.3f}, cl cost: {:.3f}'.format( weighted_vtmt_cost,  weighted_cl_cost ))

        visual_encoder.zero_grad()
        textual_encoder.zero_grad()
        mt_tre_encoder.zero_grad()
        tgt_decoder.zero_grad()

        adapter.zero_grad()

        cost.backward()
        torch.nn.utils.clip_grad_norm_(visual_encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(textual_encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(mt_tre_encoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        torch.nn.utils.clip_grad_norm_(tgt_decoder.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)

        torch.nn.utils.clip_grad_norm_(adapter.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)

        optimizer.step()

        loss_avg.add(weighted_vtmt_cost)
        loss_avg.add(weighted_cl_cost)
        
        vtmt_loss_avg.add(weighted_vtmt_cost)
        cl_loss_avg.add(weighted_cl_cost)
        
        # print loss at each step ...
        duration_time = time.time() - old_time
        
        print_str=f'step = {iteration+1}, loss = {loss_avg.val():0.5f}, vtmt_loss = {vtmt_loss_avg.val():0.5f}, cl_loss = {cl_loss_avg.val():0.5f}, duration = {duration_time:0.2f}s'
        
        old_time = time.time()
        print(print_str)
        print('-' * 100)
        
        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            print('-' * 80)
            print('Now in validation on iteration {} ...'.format(iteration + 1))
            elapsed_time = time.time() - start_time
                
            visual_encoder.eval()
            textual_encoder.eval()
            mt_tre_encoder.eval()
            adapter.eval()
            tgt_decoder.eval()

            model_list = [visual_encoder, textual_encoder, mt_tre_encoder, adapter, tgt_decoder]

            with torch.no_grad():
                valid_loss, current_accuracy, current_bleu, vtmt_preds_str, mt_preds_str, src_labels, tgt_labels, infer_time, length_of_data = validation_embedding_modal_adapter_task(
                    model_list, criterion, valid_loader, src_converter, tgt_converter, opt)
            
            for sub_model in model_list:
                sub_model.train()
            for sub_model in freeze_model_list:
                sub_model.eval()

            loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
            loss_avg.reset()

            current_model_log = f'{"Current_valid_loss":17s}: {valid_loss:0.5f}, {"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_bleu":17s}: {current_bleu:0.3f}'

            # keep best accuracy model (on valid dataset)
            if valid_loss <= best_valid_loss:
                print('Saving best_valid_loss model ...')
                best_valid_loss = valid_loss
                if opt.image_encoder_tuning == 'True':
                    torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'visual_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'visual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'visual_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'visual_encoder' + '.pth')
                if opt.text_encoder_tuning == 'True':
                    torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'textual_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'textual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'textual_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'textual_encoder' + '.pth')
                if opt.mt_tr_encoder_tuning == 'True':
                    torch.save(mt_tre_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'mt_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'mt_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'mt_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'mt_encoder' + '.pth')
                if opt.mt_tr_decoder_tuning == 'True':
                    torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'tgt_decoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'tgt_decoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'tgt_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'tgt_decoder' + '.pth')
                
                torch.save(adapter.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'adapter' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{iteration+1}_' + 'adapter' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_final_' + 'adapter' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_valid_{previous_best_valid_iter}_' + 'adapter' + '.pth')

                previous_best_valid_iter = iteration + 1
            
            # keep best accuracy model (on valid dataset)  
            if current_accuracy >= best_accuracy:
                print('Saving best_accuracy model ...')
                best_accuracy = current_accuracy
                if opt.image_encoder_tuning == 'True':
                    torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'visual_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'visual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'visual_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'visual_encoder' + '.pth')
                if opt.text_encoder_tuning == 'True':
                    torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'textual_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'textual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'textual_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'textual_encoder' + '.pth')
                if opt.mt_tr_encoder_tuning == 'True':
                    torch.save(mt_tre_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'mt_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'mt_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'mt_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'mt_encoder' + '.pth')
                if opt.mt_tr_decoder_tuning == 'True':
                    torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'tgt_decoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'tgt_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'tgt_decoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'tgt_decoder' + '.pth')
                
                torch.save(adapter.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'adapter' + '.pth') 
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{iteration+1}_' + 'adapter' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_final_' + 'adapter' + '.pth')   
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_accuracy_{previous_best_accuracy_iter}_' + 'adapter' + '.pth')
                
                previous_best_accuracy_iter = iteration + 1
            
            # keep best bleu model (on valid dataset)  
            print('Current bleu: {}'.format(current_bleu))
            print('Current best_bleu: {}'.format(best_bleu))
            if current_bleu >= best_bleu:
                print('Saving best_bleu model ...')
                best_bleu = current_bleu

                if opt.image_encoder_tuning == 'True':
                    torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'visual_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'visual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'visual_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'visual_encoder' + '.pth')
                if opt.text_encoder_tuning == 'True':
                    torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'textual_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'textual_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'textual_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'textual_encoder' + '.pth')
                if opt.mt_tr_encoder_tuning == 'True':
                    torch.save(mt_tre_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'mt_encoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'mt_encoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'mt_encoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'mt_encoder' + '.pth')
                if opt.mt_tr_decoder_tuning == 'True':
                    torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'tgt_decoder' + '.pth')
                    os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'tgt_decoder' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'tgt_decoder' + '.pth')
                    os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'tgt_decoder' + '.pth')

                torch.save(adapter.state_dict(), f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'adapter' + '.pth')
                os.system('cp -r ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{iteration+1}_' + 'adapter' + '.pth ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_final_' + 'adapter' + '.pth')
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/best_bleu_{previous_best_bleu_iter}_' + 'adapter' + '.pth')
                
                previous_best_bleu_iter = iteration + 1
            
            best_model_log = f'{"Best_valid":17s}: {best_valid_loss:0.5f}, {"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_blue":17s}: {best_bleu:0.2f}'

            loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
            print(loss_model_log)

            # show part of predicted results
            dashed_line = '-' * 80
            print('Part of VTMT predicted and ground-truth results :')
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Match-up T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred in zip(tgt_labels[:5], vtmt_preds_str[:5]):
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)

            dashed_line = '-' * 80
            print('Part of MT predicted and ground-truth results :')
            head = f'{"Ground Truth":25s} | {"Prediction":25s} | Match-up T/F'
            predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
            for gt, pred in zip(tgt_labels[:5], mt_preds_str[:5]):
                gt = gt[:gt.find('[s]')]
                pred = pred[:pred.find('[s]')]

                predicted_result_log += f'{gt:25s} | {pred:25s} | {str(pred == gt)}\n'
            predicted_result_log += f'{dashed_line}'
            print(predicted_result_log)

        ######################################################################

        # save model per opt.saveInterval, deep-text originally 1e+5 iter
        if (iteration + 1) % opt.saveInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            print('-' * 80)
            print('Saving model on set step of {} ...'.format(iteration + 1))

            if opt.image_encoder_tuning == 'True':
                torch.save(visual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'visual_encoder' + '.pth')
            if opt.text_encoder_tuning == 'True':
                torch.save(textual_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'textual_encoder' + '.pth')
            if opt.mt_tr_encoder_tuning == 'True':
                torch.save(mt_tre_encoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'mt_encoder' + '.pth')
            if opt.mt_tr_decoder_tuning == 'True':
                torch.save(tgt_decoder.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'tgt_decoder' + '.pth')

            torch.save(adapter.state_dict(), f'{opt.saved_model}/{opt.exp_name}/iter_step_{iteration+1}_' + 'adapter' + '.pth')
        
        # Final Step and offer information
        if (iteration + 1) == opt.num_iter:
            print('end the training at step {}!'.format(iteration + 1))

            print('Remove iter_step_1_* model savings, which is just a model saving to see whether it could run normally.')

            if opt.image_encoder_tuning == 'True':
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'visual_encoder' + '.pth')
            if opt.text_encoder_tuning == 'True':
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'textual_encoder' + '.pth')
            if opt.mt_tr_encoder_tuning == 'True':
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'mt_encoder' + '.pth')
            if opt.mt_tr_decoder_tuning == 'True':
                os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'tgt_decoder' + '.pth')

            os.system(f'rm -f ' + f'{opt.saved_model}/{opt.exp_name}/iter_step_1_' + 'adapter' + '.pth')
            
            sys.exit()
