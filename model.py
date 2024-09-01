import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import math

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import (
    BidirectionalLSTM, PositionalEncoding, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
)
from modules.prediction import Attention

import logging
logging.basicConfig(level = logging.INFO, format = '%(message)s')
logger = logging.getLogger(__name__)
print = logger.info

class Pre_Encoder(nn.Module):
    def __init__(self, opt):
        super(Pre_Encoder, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'Textual':
            self.FeatureExtraction = Embeddings(opt.hidden_size, opt.src_num_class)
            self.make_feature_dim = nn.Linear(opt.src_batch_max_length+1, 26)
            return
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        
        self.cv_bi_lstm = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.cv_bi_lstm_output = opt.hidden_size    # Used to initialize later layer

    def forward(self, input, text, tgt_mask, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        if self.stages['Feat'] == 'Textual':
            visual_feature = self.FeatureExtraction(text)
            visual_feature = visual_feature.permute(0, 2, 1)
            visual_feature = self.make_feature_dim(visual_feature).permute(0, 2, 1)
        else:
            visual_feature = self.FeatureExtraction(input)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]            
            visual_feature = visual_feature.squeeze(3)
            visual_feature = self.cv_bi_lstm(visual_feature)

        return visual_feature

class Text_Encoder(nn.Module):
    def __init__(self, opt):
        super(Text_Encoder, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}
        
        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        self.FeatureExtraction = Embeddings(opt.hidden_size, opt.src_num_class)
        self.make_feature_dim = nn.Linear(opt.src_batch_max_length+1, 26)

    def forward(self, input, text, tgt_mask, is_train=True):
        """ Transformation stage """        
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)
        """ Feature extraction stage """
        feature = self.FeatureExtraction(text)
        return feature

class Visual_Encoder(nn.Module):
    def __init__(self, opt):
        super(Visual_Encoder, self).__init__()
        self.opt = opt
        """ 1. Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')
            self.Transformation = None

        """ 2. FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.make_feature_dim = nn.Linear(int(opt.imgW/4+1), opt.src_batch_max_length+1)   # make the sequential length as source language
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        self.Res_LSTM = True
        self.cv_bi_lstm = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.cv_bi_lstm_output = opt.hidden_size    # Used to initialize later layer
        
    def forward(self, input, text, tgt_mask, is_train=True):
        """ 1. Transformation stage """
        if not self.Transformation is None:
            input = self.Transformation(input)

        """ 2. Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        
        visual_feature = visual_feature.squeeze(3)
        assert len(visual_feature.size()) == 3
        
        if not int(self.opt.imgW/4+1) == self.opt.src_batch_max_length+1:
            print('Now in make dim section.')
            visual_feature = self.make_feature_dim(visual_feature.permute(0, 2, 1)).permute(0, 2, 1)
        
        visual_feature = self.cv_bi_lstm(visual_feature)

        return visual_feature

class Transformer_Encoder(nn.Module):
    def __init__(self, opt, opt_from = None):
        super(Transformer_Encoder, self).__init__()
        self.opt = opt

        """ 1. Sequence modeling"""
        self.SequenceModeling_input = opt.hidden_size
        self.SequenceModeling_output = opt.hidden_size

        if opt_from == 'ocr_teacher':
            self.EncoderPositionalEmbedding = PositionalEncoding(d_model=self.SequenceModeling_output, dropout = 0, max_len = opt.src_batch_max_length + 2)
        else:
            self.EncoderPositionalEmbedding = PositionalEncoding(d_model=self.SequenceModeling_output, dropout = 0, max_len = max(opt.src_batch_max_length, opt.tgt_batch_max_length) + 2)
        
        self.Transformer_encoder_layer = TransformerEncoderLayer(d_model=self.SequenceModeling_input, nhead=8)
        self.SequenceModeling = TransformerEncoder(self.Transformer_encoder_layer, num_layers=6)

    def forward(self, visual_feature, input, text, tgt_mask, is_train=True):

        """ 1. Sequence modeling stage """
        visual_feature = self.EncoderPositionalEmbedding(visual_feature)

        batch_mid_variable = visual_feature.permute(1, 0, 2)    # Make batch dimension in the middle

        contextual_feature = self.SequenceModeling(src=batch_mid_variable)

        contextual_feature = contextual_feature.permute(1, 0, 2)

        return contextual_feature

class Transformer_Decoder(nn.Module):
    def __init__(self, opt, opt_dim=None, opt_from = None):
        super(Transformer_Decoder, self).__init__()
        self.opt = opt

        """ 1. Prediction """
        self.Prediction_input = opt.hidden_size
        self.Prediction_output = opt.hidden_size
        if opt_dim is None:
            opt_dim = opt.num_class
        self.tgt_embedding = Embeddings(opt.hidden_size, opt_dim)
        
        self.DecoderPositionalEmbedding = PositionalEncoding(d_model=self.Prediction_output, dropout = 0, max_len = max(opt.src_batch_max_length, opt.tgt_batch_max_length) + 2)
        
        self.Transformer_decoder_layer = TransformerDecoderLayer(d_model=opt.hidden_size, nhead=8)
        self.Prediction_TransformerDecoder = TransformerDecoder(self.Transformer_decoder_layer, num_layers=6, opt = opt, output_dim=opt_dim)

    def forward(self, contextual_feature, input, text, tgt_mask, is_train=True):

        """ 1. Prediction stage """
        pred_feature = contextual_feature

        pred_feature = pred_feature.permute(1, 0, 2)    # Make batch dimension in the middle

        text_input = self.tgt_embedding(text)
        text_input = self.DecoderPositionalEmbedding(text_input)
        text_input = text_input.permute(1, 0, 2)

        pred_feature = self.Prediction_TransformerDecoder(tgt = text_input, memory = pred_feature, tgt_mask = tgt_mask, is_train = is_train)

        pred_feature = pred_feature.permute(1, 0, 2)    # Make batch dimension in the top

        prediction = pred_feature
        
        return prediction

class Trasnformer_Adapter(nn.Module):
    def __init__(self, opt, layer_num = 6, opt_from = None):
        super(Trasnformer_Adapter, self).__init__()
        self.opt = opt

        """ 1. Sequence modeling"""
        self.SequenceModeling_input = opt.hidden_size
        self.SequenceModeling_output = opt.hidden_size
        
        self.EncoderPositionalEmbedding = PositionalEncoding(d_model=self.SequenceModeling_output, dropout = 0, max_len = max(opt.src_batch_max_length, opt.tgt_batch_max_length) + 2)
          
        self.Transformer_encoder_layer = TransformerEncoderLayer(d_model=self.SequenceModeling_input, nhead=8)
        self.SequenceModeling = TransformerEncoder(self.Transformer_encoder_layer, num_layers=layer_num)

    def forward(self, visual_feature, input, text, tgt_mask, is_train=True):
        """ 1. Sequence modeling stage """
        visual_feature = self.EncoderPositionalEmbedding(visual_feature)

        batch_mid_variable = visual_feature.permute(1, 0, 2)    # Make batch dimension in the middle

        contextual_feature = self.SequenceModeling(src=batch_mid_variable)

        contextual_feature = contextual_feature.permute(1, 0, 2)

        return contextual_feature

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = subsequent_mask(tgt.size(-1))
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        return Variable(tgt_mask.cuda(), requires_grad=False)

def dissym_subsequent_mask(size1, size2):
    "Mask out subsequent positions."
    attn_shape = (1, size1, size2)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_dissym_mask(size1, size2, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = dissym_subsequent_mask(size1, size2)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
        return Variable(tgt_mask.cuda(), requires_grad=False)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

