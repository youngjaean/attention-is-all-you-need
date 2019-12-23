import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from attention import *
from EncoderDecoder import *
from subsequent_mask import subsequent_mask

def make_model(src_vocab, tgt_vocab, N=6, 
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)), #각 레이어를 데이터를 한 컨테이너에 넣고 한번에 돌림(embedding(num_embeddings, emdbeding_dim))
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)) #d_model과 tgt_vocab을 linear 계산하여 log_softmax로 계산
    #EncoderDecoder(size, encoer(layer(self_attn, feed_forward, sublayer(module(ModuleList), N), dropout), N),
    # decoder(layer, N),src_embed, tgt_embed, generator)
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters(): #매개 변수에 대한 iterator를 리턴
        if p.dim() > 1: #pytorch 차원 
            nn.init.xavier_uniform_(p) #균일 분포를 사용하여 입력 텐서를 값ㅇ로 채움 
    return model

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) #인수로 받은 위치에 새로운 차원을 삽입
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask