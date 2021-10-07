#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from models.embedding import BertCNNForTripletNet
from transformers import BertTokenizer
import torch, json
import numpy as np

BERT_MODEL_PATH = "output/ckpts"
BERT_VOCAB_PATH = "output/ckpts/vocab.txt"
tokenizer = BertTokenizer.from_pretrained('output/ckpts')
model = BertCNNForTripletNet.from_pretrained(BERT_MODEL_PATH)

max_seq_len = 500
texts = []
f1 = open('datasets/all/train.txt', 'r', encoding='utf-8')
for line in f1:
    x  = json.loads(line)
    if x['A'] not in texts:
         texts.append(x['A'])
    if x['B'] not in texts:       
         texts.append(x['B'])
    if x['C'] not in texts:
         texts.append(x['C'])
    
a = []
for i, sentenceA in enumerate(texts):
    if(i%100==0):
        print(i)
    if len(sentenceA) > max_seq_len:
            sentenceA = sentenceA[-max_seq_len:]
    text_dict = tokenizer.encode_plus(sentenceA, add_special_tokens=True, return_attention_mask=True)
    input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
    token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
    attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)

    res = model(input_ids, attention_mask_a=attention_mask, token_type_ids_a=token_type_ids)
    a.append(res.detach().numpy())
             
f2 = open('output/embedding.txt', 'w', encoding='utf-8')
np.savetxt(f2, a)