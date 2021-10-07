#!/usr/bin/env python 
# -*- coding:utf-8 -*-

from models.embedding import BertLSTMCNNForTripletNet
from transformers import BertTokenizer
import torch, json
import numpy as np

print('Loading model...')
res = np.loadtxt('output/embedding.txt')
print(1)


BERT_MODEL_PATH = "output/ckpts"
BERT_VOCAB_PATH = "output/ckpts/vocab.txt"
tokenizer = BertTokenizer.from_pretrained('output/ckpts')
model = BertLSTMCNNForTripletNet.from_pretrained(BERT_MODEL_PATH)

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

print(2)

print('Finish!')
sentenceB = ''
idx = 0

while sentenceB != 'q':
    print('Please input a case.')
    sentenceB = input()
    
    if len(sentenceB) > max_seq_len:
            sentenceB = sentenceB[-max_seq_len:]
    text_dict = tokenizer.encode_plus(sentenceB, add_special_tokens=True, return_attention_mask=True)
    input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
    token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
    attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)

    resB = model(input_ids, attention_mask_a=attention_mask, token_type_ids_a=token_type_ids)
    embbedingB = resB.detach().numpy()

    max_dis = 999
    dis = []
    most_similar_sentence = ''
    for i, embedding in enumerate(res):
        dis.append(np.linalg.norm(embedding - embbedingB))

    f2 = open("output/top_{}.txt".format(idx), 'w', encoding='utf-8')

    index = sorted(range(len(dis)), key=lambda x: dis[x])
    
    top = 0
    for i in index:
        if top == 50:
             break
        #print(texts[i], dis[i], '\n')
        f2.write(texts[i])
        f2.write('\t')
        f2.write(str(dis[i]))
        f2.write('\n')

        top += 1

    f2.close()
    idx += 1     
    
    
