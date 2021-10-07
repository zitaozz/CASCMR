#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
from torch import nn
from torch.nn import CrossEntropyLoss


class SpatialDropout1D(nn.Module):
    def __init__(self, p=0.5):
        super(SpatialDropout1D, self).__init__()
        self.p = p
        self.dropout2d = nn.Dropout2d(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, maxlen, 1, embed_size)
        x = x.permute(0, 3, 2, 1)  # (N, embed_size, 1, maxlen)
        x = self.dropout2d(x)  # (N, embed_size, 1, maxlen)
        x = x.permute(0, 3, 2, 1)  # (N, maxlen, 1, embed_size)
        x = x.squeeze(2)  # (N, maxlen, embed_size)

        return x


LSTM_UNITS = 128
CHANNEL_UNITS = 64


class BertLSTMCNNForTripletNet(BertPreTrainedModel):
    def __init__(self, config):
        super(BertLSTMCNNForTripletNet, self).__init__(config)

        filters = [3, 4, 5]

        self.bert = BertModel(config)
        self.embedding_dropout = SpatialDropout1D(config.hidden_dropout_prob)

        self.conv_layers = nn.ModuleList()
        for filter_size in filters:
            conv_block = nn.Sequential(
                nn.Conv1d(
                    512,  # config.hidden_size,
                    CHANNEL_UNITS,
                    kernel_size=filter_size,
                    padding=1,
                ),
                # nn.BatchNorm1d(CHANNEL_UNITS),
                # nn.ReLU(inplace=True),
            )
            self.conv_layers.append(conv_block)

        self.lstm = nn.LSTM(
            config.hidden_size, 256, bidirectional=True, batch_first=True
        )  # 30
        self.apply(self.init_bert_weights)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(40, 40)  # 300
        self.norm = nn.BatchNorm1d(808)  # 1068 808
        self.ly = nn.LayerNorm(808)

    def forward(
            self,
            input_ids_a,
            token_type_ids_a=None,
            attention_mask_a=None,
    ):
        bert_output_a, pooled_output_a = self.bert(
            input_ids=input_ids_a,
            token_type_ids=token_type_ids_a,
            attention_mask=attention_mask_a,
            output_all_encoded_layers=False,
        )

        h_lstm_a, (hidden_state_a, cell_state_a) = self.lstm(bert_output_a)

        h_embedding_a = h_lstm_a

        h_embedding_a = h_embedding_a.permute(0, 2, 1)
        feature_maps_a = []
        for layer in self.conv_layers:
            h_x_a = layer(h_embedding_a)
            feature_maps_a.append(
                F.max_pool1d(h_x_a, kernel_size=h_x_a.size(2)).squeeze()
            )
            feature_maps_a.append(
                F.avg_pool1d(h_x_a, kernel_size=h_x_a.size(2)).squeeze()
            )
        
        conv_features_a = torch.cat(feature_maps_a, -1)

        return conv_features_a
