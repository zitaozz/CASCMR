#!/usr/bin/python
# coding: utf-8

import os
import warnings
from timeit import default_timer as timer

import torch
from torch.utils.data import DataLoader

from datasets.cail_dataset import CAILDataset
from loss.loss import TripletLoss_op
from models.net import BertLSTMCNNForTripletNet
from train.tester import Tester
from utils.logger import init_logger
from utils.utils import seed_everything, time_to_str, write2file

MULTI_GPUS = [0, 1, 2, 3]
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, MULTI_GPUS))

ARCH = "bert"
SEED = 2323

TEST_PATH = f"datasets/scm/test.txt"  #
OUTPUT_PATH = "output/output.txt"  #
LOG_DIR = "output/logs"

MAX_SEQ_LENGTH = 500
BATCH_SIZE = 32

seed_everything(SEED)
logger = init_logger(log_name=ARCH, log_dir=LOG_DIR)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("---------- Bert Eval ... ----------")
start_time = timer()

# bert_config.json, pytorch_model.bin vocab.txt in chpts
BERT_MODEL_PATH = "output/ckpts"
BERT_VOCAB_PATH = "models/pretrained/vocab.txt"

test_dataset = CAILDataset(
    data_path=TEST_PATH,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_path=BERT_VOCAB_PATH,
    fts_flag=False,
    exft_a_df="",
    exft_b_df="",
    exft_c_df="",
    mode="test",
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

model = BertLSTMCNNForTripletNet.from_pretrained(BERT_MODEL_PATH)
model = model.to(device)

tester = Tester(
    model=model,
    test_loader=test_loader,
    device=device,
    criterion=TripletLoss_op(),
    fts_flag=False,
    logger=logger,
)

# tester.eval()
test_probs_model = tester.test()

test_probs = []
correct = 0
for i in range(len(test_probs_model)):
    probs = (
        test_probs_model[i]
    )
    test_probs.append(probs)
    if probs <= 0:
        correct += 1

print(correct)
print(correct / len(test_probs))

write2file(OUTPUT_PATH, test_probs)

print(f'Took {time_to_str((timer() - start_time), "sec")}')
