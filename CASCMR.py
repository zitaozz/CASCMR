#!/usr/bin/python
# coding: utf-8

import os
import warnings
from timeit import default_timer as timer

import torch
from pytorch_pretrained_bert import BertAdam
from torch.utils.data import DataLoader

from callback.modelcheckpoint import ModelCheckpoint
from datasets.cail_dataset import CAILDataset
from loss.loss import TripletLoss_op
from models.net import BertLSTMCNNForTripletNet
from train.tester import Tester
from train.trainer import Trainer
from utils.logger import init_logger
from utils.utils import seed_everything, time_to_str
from transformers import AdamW, get_linear_schedule_with_warmup
warnings.filterwarnings("ignore")


ARCH = "bert"
SEED = 2323
MULTI_GPUS = [0, 1, 2, 3]
RESUME = False


LOG_DIR = "output/logs"
CHECKPOINT_DIR = "output/ckpts/"
BERT_MODEL_PATH = "models/pretrained"
BERT_VOCAB_PATH = "models/pretrained/vocab.txt"
BEST_MODEL_NAME = "{arch}_best.pth"
EPOCH_MODEL_NAME = "{arch}_{epoch}_{val_loss}.pth"

TRAIN_PATH = f"datasets/scm/train.txt"
VALID_PATH = f"datasets/scm/dev.txt"
TEST_PATH = f"datasets/scm/test.txt"


MAX_SEQ_LENGTH = 500
BATCH_SIZE = 32
NUM_EPOCHS = 6
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-5
WARMUP_PROPORTION = 0.05  # 0.05


seed_everything(SEED)
logger = init_logger(log_name=ARCH, log_dir=LOG_DIR)
os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join(map(str, MULTI_GPUS))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_dataset = CAILDataset(
    data_path=TRAIN_PATH,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_path=BERT_VOCAB_PATH,
    fts_flag=False,
    exft_a_df="",
    exft_b_df="",
    exft_c_df="",
    mode="train",
)
valid_dataset = CAILDataset(
    data_path=VALID_PATH,
    max_seq_len=MAX_SEQ_LENGTH,
    vocab_path=BERT_VOCAB_PATH,
    fts_flag=False,
    exft_a_df="",
    exft_b_df="",
    exft_c_df="",
    mode="valid",
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=True,
    drop_last=False,
    pin_memory=True,
)
valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

logger.info("---------- Bert Train ... ----------")
start_time = timer()

seed_everything(SEED)

model = BertLSTMCNNForTripletNet.from_pretrained(BERT_MODEL_PATH)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.01,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.00,  # 0
                },
            ]
num_train_optimization_steps = (
    int(len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
)

if WARMUP_PROPORTION < 1:
    num_warmup_steps = (
            num_train_optimization_steps * WARMUP_PROPORTION
    )
else:
    num_warmup_steps = WARMUP_PROPORTION

optimizer = AdamW(
    optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8
)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_train_optimization_steps,
)

model_checkpoint = ModelCheckpoint(
    checkpoint_dir=CHECKPOINT_DIR,
    mode="min",
    monitor="val_loss",
    save_best_only=False,
    best_model_name=BEST_MODEL_NAME,
    epoch_model_name=EPOCH_MODEL_NAME,
    arch=ARCH,
    logger=logger,
)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    device=device,
    scheduler=scheduler,
    n_gpus=len(MULTI_GPUS),
    criterion=TripletLoss_op(),
    fts_flag=False,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    model_checkpoint=model_checkpoint,
    logger=logger,
    resume=RESUME,
)

trainer.summary()
trainer.train()

logger.info(f'Took {time_to_str((timer() - start_time), "sec")}')

print("---------- Bert Eval ... ----------")
start_time = timer()

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

tester.eval()

print(f'Took {time_to_str((timer() - start_time), "sec")}')
