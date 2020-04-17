from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from transformers import *
from torch import nn
from .model import CNNBert
from .data import *
import torch.nn.functional as F
import unicodedata
import numpy as np
import time
import datetime
import torch
import random
import json
import os
import re
import sys


def prepare_set(text, max_length=64):
    """returns input_ids, attention_mask, token_type_ids for set of data ready in BERT format"""
    global tokenizer

    t = tokenizer.batch_encode_plus(text,
                        pad_to_max_length=True,
                        add_special_tokens=True,
                        max_length=max_length,
                        return_tensors='pt')

    return t["input_ids"], t["attention_mask"], t["token_type_ids"]

def predict(model, test_set, batch_size=32):
    test_inputs, test_masks, test_type_ids = prepare_set(test_set)
    test_data = TensorDataset(test_inputs, test_masks, test_type_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model.eval()
    with torch.no_grad(): 
        preds = []
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(device) for t in batch)
            y_pred = model(b_input_ids, b_input_mask, b_token_type_ids)
            preds += list(y_pred.cpu().numpy().flatten())

    return preds


def train_bert_cnn(x_train, x_dev, y_train, y_dev, pretrained_model, n_epochs=10, model_path="temp.pt", batch_size=32):
    bert_model = BertModel.from_pretrained(pretrained_model, output_hidden_states=True)
    
    print([len(x) for x in (y_train, y_dev)])
    y_train, y_dev = ( torch.FloatTensor(t) for t in (y_train, y_dev) )

    train_inputs, train_masks, train_type_ids = prepare_set(x_train, max_length=max_length)
    train_data = TensorDataset(train_inputs, train_masks, train_type_ids, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our dev set.
    dev_inputs, dev_masks, dev_type_ids = prepare_set(x_dev, max_length=max_length)
    dev_data = TensorDataset(dev_inputs, dev_masks, dev_type_ids, y_dev)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    model = CNNBert(768, bert_model)
    if len(device_ids) > 1 and device.type == "cuda":
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.9)
    loss_fn = nn.BCELoss()
    train_losses, val_losses = [], []
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    total_steps = len(train_dataloader) * n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = 0,
                                        num_training_steps = total_steps)

    model.zero_grad()
    best_score = 0
    best_loss = 1e6

    for epoch in range(n_epochs):

        start_time = time.time()
        train_loss = 0 
        model.train(True)

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_token_type_ids, b_labels  = tuple(t.to(device) for t in batch)
            y_pred = model(b_input_ids, b_input_mask, b_token_type_ids)
            loss = loss_fn(y_pred, b_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            scheduler.step()
            model.zero_grad()

        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        elapsed = time.time() - start_time
        model.eval()
        val_preds = []

        with torch.no_grad(): 
            val_loss = 0
            for batch in dev_dataloader:
                b_input_ids, b_input_mask, b_token_type_ids, b_labels  = tuple(t.to(device) for t in batch)
                y_pred = model(b_input_ids, b_input_mask, b_token_type_ids)
                loss = loss_fn(y_pred, b_labels.unsqueeze(1))
                val_loss += loss.item()
                y_pred = y_pred.cpu().numpy().flatten()
                val_preds += [ int(p >= 0.5) for p in y_pred ] 
                model.zero_grad()

        val_loss /= len(dev_dataloader)
        # val_score = f1_score(y_dev.cpu().numpy().tolist(), val_preds)
        val_score = recall_score(y_dev, val_preds)
        val_losses.append(val_loss)    
        print("Epoch %d Train loss: %.4f. Validation F1-Macro: %.4f  Validation loss: %.4f. Elapsed time: %.2fs."% (epoch + 1, train_losses[-1], val_score, val_losses[-1], elapsed))

        if val_score > best_score:
            torch.save(model.state_dict(), "temp.pt")
            print(classification_report(y_dev.cpu().numpy().tolist(), val_preds, digits=4))
            best_score = val_score

    model.load_state_dict(torch.load("temp.pt"))
    model.to(device)
    model.predict = predict.__get__(model)
    model.eval()
    os.remove("temp.pt")
    return model

use_gpu = True
seed = 1234
max_length = 128
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
batch_size = 32 * len(device_ids)
lr = 2e-5

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:%d"%(device_ids[0]))
else:
    device = torch.device("cpu")

if __name__ == "__main__":
    pretrained_model = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    
    train = read_file("data/corpus_sent_data/train.json")
    dev = read_file("data/corpus_sent_data/dev.json")
    test = read_file("data/corpus_sent_data/test.json")
    pipeline = read_file("data/corpus_sent_data/pipeline.json")
    
    x_train, y_train = [ x["sentence"] for x in train ], [ x["label"] for x in train ]
    x_dev, y_dev = [ x["sentence"] for x in dev ], [ x["label"] for x in dev ]
    x_test, y_test = [ x["sentence"] for x in test ], [ x["label"] for x in test ]
    x_pipeline, y_pipeline = [ x["sentence"] for x in pipeline ], [ x["label"] for x in pipeline ]
    model = train_bert_cnn(x_train, x_dev, y_train, y_dev, pretrained_model, n_epochs=6, batch_size=batch_size)

    preds = model.predict(x_test)
    print("="*50, "\nTest set")
    print(classification_report(y_test, [ int(x >= 0.5) for x in preds], digits=4))

    preds = model.predict(x_pipeline)
    print("="*50, "\nPipeline")
    print(classification_report(y_pipeline, [ int(x >= 0.5) for x in preds], digits=4))