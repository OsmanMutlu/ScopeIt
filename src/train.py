from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from transformers import *
from torch import nn
from src.model import ScopeIt
from src.data import group_set, read_file
import numpy as np
import time
import datetime
import torch
import random
import json
import os
import sys
import unicodedata
import re

use_gpu = True
seed = 1234
max_length = 64
num_layers = 2
hidden_size = 512 # size of GRU hidden layer (in the paper they use 128)
device_ids = [0, 1]
batch_size = 300
lr = 1e-3 # 1e-4 -> in the paper
tokenizer = None

if use_gpu and torch.cuda.is_available():
    bert_device = torch.device("cuda:%d"%(device_ids[1]))
else:
    bert_device = torch.device("cpu")

if use_gpu and torch.cuda.is_available():
    model_device = torch.device("cuda:%d"%(device_ids[0]))
else:
    model_device = torch.device("cpu")

def prepare_set(text, max_length=max_length):
    """returns input_ids, attention_mask, token_type_ids for set of data ready in BERT format"""
    global tokenizer

    t = tokenizer.batch_encode_plus(text,
                        pad_to_max_length=True,
                        add_special_tokens=True,
                        max_length=max_length,
                        return_tensors='pt')

    return t["input_ids"], t["attention_mask"], t["token_type_ids"]

def predict(self, bert, x_test):
    x_test = [ prepare_set(d) for d in x_test]

    bert.eval()
    self.eval()
    with torch.no_grad():
        test_preds = []
        for batch in x_test:
            if len(batch) < 1:
                continue
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch)
            embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0].detach() # since no gradients will flow back
            embeddings = embeddings.to(model_device)
            output = model(embeddings)
            preds = torch.sigmoid(output).detach().cpu().numpy().flatten()
            test_preds += list(preds)

    return test_preds

def build_scopeit(x_train, x_dev, y_train, y_dev, pretrained_model, n_epochs=10, model_path="temp.pt"):
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    bert = AutoModel.from_pretrained(pretrained_model)

    print([len(x) for x in (y_train, y_dev)])
    dev_labels = sum(y_dev, [])
    train_labels = sum(y_train, [])

    x_train = [ prepare_set(d) for d in x_train]
    x_dev = [ prepare_set(d) for d in x_dev]
    y_train = [ torch.FloatTensor(t).unsqueeze(1) for t in y_train ]
    y_dev = [ torch.FloatTensor(t).unsqueeze(1) for t in y_dev ]

    model = ScopeIt(bert, hidden_size, num_layers=num_layers) 
    model.to(model_device)

    if torch.cuda.device_count() > 1 and bert_device.type == "cuda":
        bert = nn.DataParallel(bert, device_ids=device_ids[1:])
    bert.to(bert_device)

    # model.load_state_dict(torch.load("scopeit_" + model_path))
    # model.to(model_device)
    # model.predict = predict.__get__(model)
    # bert.load_state_dict(torch.load("bert_" + model_path))

    # return bert, model

    np.random.seed(seed)
    torch.manual_seed(seed)
    if model_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    total_steps = len(x_train) * n_epochs
    criterion = torch.nn.BCEWithLogitsLoss()
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # fine tune bert
    bert_optimizer = torch.optim.AdamW(bert.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(bert_optimizer, 
                                        num_warmup_steps = 0,
                                        num_training_steps = total_steps)
    ###

    model.zero_grad()
    best_score = -1e6
    best_loss = 1e6

    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0 
        model.train()
        for batch, labels in zip(x_train, y_train): # in this case each doc is a batch, so there is no constant batchsize
            if len(labels) < 1:
                continue
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch)
            embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0] #.detach()
            labels = labels.to(model_device)
            embeddings = embeddings.to(model_device)
            output = model(embeddings)
            loss = criterion(output, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # fine tune bert
            torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
            bert_optimizer.step()
            ####

            model_optimizer.step()
            train_loss += loss.item()
            model.zero_grad()
            bert.zero_grad()
            scheduler.step()

        train_loss /= len(train_labels)
        elapsed = time.time() - start_time
        model.eval()
        val_preds = []
        with torch.no_grad():
            val_loss = 0
            for batch, labels in zip(x_dev, y_dev):
                if len(labels) < 1:
                    continue
                b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch)
                embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0].detach() # since no gradients will flow back
                embeddings = embeddings.to(model_device)
                labels = labels.to(model_device)
                output = model(embeddings)
                loss = criterion(output, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(output).detach().cpu().numpy().flatten()
                val_preds += list(preds)
                model.zero_grad()
                bert.zero_grad()

        val_loss /= len(dev_labels)
        # val_score = f1_score(dev_labels,  [ int(x >= 0.5) for x in val_preds], average="macro")
        val_score = recall_score(dev_labels, [ int(x >= 0.5) for x in val_preds])
        print("Epoch %d Train loss: %.4f. Validation Recall: %.4f  Validation loss: %.4f. Elapsed time: %.2fs."% (epoch + 1, train_loss, val_score, val_loss, elapsed))
        if val_score > best_score:
            torch.save(model.state_dict(), "scopeit_" + model_path)
            torch.save(bert.state_dict(), "bert_" + model_path)
            print(classification_report(dev_labels, [ int(x >= 0.5) for x in val_preds], digits=4))
            best_score = val_score

    model.load_state_dict(torch.load("scopeit_" + model_path))
    model.to(model_device)
    model.predict = predict.__get__(model)
    
    bert.load_state_dict(torch.load("bert_" + model_path))
    bert.to(bert_device)
    # os.remove(model_path)

    return bert, model


if __name__ == '__main__':
    train = read_file("data/train.json")
    dev = read_file("data/dev.json")
    test = read_file("data/test.json")
    pipeline = read_file("data/pipeline.json")

    x_train, y_train = group_set(train, batch_size)
    x_dev, y_dev = group_set(dev, batch_size)
    x_test, y_test = group_set(test, batch_size)
    x_pipeline, y_pipeline = group_set(pipeline, batch_size)

    print("batch size (max sentences in doc): ", batch_size)
    bert, model = build_scopeit(x_train, x_dev, y_train, y_dev, "bert-base-uncased", n_epochs=5, model_path="bert-base-uncased.pt")

    preds = model.predict(bert, x_test)
    print("="*50, "\nTest set")
    test_labels = sum(y_test, [])
    print(classification_report(test_labels, [ int(x >= 0.5) for x in preds], digits=4))

    preds = model.predict(bert, x_pipeline)
    print("="*50, "\nPipeline")
    pipeline_labels = sum(y_pipeline, [])
    print(classification_report(pipeline_labels, [ int(x >= 0.5) for x in preds], digits=4))