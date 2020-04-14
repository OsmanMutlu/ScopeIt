from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from transformers import *
from torch import nn
import numpy as np
import time
import datetime
import torch
import random
import json
import os
import sys
import torch.nn.functional as F
import unicodedata
import re

use_gpu = True
seed = 1234
max_length = 128
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
batch_size = 24 * len(device_ids)
lr = 2e-5
tokenizer = None

if use_gpu and torch.cuda.is_available():
    device = torch.device("cuda:%d"%(device_ids[0]))
else:
    device = torch.device("cpu")

def prepare_set(text, max_length=max_length):
    """returns input_ids, attention_mask, token_type_ids for set of data ready in BERT format"""
    global tokenizer

    t = tokenizer.batch_encode_plus(text,
                        pad_to_max_length=True,
                        add_special_tokens=True,
                        max_length=max_length,
                        return_tensors='pt')

    return t["input_ids"], t["attention_mask"], t["token_type_ids"]


def predict(self, test_set, batch_size=batch_size):
    test_inputs, test_masks, test_type_ids = prepare_set(test_set)
    test_data = TensorDataset(test_inputs, test_masks, test_type_ids)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    self.eval()
    with torch.no_grad(): 
        preds = []
        for batch in test_dataloader:
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(device) for t in batch)
            output = self(b_input_ids, 
            attention_mask=b_input_mask,
            token_type_ids=b_token_type_ids)
            logits = output[0].detach().cpu()
            preds += list(torch.nn.functional.softmax(logits, dim=1)[:, 1].numpy())

    return preds


def load_model(model_dir):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    if torch.cuda.device_count() > 1 and device.type == "cuda" and len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    model.to(device)
    model.predict = predict.__get__(model)

    return model


def save_model(model, output_dir):
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME) 
    output_config_file = os.path.join(output_dir, CONFIG_NAME) 
    
    torch.save(model_to_save.state_dict(), output_model_file) 
    model_to_save.config.to_json_file(output_config_file) 
    tokenizer.save_vocabulary(output_dir)
    return output_dir


def build_bert(x_train, x_dev, y_train, y_dev, pretrained_model, n_epochs=10, model_path="temp.pt", batch_size=batch_size):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    print([len(x) for x in (y_train, y_dev)])
    y_train, y_dev = ( torch.tensor(t) for t in (y_train, y_dev) )

    # Create the DataLoader for training set.
    train_inputs, train_masks, train_type_ids = prepare_set(x_train)
    train_data = TensorDataset(train_inputs, train_masks, train_type_ids, y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for dev set.
    dev_inputs, dev_masks, dev_type_ids = prepare_set(x_dev)
    dev_data = TensorDataset(dev_inputs, dev_masks, dev_type_ids, y_dev)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    total_steps = len(train_dataloader) * n_epochs
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = 0,
                                        num_training_steps = total_steps)

    model.zero_grad()
    best_score = 0
    best_loss = 1e6

    for epoch in range(n_epochs):

        start_time = time.time()
        train_loss = 0 
        model.train()

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_token_type_ids, b_labels = tuple(t.to(device) for t in batch)
            output = model(b_input_ids, 
                            attention_mask=b_input_mask,
                            token_type_ids=b_token_type_ids,
                            labels=b_labels)

            loss = output[0].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            model.zero_grad()

        train_loss /= len(train_dataloader)  

        scheduler.step()
        elapsed = time.time() - start_time
        model.eval()
        val_preds = []
        with torch.no_grad():
            val_loss, batch = 0, 1
            for batch in dev_dataloader:
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = tuple(t.to(device) for t in batch)
                output = model(b_input_ids, 
                            attention_mask=b_input_mask,
                            token_type_ids=b_token_type_ids,
                            labels=b_labels)
                
                loss = output[0].sum()
                val_loss += loss.item()
                logits = output[1].detach().cpu().numpy()
                val_preds += list(np.argmax(logits, axis=1).flatten())
                model.zero_grad()

        val_loss /= len(dev_dataloader)
        # val_score = f1_score(y_dev, val_preds, average="macro")
        val_score = recall_score(y_dev, val_preds)
        print("Epoch %d Train loss: %.4f. Validation Recall: %.4f  Validation loss: %.4f. Elapsed time: %.2fs."% (epoch + 1, train_loss, val_score, val_loss, elapsed))

        if val_score > best_score:
            torch.save(model.state_dict(), model_path)
            print(classification_report(y_dev, val_preds, digits=4))
            best_score = val_score

    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.predict = predict.__get__(model)
    os.remove(model_path)

    return model

if __name__ == '__main__':
    from .data import read_file
    train = read_file("data/train.json")
    dev = read_file("data/dev.json")
    test = read_file("data/test.json")
    pipeline = read_file("data/pipeline.json")
    x_train, y_train = [ x["sentence"] for x in train ], [ x["label"] for x in train ]
    x_dev, y_dev = [ x["sentence"] for x in dev ], [ x["label"] for x in dev ]
    x_test, y_test = [ x["sentence"] for x in test ], [ x["label"] for x in test ]
    x_pipeline, y_pipeline = [ x["sentence"] for x in pipeline ], [ x["label"] for x in pipeline ]

    model = build_bert(x_train, x_dev, y_train, y_dev, "bert-base-uncased", n_epochs=6, model_path="bert-base-uncased.pt")

    preds = model.predict(x_test)
    print("="*50, "\nTest set")
    print(classification_report(y_test, [ int(x >= 0.5) for x in preds], digits=4))

    preds = model.predict(x_pipeline)
    print("="*50, "\nPipeline")
    print(classification_report(y_pipeline, [ int(x >= 0.5) for x in preds], digits=4))

    
