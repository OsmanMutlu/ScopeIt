from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, adjusted_rand_score
from transformers import *
from torch import nn
from model import ScopeIt, CorefHead
from data import group_set, read_file
import numpy as np
import time
import torch
import random
from conlleval import evaluate2
from tqdm import tqdm
import json
from itertools import combinations
# import ipdb

use_gpu = True
seed = 1234
max_length = 128 # max length of a sentence
fine_tune_bert = True # set False to use bert only as embedder
num_layers = 2  # GRU num of layer
hidden_size = 512 # size of GRU hidden layer (in the paper they use 128)
batch_size = 200 # max sentence number in documents
lr = 1e-4 # 1e-4 -> in the paper
tokenizer = None
is_pretokenized = True # True when training with token level data
num_token_labels = 15
repo_path = "/home/omutlu/ScopeIt"
num_epochs = 30
only_test = True
predict = False

# For model path name
generate_labels_for_neg_docs = False
number_of_mlps = 2
aftergru = 1
do_rescoring = False # Doesn't matter in training

frozen_model_path = "128_2_512_200_0.0001_False.pt"

# TODO : Maybe don't take loss from coref gold matrix's upper part, since it is symmetrical to the lower side. If we do this, we need to do a similar thing in evaluation
# TODO : Freeze already trained "coref-less" multi-task model, than train coref head with token level train data.

# Parameters of the clustering algorithm,
threshold = 0.5
reward = penalty = 0.1
clustering_threshold = 0.5

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

criterion = torch.nn.BCEWithLogitsLoss()
token_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

label_list = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger", "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O"]
idtolabel = {}
for i,lab in enumerate(label_list):
    idtolabel[i] = lab

if use_gpu and torch.cuda.is_available():
    bert_device = torch.device("cuda:%d"%(device_ids[2]))
else:
    bert_device = torch.device("cpu")

if use_gpu and torch.cuda.is_available():
    model_device = torch.device("cuda:%d"%(device_ids[0]))
else:
    model_device = torch.device("cpu")

if use_gpu and torch.cuda.is_available():
    coref_head_device = torch.device("cuda:%d"%(device_ids[1]))
else:
    coref_head_device = torch.device("cpu")

def prepare_set(texts, max_length=max_length):
    """returns input_ids, attention_mask, token_type_ids for set of data ready in BERT format"""
    global tokenizer

    t = tokenizer.batch_encode_plus(texts,
                        is_pretokenized=is_pretokenized,
                        pad_to_max_length=True,
                        add_special_tokens=True,
                        max_length=max_length,
                        return_tensors='pt')

    return t["input_ids"], t["attention_mask"], t["token_type_ids"]

def cluster_to_relations(cluster, pos_idxs):
    ids = list(combinations(sorted(pos_idxs), 2))
    pairs = {}
    for x in ids:
        pairs[x] = 0

    for c in cluster:
        curr_ids = list(combinations(sorted(c), 2))
        for x in curr_ids:
            pairs[x] = 1

    return list(pairs.values())

def to_cluster_gold(clusters):
    gold = []
    seen = set()
    dupes = set(x for v in clusters.values() for x in v if x in seen or seen.add(x))
    for v in clusters.values():
        v = [x-1 for x in v if x not in dupes] # -1 because these indexes start from 1
        if v:
            gold.append(v)

    if len(gold) == 1 and len(gold[0]) == 1: # nothing to evaluate since we have only one option for clustering
        return []

    return gold

def clustering_algo(coref_out, pos_idxs, rescoring=False):
    ids = list(combinations(pos_idxs, 2))
    pos_id_to_order = {}
    for i, idx in enumerate(pos_idxs):
        pos_id_to_order[idx] = i

    pairs = {}
    for v in ids:
        pairs[v] = coref_out[pos_id_to_order[v[0]], pos_id_to_order[v[1]]]

    coref_out = (coref_out >= 0.5).astype(int)

    # rescoring
    if rescoring:
        for s1, s2 in ids:
            for s in pos_idxs:
                if s1 == s or s2 == s:
                    continue

                # if s > s1 then symmetry is lost
                s1_s_label = coref_out[pos_id_to_order[s1],pos_id_to_order[s]]
                s2_s_label = coref_out[pos_id_to_order[s2],pos_id_to_order[s]]
                if  s1_s_label == 1 and s2_s_label == 1:
                    pairs[(s1,s2)] += reward
                elif s1_s_label != s2_s_label:
                    pairs[(s1,s2)] -= penalty

    clusters = { n: 0 for n in pos_idxs }
    filtered_pairs = { k : v  for k, v in pairs.items() if v >= clustering_threshold }
    sorted_pairs = sorted(filtered_pairs, key=lambda x: (filtered_pairs[x], x[0] - x[1]), reverse=True)

    # clustering
    group_no = 0
    for s1, s2 in sorted_pairs:
        if clusters[s1] == clusters[s2] == 0:
            group_no += 1
            clusters[s1] = clusters[s2] = group_no
        elif clusters[s1] == 0:
            clusters[s1] = clusters[s2]
        else:
            clusters[s2] = clusters[s1]

    for s in pos_idxs:
        if clusters[s] == 0:
            group_no += 1
            clusters[s] = group_no

    cluster_grouped = {}
    for k, v in clusters.items():
        if v in cluster_grouped:
            cluster_grouped[v].append(k)
        else:
            cluster_grouped[v] = [k, ]

    return list(cluster_grouped.values())


def prepare_labels(json_data, max_length=max_length, only_token=False, truncate=False):

    sent_labels = json_data["sent_labels"]
    if truncate:
        sent_labels = sent_labels[:batch_size]

    sent_labels = np.array(sent_labels)

    coref_gold = torch.FloatTensor(json_data["coref_gold"]) # Pos_sentsxPos_sents
    if truncate and len(coref_gold) != 0 and len(sent_labels) > batch_size:
        pos_idxs_over_limit = [i for i,v in enumerate(sent_labels.tolist()) if v == 1 and i >= batch_size]
        if pos_idxs_over_limit: # since we will be discarding these sentences from input, we remove them from here as well
            coref_gold = coref_gold[:-len(pos_idxs_over_limit),:-len(pos_idxs_over_limit)]

    return sent_labels, coref_gold

def convert_to_sklearn_format(clusters):
    sentences = sorted(sum(clusters, []))
    labels = list(sentences)
    assert len(set(labels)) == len(labels)

    for i, cl in enumerate(clusters):
        for e in cl:
            labels[sentences.index(e)] = i

    return labels

def test_model(bert, model, coref_head, x_test, cluster_golds=[]):
    sent_sizes = []
    coref_scores = []
    all_gold_relations = []
    all_pred_relations = []

    with torch.no_grad():
        for idx, batch in enumerate(x_test):
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch)
            embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0].detach()
            embeddings = embeddings.to(model_device)

            token_out, sent_out, doc_out, sent_embeddings = model(embeddings)
            sent_embeddings = sent_embeddings.detach()
            sent_embeddings = sent_embeddings.to(coref_head_device)
            coref_out = coref_head(sent_embeddings)

            gold_cluster = cluster_golds[idx]
            if gold_cluster:
                pos_idxs = sorted([x for c in gold_cluster for x in c])
                sent_sizes.append(len(pos_idxs))

                coref_out = torch.sigmoid(coref_out).detach().cpu().numpy()
                coref_out = coref_out[pos_idxs,:][:,pos_idxs]

                pred_cluster = clustering_algo(coref_out, pos_idxs, rescoring=do_rescoring)
                coref_score = adjusted_rand_score(convert_to_sklearn_format(gold_cluster), convert_to_sklearn_format(pred_cluster))
                coref_scores.append(coref_score)

                # Just for f1 macro, may remove later
                gold_relations = cluster_to_relations(gold_cluster, pos_idxs)
                all_gold_relations.extend(gold_relations)
                pred_relations = cluster_to_relations(pred_cluster, pos_idxs)
                all_pred_relations.extend(pred_relations)

    coref_f1_macro = f1_score(all_gold_relations, all_pred_relations, average="macro")
    macro_rand_score = sum(coref_scores) / len(coref_scores)
    micro_rand_score = sum(s * c for s, c in zip(coref_scores, sent_sizes)) / sum(sent_sizes)

    return micro_rand_score, macro_rand_score, coref_f1_macro

def build_scopeit(train_data, dev_data, pretrained_model, n_epochs=10, model_path="temp.pt"):
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    bert = AutoModel.from_pretrained(pretrained_model)

    x_train = []
    y_train = []
    for d in train_data:
        if len(d["tokens"]) > batch_size:
            x_train.append(prepare_set(d["tokens"][:batch_size]))
            y_train.append(prepare_labels(d, truncate=True))
        else:
            x_train.append(prepare_set(d["tokens"]))
            y_train.append(prepare_labels(d, truncate=False))

    x_dev = []
    dev_cluster_golds = []
    for d in dev_data:
        x_dev.append(prepare_set(d["tokens"]))
        clusters = d["event_clusters"]
        cluster_gold = []
        if clusters:
            cluster_gold = to_cluster_gold(clusters)

        dev_cluster_golds.append(cluster_gold)


    model = ScopeIt(bert, hidden_size, num_layers=num_layers, num_token_labels=num_token_labels)
    model.load_state_dict(torch.load(repo_path + "/models/scopeit_" + frozen_model_path))
    model.to(model_device)

    if torch.cuda.device_count() > 1 and bert_device.type == "cuda":
        bert = nn.DataParallel(bert, device_ids=device_ids[2:])

    bert.load_state_dict(torch.load(repo_path + "/models/bert_" + frozen_model_path))
    bert.to(bert_device)

    coref_head = CorefHead(hidden_size * aftergru)
    coref_head.to(coref_head_device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if model_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    total_steps = len(x_train) * n_epochs
    coref_head_optimizer = torch.optim.AdamW(coref_head.parameters(), lr=lr)

    best_score = -1e6
    best_loss = 1e6

    model.eval()
    bert.eval()
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = 0
        coref_head.train()

        # shuffle training data
        train_data = list(zip(x_train, y_train))
        random.shuffle(train_data)
        x_train, y_train = zip(*train_data)
        ##

        print("Starting Epoch %d"%(epoch+1))
        for step, (batch, labels) in enumerate(tqdm(zip(x_train, y_train), desc="Iteration")): # each doc is a batch
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch)
            
            with torch.no_grad():
                embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0].detach()
                embeddings = embeddings.to(model_device)
                token_out, sent_out, doc_out, sent_embeddings = model(embeddings)
                sent_embeddings = sent_embeddings.detach()

            sent_embeddings = sent_embeddings.to(coref_head_device)
            coref_out = coref_head(sent_embeddings)

            sent_labels, coref_gold = labels
            coref_gold = coref_gold.to(coref_head_device)

            # WARNING : All training samples must have coref_gold. TODO : Choose data accordingly
            pos_indexes = torch.arange(len(sent_labels))[sent_labels == 1]
            fixed_coref_out = coref_out[pos_indexes,:][:,pos_indexes] # Pos_sentsxPos_sents
            fixed_coref_out = fixed_coref_out[coref_gold != -1]
            coref_gold = coref_gold[coref_gold != -1]
            loss = criterion(fixed_coref_out, coref_gold)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(coref_head.parameters(), 1.0)

            coref_head_optimizer.step()
            train_loss += loss.item()
            coref_head.zero_grad()

        train_loss /= len(x_train)
        elapsed = time.time() - start_time

        # Validation
        coref_head.eval()
        coref_micro_score, coref_macro_score, coref_f1_macro = test_model(bert, model, coref_head, x_dev, cluster_golds=dev_cluster_golds)
        print("Epoch %d - Train loss: %.4f. Adjusted Rand; Micro Score: %.6f, Macro Score: %.6f. Pairwise Relation F1 Macro : %.6f. Elapsed time: %.2fs."% (epoch + 1, train_loss, coref_micro_score, coref_macro_score, coref_f1_macro, elapsed))
        # val_score = (coref_micro_score + coref_f1_macro) / 2
        val_score = coref_micro_score
        if val_score > best_score:
            print("Saving model!")
            torch.save(coref_head.state_dict(), repo_path + "/models/coref_head_" + model_path)
            best_score = val_score

        print("========================================================================")

    return bert, model, coref_head

if __name__ == '__main__':
    train = read_file(repo_path + "/data/fixed_frozen_coref_train_data.json")
    # train = read_file(repo_path + "/data/fixed_train_data.json")
    dev = read_file(repo_path + "/data/fixed_dev_data.json")
    # Test file must contain one more column than others, the "old_token_labels" referring to original token_labels before fix_token_labels.py is applied. This is needed in length_fix for testing.
    # test = read_file(repo_path + "/data/fixed_test_data.json")
    test = read_file(repo_path + "/data/fixed_pipeline_review_data_only_pos.json")
    # test = read_file(repo_path + "/data/fixed_pipeline_review_data.json")

    # TODO : print parameters here
    print("max batch size (max sentences in doc): ", batch_size)

    model_path = str(max_length) + "_" + str(num_layers) + "_" + str(hidden_size) + "_" + str(batch_size) + "_" + str(lr) + "_" + str(generate_labels_for_neg_docs) + "_coref_" + str(number_of_mlps) + "mlp_aftergru" + str(aftergru) + ".pt"
    if not only_test:
        bert, model, coref_head = build_scopeit(train, dev, "bert-base-uncased", n_epochs=num_epochs, model_path=model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert = AutoModel.from_pretrained("bert-base-uncased")
        model = ScopeIt(bert, hidden_size, num_layers=num_layers, num_token_labels=num_token_labels)
        if torch.cuda.device_count() > 1 and bert_device.type == "cuda":
            bert = nn.DataParallel(bert, device_ids=device_ids[2:])

        coref_head = CorefHead(hidden_size * aftergru)

    model.load_state_dict(torch.load(repo_path + "/models/scopeit_" + frozen_model_path))
    model.to(model_device)
    bert.load_state_dict(torch.load(repo_path + "/models/bert_" + frozen_model_path))
    bert.to(bert_device)
    coref_head.load_state_dict(torch.load(repo_path + "/models/coref_head_" + model_path))
    coref_head.to(coref_head_device)
    bert.eval()
    model.eval()
    coref_head.eval()

    x_test = [prepare_set(d["tokens"]) for d in test]
    test_cluster_golds = []
    for t in test:
        clusters = t["event_clusters"]
        cluster_gold = []
        if clusters:
            cluster_gold = to_cluster_gold(clusters)

        test_cluster_golds.append(cluster_gold)

    coref_micro_score, coref_macro_score, coref_f1_macro = test_model(bert, model, coref_head, x_test, cluster_golds=test_cluster_golds)
    print("Adjusted Rand; Micro Score: %.6f, Macro Score: %.6f. Pairwise Relation F1 Macro : %.6f."% (coref_micro_score, coref_macro_score, coref_f1_macro))
