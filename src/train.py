from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, adjusted_rand_score
from transformers import *
from torch import nn
from model import ScopeIt
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
num_epochs = 15
only_test = False
predict = False

# For model path name
generate_labels_for_neg_docs = False
number_of_mlps = 2
aftergru = 1

# TODO : Maybe don't take loss from coref gold matrix's upper part, since it is symmetrical to the lower side. If we do this, we need to do a similar thing in evaluation
# TODO : Freeze already trained "coref-less" multi-task model, than train coref head with token level train data.

# Parameters of the clustering algorithm,
threshold = 0.5
reward = penalty = 0.1
clustering_threshold = 0.5

device_ids = [0, 1, 2, 3, 4, 5, 6, 7] if fine_tune_bert else [0, 1, 2, 3, 4]
# device_ids = [4, 5, 6, 7] if fine_tune_bert else [0, 1, 2, 3, 4]

criterion = torch.nn.BCEWithLogitsLoss()
token_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

label_list = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger", "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O"]
idtolabel = {}
for i,lab in enumerate(label_list):
    idtolabel[i] = lab

if use_gpu and torch.cuda.is_available():
    bert_device = torch.device("cuda:%d"%(device_ids[1]))
else:
    bert_device = torch.device("cpu")

if use_gpu and torch.cuda.is_available():
    model_device = torch.device("cuda:%d"%(device_ids[0]))
else:
    model_device = torch.device("cpu")

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

def clustering_algo(coref_out, pos_idxs):
    ids = list(combinations(pos_idxs, 2))
    pos_id_to_order = {}
    for i, idx in enumerate(pos_idxs):
        pos_id_to_order[idx] = i

    pairs = {}
    for v in ids:
        pairs[v] = coref_out[pos_id_to_order[v[0]], pos_id_to_order[v[1]]]

    coref_out = (coref_out >= 0.5).astype(int)

    # TODO : Preliminary tests show that 3 out of 4 models do better without rescoring when testing. Maybe remove.
    # rescoring
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


def prepare_labels(json_data, max_length=max_length, only_token=False, truncate=False, coref_gold=False):

    # No problem occurs if sent_labels or token_labels are empty in case they are not available when training.
    # When in validation sent_labels and token_labels must not be empty !!!
    token_labels = json_data["token_labels"]

    if truncate:
        token_labels = token_labels[:batch_size]

    # Token labels
    for i in range(len(token_labels)):
        labs = token_labels[i]
        if len(labs) > max_length - 2:
            labs = labs[:(max_length-2)]
        else:
            labs = labs + [-1] * (max_length - 2 - len(labs)) # for the [PAD] tokens

        labs = [-1] + labs + [-1] # For "[CLS]" at the beginning and "[SEP]" at the end
        token_labels[i] = labs

    token_labels = torch.LongTensor(token_labels) # Since this will be used in categorical loss, we use LongTensor instead of FloatTensor

    if only_token:
        return token_labels

    sent_labels = json_data["sent_labels"]
    if truncate:
        sent_labels = sent_labels[:batch_size]

    doc_label = torch.FloatTensor([json_data["doc_label"]])
    sent_labels = torch.FloatTensor(sent_labels)

    if coref_gold:
        coref = torch.FloatTensor(json_data["coref_gold"]) # Pos_sentsxPos_sents
        if truncate and len(coref) != 0 and len(sent_labels) > batch_size:
            pos_idxs_over_limit = [i for i,v in enumerate(sent_labels) if v == 1 and i >= batch_size]
            if pos_idxs_over_limit: # since we will be discarding these sentences from input, we remove them from here as well
                coref = coref[:-len(pos_idxs_over_limit),:-len(pos_idxs_over_limit)]

        return token_labels, sent_labels, doc_label, coref


    return token_labels, sent_labels, doc_label

# Not used here. This can be used when predicting with no actual token label available
def model_predict(bert, model, x_test, all_mock_token_labels, all_mock_org_token_labels=[]):
    # These mock_token_labels must be in the json data before fix_token_labels.py is applied. They can be all "O" labels with length the same as tokens before that script is applied.
    all_token_preds = []
    all_sent_preds = []
    all_doc_preds = []
    with torch.no_grad():
        for idx, (batch, mock_token_labels) in enumerate(zip(x_test, all_mock_token_labels)):
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch)
            embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0].detach() # since no gradients will flow back -> Is this necessary since we use torch.no_grad() ?
            embeddings = embeddings.to(model_device)
            token_out, sent_out, doc_out, coref_out = model(embeddings) # TODO : deal with coref_out somehow

            mock_token_labels = mock_token_labels.detach().cpu().numpy()

            token_preds = token_out.detach().cpu().numpy()
            token_preds = np.argmax(token_preds, axis=2) # [Sentences, Seq_len]

            curr_token_preds = []
            for i in range(token_preds.shape[0]): # For each sentence
                preds = token_preds[i,:]
                labs = mock_token_labels[i,:]
                preds = preds[labs != -1].tolist() # get rid of extra subwords, CLS, SEP, PAD

                if all_mock_org_token_labels: # if these are given we can do length fix
                    org_labs = all_mock_org_token_labels[idx][i]
                    preds.extend([14] * (len(org_labs) - len(preds))) # 14 is for "O" label

                curr_token_preds.append(preds)

            all_token_preds.append(curr_token_preds)

            doc_pred = torch.sigmoid(doc_out).detach().cpu().numpy().flatten()
            sent_preds = torch.sigmoid(sent_out).detach().cpu().numpy().flatten()
            all_doc_preds.append(int(doc_pred.tolist()[0] >= 0.5))
            all_sent_preds.append([int(x >= 0.5) for x in sent_preds.tolist()])

    return all_token_preds, all_sent_preds, all_doc_preds

def convert_to_sklearn_format(clusters):
    sentences = sorted(sum(clusters, []))
    labels = list(sentences)
    assert len(set(labels)) == len(labels)

    for i, cl in enumerate(clusters):
        for e in cl:
            labels[sentences.index(e)] = i

    return labels

def test_model(bert, model, x_test, y_test, org_token_labels=[], cluster_golds=[]):
    sent_test_preds = []
    doc_test_preds = []
    all_sent_labels = []
    all_doc_labels = []
    if cluster_golds:
        sent_sizes = []
        coref_scores = []
        all_gold_relations = []
        all_pred_relations = []

    with torch.no_grad():
        test_loss = 0
        all_preds = []
        all_label_ids = []
        for idx, (batch, labels) in enumerate(zip(x_test, y_test)):
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch)
            embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0].detach() # since no gradients will flow back -> Is this necessary since we use torch.no_grad() ?
            embeddings = embeddings.to(model_device)
            token_labels, sent_labels, doc_label = tuple(t.to(model_device) for t in labels)
            token_out, sent_out, doc_out, coref_out = model(embeddings)
            doc_loss = criterion(doc_out.view(-1), doc_label)
            token_loss = token_criterion(token_out.view(-1, num_token_labels), token_labels.view(-1))
            sent_loss = criterion(sent_out.view(-1), sent_labels)
            loss = doc_loss + token_loss + sent_loss

            test_loss += loss.item()

            token_labels = token_labels.detach().cpu().numpy()
            sent_labels = sent_labels.detach().cpu().numpy()
            doc_label = doc_label.detach().cpu().numpy()

            all_sent_labels += sent_labels.tolist()
            all_doc_labels += doc_label.tolist()

            token_preds = token_out.detach().cpu().numpy() # no need for softmax since we only need argmax
            token_preds = np.argmax(token_preds, axis=2) # [Sentences, Seq_len]

            if org_token_labels: # if these are given we can do length fix
                for i in range(len(org_token_labels[idx])): # For each sentence
                    preds = token_preds[i,:]
                    labs = token_labels[i,:]
                    org_labs = org_token_labels[idx][i]

                    preds = preds[labs != -1].tolist() # get rid of extra subwords, CLS, SEP, PAD
                    preds.extend([14] * (len(org_labs) - len(preds))) # 14 is for "O" label
                    all_preds.extend(preds)
                    all_label_ids.extend(org_labs)
            else:
                all_preds.extend(token_preds[token_labels != -1].tolist())
                all_label_ids.extend(token_labels[token_labels != -1].tolist())

            doc_pred = torch.sigmoid(doc_out).detach().cpu().numpy().flatten()
            sent_preds = torch.sigmoid(sent_out).detach().cpu().numpy().flatten()
            doc_test_preds += doc_pred.tolist()
            sent_test_preds += sent_preds.tolist()

            if cluster_golds:
                gold_cluster = cluster_golds[idx]
                if gold_cluster:
                    pos_idxs = sorted([x for c in gold_cluster for x in c])
                    sent_sizes.append(len(pos_idxs))

                    coref_out = torch.sigmoid(coref_out).detach().cpu().numpy()
                    coref_out = coref_out[pos_idxs,:][:,pos_idxs]

                    pred_cluster = clustering_algo(coref_out, pos_idxs)
                    coref_score = adjusted_rand_score(convert_to_sklearn_format(gold_cluster), convert_to_sklearn_format(pred_cluster))
                    coref_scores.append(coref_score)

                    # Just for f1 macro, may remove later
                    gold_relations = cluster_to_relations(gold_cluster, pos_idxs)
                    all_gold_relations.extend(gold_relations)
                    pred_relations = cluster_to_relations(pred_cluster, pos_idxs)
                    all_pred_relations.extend(pred_relations)

    test_loss /= len(x_test)
    (precision, recall, f1), _ = evaluate2([idtolabel[x] for x in all_label_ids], [idtolabel[x] for x in all_preds])
    precision /= 100
    recall /= 100
    f1 /= 100

    if cluster_golds:
        # TODO : Think about returning these values outside of this function
        coref_f1_macro = f1_score(all_gold_relations, all_pred_relations, average="macro")
        macro_rand_score = sum(coref_scores) / len(coref_scores)
        micro_rand_score = sum(s * c for s, c in zip(coref_scores, sent_sizes)) / sum(sent_sizes)
        print("Coreference Adjusted Rand Score; Micro : %.6f , Macro : %.6f . Coreference Relations F1 Macro : %.6f" %(micro_rand_score, macro_rand_score, coref_f1_macro))

    doc_test_score = f1_score(all_doc_labels, [ int(x >= 0.5) for x in doc_test_preds], average="macro")
    sent_test_score = f1_score(all_sent_labels, [ int(x >= 0.5) for x in sent_test_preds], average="macro")

    return f1, sent_test_score, doc_test_score, test_loss

def build_scopeit(train_data, dev_data, pretrained_model, n_epochs=10, model_path="temp.pt"):
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    bert = AutoModel.from_pretrained(pretrained_model)

    x_train = []
    y_train = []
    sent_avail = []
    token_avail = []
    for d in train_data:
        if len(d["tokens"]) > batch_size:
            x_train.append(prepare_set(d["tokens"][:batch_size]))
            y_train.append(prepare_labels(d, truncate=True, coref_gold=True))
        else:
            x_train.append(prepare_set(d["tokens"]))
            y_train.append(prepare_labels(d, truncate=False, coref_gold=True))

        if not generate_labels_for_neg_docs:
            sent_avail.append(d["sent"])
            token_avail.append(d["token"])

    x_dev = []
    y_dev = []
    dev_cluster_golds = []
    for d in dev_data:
        x_dev.append(prepare_set(d["tokens"]))
        y_dev.append(prepare_labels(d))

        clusters = d["event_clusters"]
        cluster_gold = []
        if clusters:
            cluster_gold = to_cluster_gold(clusters)

        dev_cluster_golds.append(cluster_gold)


    model = ScopeIt(bert, hidden_size, num_layers=num_layers, num_token_labels=num_token_labels)
    # model.load_state_dict(torch.load(repo_path + "/models/scopeit_" + model_path))
    model.to(model_device)

    if torch.cuda.device_count() > 1 and bert_device.type == "cuda":
        bert = nn.DataParallel(bert, device_ids=device_ids[1:])

    # bert.load_state_dict(torch.load(repo_path + "/models/bert_" + model_path))
    bert.to(bert_device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if model_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    total_steps = len(x_train) * n_epochs
    model_optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # fine tune bert
    if fine_tune_bert:
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

        # shuffle training data
        train_data = list(zip(x_train, y_train, sent_avail, token_avail))
        random.shuffle(train_data)
        x_train, y_train, sent_avail, token_avail = zip(*train_data)
        ##

        print("Starting Epoch %d"%(epoch+1))
        for step, (batch, labels) in enumerate(tqdm(zip(x_train, y_train), desc="Iteration")): # each doc is a batch
            b_input_ids, b_input_mask, b_token_type_ids = tuple(t.to(bert_device) for t in batch)

            if fine_tune_bert:
                embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0] #.detach()
            else:
                embeddings = bert(b_input_ids, attention_mask=b_input_mask, token_type_ids=b_token_type_ids)[0].detach()

            token_labels, sent_labels, doc_label, coref_gold = tuple(t.to(model_device) for t in labels)
            embeddings = embeddings.to(model_device)
            token_out, sent_out, doc_out, coref_out = model(embeddings)

            doc_loss = criterion(doc_out.view(-1), doc_label)
            loss = doc_loss
            if not generate_labels_for_neg_docs:
                if sent_avail[step]:
                    sent_loss = criterion(sent_out.view(-1), sent_labels)
                    loss += sent_loss

                if token_avail[step]:
                    token_loss = token_criterion(token_out.view(-1, num_token_labels), token_labels.view(-1))
                    loss += token_loss

                    if len(coref_gold) != 0: # This is empty if there is only one positive sentence
                        pos_indexes = torch.arange(len(sent_labels))[sent_labels == 1]
                        fixed_coref_out = coref_out[pos_indexes,:][:,pos_indexes] # Pos_sentsxPos_sents
                        fixed_coref_out = fixed_coref_out[coref_gold != -1]
                        coref_gold = coref_gold[coref_gold != -1]
                        coref_loss = criterion(fixed_coref_out, coref_gold)
                        loss += coref_loss

            else:
                token_loss = token_criterion(token_out.view(-1, num_token_labels), token_labels.view(-1))
                sent_loss = criterion(sent_out.view(-1), sent_labels)
                loss += token_loss + sent_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # fine tune bert
            if fine_tune_bert:
                torch.nn.utils.clip_grad_norm_(bert.parameters(), 1.0)
                bert_optimizer.step()
                scheduler.step()
            ####

            model_optimizer.step()
            train_loss += loss.item()
            model.zero_grad()
            bert.zero_grad()

        train_loss /= len(x_train)
        elapsed = time.time() - start_time

        # Validation
        bert.eval()
        model.eval()
        token_val_score, sent_val_score, doc_val_score, val_loss = test_model(bert, model, x_dev, y_dev, cluster_golds=dev_cluster_golds)
        print("Epoch %d - Train loss: %.4f. Document Validation Score: %.4f. Sentence Validation Score: %.4f. Token Validation Score: %.4f.  Validation loss: %.4f. Elapsed time: %.2fs."% (epoch + 1, train_loss, doc_val_score, sent_val_score, token_val_score, val_loss, elapsed))
        val_score = (token_val_score + sent_val_score + doc_val_score) / 3
        if val_score > best_score:
            print("Saving model!")
            torch.save(model.state_dict(), repo_path + "/models/scopeit_" + model_path)
            # bert_to_save = bert.module if hasattr(bert, 'module') else bert  # To handle multi gpu
            torch.save(bert.state_dict(), repo_path + "/models/bert_" + model_path)
            best_score = val_score

        print("========================================================================")

    return bert, model

if __name__ == '__main__':
    train = read_file(repo_path + "/data/fixed_no_gen_train_data.json")
    # train = read_file(repo_path + "/data/fixed_train_data.json")
    dev = read_file(repo_path + "/data/fixed_dev_data.json")
    # Test file must contain one more column than others, the "old_token_labels" referring to original token_labels before fix_token_labels.py is applied. This is needed in length_fix for testing.
    test = read_file(repo_path + "/data/fixed_test_data.json")
    # test = read_file(repo_path + "/data/fixed_pipeline_review_data_only_pos.json")
    # test = read_file(repo_path + "/data/fixed_pipeline_review_data.json")

    # TODO : print parameters here
    print("max batch size (max sentences in doc): ", batch_size)

    model_path = str(max_length) + "_" + str(num_layers) + "_" + str(hidden_size) + "_" + str(batch_size) + "_" + str(lr) + "_" + str(generate_labels_for_neg_docs) + "_coref_" + str(number_of_mlps) + "mlp_aftergru" + str(aftergru) + ".pt"
    if not only_test:
        bert, model = build_scopeit(train, dev, "bert-base-uncased", n_epochs=num_epochs, model_path=model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        bert = AutoModel.from_pretrained("bert-base-uncased")
        model = ScopeIt(bert, hidden_size, num_layers=num_layers, num_token_labels=num_token_labels)
        if torch.cuda.device_count() > 1 and bert_device.type == "cuda":
            bert = nn.DataParallel(bert, device_ids=device_ids[1:])

    model.load_state_dict(torch.load(repo_path + "/models/scopeit_" + model_path))
    model.to(model_device)
    bert.load_state_dict(torch.load(repo_path + "/models/bert_" + model_path))
    bert.to(bert_device)
    bert.eval()
    model.eval()

    x_test = [prepare_set(d["tokens"]) for d in test]
    # Normally when predicting we would not use any labels. But, here we need some token_labels to do length fix. These can be any mock labels assigned randomly.
    if predict:
        y_test = [prepare_labels(t, only_token=True) for t in test]
    else:
        y_test = []
        test_cluster_golds = []
        for t in test:
            y_test.append(prepare_labels(t))

            clusters = t["event_clusters"]
            cluster_gold = []
            if clusters:
                cluster_gold = to_cluster_gold(clusters)

            test_cluster_golds.append(cluster_gold)


    original_token_labels = [d["old_token_labels"] for d in test] # Necessary for length fix

    if predict:
        all_token_preds, all_sent_preds, all_doc_preds = model_predict(bert, model, x_test, y_test, all_mock_org_token_labels=original_token_labels)
        with open(repo_path + "/data/test_with_preds.json", "w", encoding="utf-8") as g:
            for i, t in enumerate(test):
                t["doc_pred"] = all_doc_preds[i]
                t["sent_preds"] = all_sent_preds[i]
                t["token_preds"] = all_token_preds[i]
                g.write(json.dumps(t) + "\n")
    else:
        token_test_score, sent_test_score, doc_test_score, test_loss = test_model(bert, model, x_test, y_test, org_token_labels=original_token_labels, cluster_golds=test_cluster_golds)
        print("Document F1 Macro: %.6f. Sentence F1 Macro: %.6f. Token F1 Macro: %.6f.  Test loss: %.6f."% (doc_test_score, sent_test_score, token_test_score, test_loss))
