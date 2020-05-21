from transformers import AutoTokenizer
import pandas as pd
import sys
from itertools import combinations
import ipdb

input_file = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Need this label list to be the same across all the scripts
label_list = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger", "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O"]
no_gen = True

label_map = {}
for i,lab in enumerate(label_list):
    label_map[lab] = i

def labels_to_idxs(labels): # returns empty if empty
    return [[label_map[x] for x in labs] for labs in labels]

# We could have processed all sentences' tokens in a document at the same time, but returning list of lists as a result disrupts the use of this function in apply
# def get_tokenized(tokenizer, tokens, token_labels, no_gen=False):
#     do_labels = False
#     if not no_gen:
#         do_labels = True
#     else:
#         if token_labels:
#             do_labels = True

#     new_tokens = []
#     new_token_labels = []
#     old_token_labels = []
#     for i, tok in enumerate(tokens):
#         new_toks = tokenizer.tokenize(tok)

#         if len(new_toks) > 0:
#             # TODO : Sometimes words get divided, but not as subwords. For example; free-for-all to free, -, for, -, all or 20:30 to 20, :, 30. What to do? Just do as what we do with ## ones? -> We do it like this here, but this can be changed!
#             new_tokens.extend(new_toks)
#             if do_labels:
#                 new_token_labels.append(token_labels[i])
#                 new_token_labels.extend([-1] * (len(new_toks) - 1)) # we do not get loss for extra words, but they still get subjected to attention since we don't mask them.
#                 old_token_labels.append(token_labels[i])

#     return new_tokens, new_token_labels, old_token_labels

def to_coref_gold(row):
    if "token" in row.index and not row.token:
        return list()

    if len(row.event_clusters) == 0:
        return list()

    pos_sents = [i for i,v in enumerate(row.sent_labels) if v == 1] # sorted
    if len(pos_sents) == 1:
        return list()

    sent_to_order = {}
    for i,v in enumerate(pos_sents):
        sent_to_order[v] = i

    coref_gold =  [[0 if i != j else -1 for j in range(len(pos_sents))] for i in range(len(pos_sents))] # -1 for diagonals, we ignore these in loss
    for v in row.event_clusters.values():
        for i,j in list(combinations(v,2)): # for each pair
            first_sent = sent_to_order[i-1] # idxs start from 1
            second_sent = sent_to_order[j-1] # idxs start from 1

            coref_gold[first_sent][second_sent] = 1
            coref_gold[second_sent][first_sent] = 1 # symmetric

    return coref_gold

def get_tokenized(tokenizer, row, no_gen=False):
    do_labels = False
    if not no_gen:
        do_labels = True
    else:
        if row.token_labels:
            do_labels = True

    all_new_tokens = []
    all_new_token_labels = []
    all_old_token_labels = []
    for i, toks in enumerate(row.tokens):
        new_tokens = []
        new_token_labels = []
        old_token_labels = []
        for j,tok in enumerate(toks):
            new_toks = tokenizer.tokenize(tok)

            if len(new_toks) > 0:
                # TODO : Sometimes words get divided, but not as subwords. For example; free-for-all to free, -, for, -, all or 20:30 to 20, :, 30. What to do? Just do as what we do with ## ones? -> We do it like this here, but this can be changed!
                new_tokens.extend(new_toks)
                if do_labels:
                    new_token_labels.append(row.token_labels[i][j])
                    new_token_labels.extend([-1] * (len(new_toks) - 1)) # we do not get loss for extra words, but they still get subjected to attention since we don't mask them.
                    old_token_labels.append(row.token_labels[i][j])

        all_new_tokens.append(new_tokens)
        if do_labels:
            all_new_token_labels.append(new_token_labels)
            all_old_token_labels.append(old_token_labels)

    return all_new_tokens, all_new_token_labels, all_old_token_labels


def remove_empty_sentences(row):
    for i,toks in enumerate(row.tokens):
        if len(toks) == 0:
            row.tokens.pop(i)
            if row.token_labels:
                row.token_labels.pop(i)
                row.old_token_labels.pop(i) # Necessary for length fix in test

            if len(row.event_clusters) != 0:
                new_clusters = {}
                for k,v in row.event_clusters.items():
                    new_clusters[k] = [idx-1 if idx >= i else idx for idx in v]

                row.event_clusters = new_clusters

            if row.sentences:
                row.sentences.pop(i)
                row.sent_labels.pop(i)
                # row.semantic_labels.pop(i) # TODO : Make sure length of these match the length of sentences

    return row


df = pd.read_json(input_file, orient="records", lines=True)
df.token_labels = df.token_labels.apply(labels_to_idxs)
df["old_token_labels"] = df.token_labels

# r = df.apply(lambda row: [get_tokenized(tokenizer, toks, row.token_labels[i] if row.token_labels else [], no_gen=no_gen) for i,toks in enumerate(row.tokens)], axis=1)
# df["tokens"] = r.apply(lambda x: [a[0] for a in x])
# df["token_labels"] = r.apply(lambda x: [a[1] for a in x])
# df["old_token_labels"] = r.apply(lambda x: [a[2] for a in x])

r = df.apply(lambda row: get_tokenized(tokenizer, row, no_gen=no_gen), axis=1)
df["tokens"] = r.apply(lambda x: x[0])
df["token_labels"] = r.apply(lambda x: x[1])
df["old_token_labels"] = r.apply(lambda x: x[2])

# We need this part since the tokenizer may return an empty list for a sentence. For example a sentence just contains "\222\222", then it would be empty after tokenized.
df = df.apply(remove_empty_sentences, axis=1)

# Coreference gold data
df["coref_gold"] = df.apply(to_coref_gold, axis=1)

df.to_json("fixed_" + input_file, orient="records", lines=True, force_ascii=False)
# NOTE : In train.py these will be fed into batch_encode_plus function with is_pretokenized option as True, so that our tokens and token_labels match!
