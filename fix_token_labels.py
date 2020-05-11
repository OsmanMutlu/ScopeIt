from transformers import AutoTokenizer
import pandas as pd
import sys

input_file = sys.argv[1]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Need this label list to be the same across all the scripts
label_list = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger", "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O"]

label_map = {}
for i,lab in enumerate(label_list):
    label_map[lab] = i

def labels_to_idxs(labels):
    return [[label_map[x] for x in labs] for labs in labels]

# We could have processed all sentences' tokens in a document at the same time, but returning list of lists as a result disrupts the use of this function in apply
def get_tokenized(tokenizer, tokens, token_labels):
    new_tokens = []
    new_token_labels = []
    old_token_labels = []
    for i, tok in enumerate(tokens):
        new_toks = tokenizer.tokenize(tok)

        if len(new_toks) > 0:
            # TODO : Sometimes words get divided, but not as subwords. For example; free-for-all to free, -, for, -, all or 20:30 to 20, :, 30. What to do? Just do as what we do with ## ones? -> We do it like this here, but this can be changed!
            new_tokens.extend(new_toks)
            new_token_labels.append(token_labels[i])
            new_token_labels.extend([-1] * (len(new_toks) - 1)) # we do not get loss for extra words, but they still get subjected to attention since we don't mask them.
            old_token_labels.append(token_labels[i])

    return new_tokens, new_token_labels, old_token_labels

def remove_empty_sentences(row):
    for i,toks in enumerate(row.tokens):
        if len(toks) == 0:
            row.tokens.pop(i)
            row.token_labels.pop(i)
            row.old_token_labels.pop(i) # Necessary for length fix in test
            row.sentences.pop(i)
            row.sent_labels.pop(i)
            # row.semantic_labels.pop(i) # Make sure length of these match the length of sentences
            # TODO : Remove this sentence from coreference data

    return row

df = pd.read_json(input_file, orient="records", lines=True)
df.token_labels = df.token_labels.apply(labels_to_idxs)
df["old_token_labels"] = df.token_labels

r = df.apply(lambda row: [get_tokenized(tokenizer, toks, labs) for toks, labs in zip(row.tokens, row.token_labels)], axis=1)
df["tokens"] = r.apply(lambda x: [a[0] for a in x])
df["token_labels"] = r.apply(lambda x: [a[1] for a in x])
df["old_token_labels"] = r.apply(lambda x: [a[2] for a in x])

# We need this part since the tokenizer may return an empty list for a sentence. For example a sentence just contains "\222\222", then it would be empty after tokenized.
df = df.apply(remove_empty_sentences, axis=1)

df.to_json("fixed_" + input_file, orient="records", lines=True, force_ascii=False)
# NOTE : In train.py these will be fed into batch_encode_plus function with is_pretokenized option as True, so that our tokens and token_labels match!
