from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
from src.conlleval import evaluate2
import ipdb

label_list = ["B-etime", "B-fname", "B-organizer", "B-participant", "B-place", "B-target", "B-trigger", "I-etime", "I-fname", "I-organizer", "I-participant", "I-place", "I-target", "I-trigger", "O"]
idtolabel = {}
for i,lab in enumerate(label_list):
    idtolabel[i] = lab

df = pd.read_json("data/test_with_preds.json", orient="records", lines=True)

doc_labels = df.doc_label
doc_preds = df.doc_pred

sent_labels = [label for labels in df.sent_labels.tolist() for label in labels]
sent_preds = [pred for preds in df.sent_preds.tolist() for pred in preds]

all_token_labels = []
all_token_preds1 = [] # Only Tok
all_token_preds2 = [] # Doc + Tok
all_token_preds3 = [] # Sent + Tok
all_token_preds4 = [] # Doc + Sent + Tok
for i, (doc_tok_labels, doc_tok_preds) in enumerate(zip(df.old_token_labels.tolist(), df.token_preds.tolist())):
    doc_neg = False
    if df.iloc[i].doc_pred == 0:
        doc_neg = True

    for j, (sent_tok_labels, sent_tok_preds) in enumerate(zip(doc_tok_labels, doc_tok_preds)):
        sent_neg = False
        if df.iloc[i].sent_preds[j] == 0:
            sent_neg = True

        all_token_labels.extend([idtolabel[x] for x in sent_tok_labels])
        all_token_preds1.extend([idtolabel[x] for x in sent_tok_preds])

        if doc_neg and sent_neg:
            all_token_preds2.extend(["O" for x in sent_tok_preds])
            all_token_preds3.extend(["O" for x in sent_tok_preds])
            all_token_preds4.extend(["O" for x in sent_tok_preds])
        elif doc_neg:
            all_token_preds2.extend(["O" for x in sent_tok_preds])
            all_token_preds3.extend([idtolabel[x] for x in sent_tok_preds])
            all_token_preds4.extend(["O" for x in sent_tok_preds])
        elif sent_neg:
            all_token_preds2.extend([idtolabel[x] for x in sent_tok_preds])
            all_token_preds3.extend(["O" for x in sent_tok_preds])
            all_token_preds4.extend(["O" for x in sent_tok_preds])
        else:
            all_token_preds2.extend([idtolabel[x] for x in sent_tok_preds])
            all_token_preds3.extend([idtolabel[x] for x in sent_tok_preds])
            all_token_preds4.extend([idtolabel[x] for x in sent_tok_preds])

# DOC LEVEL
print("DOC LEVEL RESULTS : ")
# print(classification_report(doc_labels, doc_preds, digits=4))
doc_precision, doc_recall, doc_f1, _ = precision_recall_fscore_support(doc_labels, doc_preds, average="macro", labels=list(range(0,2)))
print("Prec : %.4f, Recall : %.4f, F1 : %.4f" %(doc_precision, doc_recall, doc_f1))
print("")

# SENT LEVEL
print("SENT LEVEL RESULTS : ")
# print(classification_report(sent_labels, sent_preds, digits=4))
sent_precision, sent_recall, sent_f1, _ = precision_recall_fscore_support(sent_labels, sent_preds, average="macro", labels=list(range(0,2)))
print("Prec : %.4f, Recall : %.4f, F1 : %.4f" %(sent_precision, sent_recall, sent_f1))
print("")

# TOK LEVEL
print("TOK LEVEL RESULTS : ")
(precision, recall, f1), _ = evaluate2(all_token_labels, all_token_preds1, verbose=False)
print("Only Tok : Prec : %.4f, Recall : %.4f, F1 : %.4f" %(precision, recall, f1))

(precision, recall, f1), _ = evaluate2(all_token_labels, all_token_preds3, verbose=False)
print("Sent + Tok : Prec : %.4f, Recall : %.4f, F1 : %.4f" %(precision, recall, f1))

(precision, recall, f1), _ = evaluate2(all_token_labels, all_token_preds2, verbose=False)
print("Doc + Tok : Prec : %.4f, Recall : %.4f, F1 : %.4f" %(precision, recall, f1))

(precision, recall, f1), _ = evaluate2(all_token_labels, all_token_preds4, verbose=False)
print("Doc + Sent + Tok : Prec : %.4f, Recall : %.4f, F1 : %.4f" %(precision, recall, f1))
