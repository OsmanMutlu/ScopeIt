from torch import nn
import torch.nn.functional as F
import torch
# import ipdb

# this implementation separates scopeit model from the embedder (bert).
# We are doing this to avoid using torch.nn.dataparallel since it has some problems.
# see: https://github.com/pytorch/pytorch/issues/7092#issuecomment-385357970
# this problem isn't solved when flatten_parameters is used too.
class ScopeIt(nn.Module):
    def __init__(self, bert, hidden_size, num_layers=1, dropout=0.1, num_token_labels=15):
        super(ScopeIt, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = bert.config.hidden_size
        self.bigru1 = nn.GRU(self.embedding_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.bigru2 = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

        # In case we use the biGRU's output for token classification
        self.token_boomer = Boom(hidden_size * 2, dim_feedforward=hidden_size*2*4, dropout=dropout, shortcut=True)
        self.token_linear = nn.Linear(self.hidden_size * 2, num_token_labels)

        # In case we use BERT embeddings for token classification
        # self.token_boomer = Boom(hidden_size, dim_feedforward=hidden_size*4, dropout=dropout, shortcut=True)
        # self.token_linear = nn.Linear(self.hidden_size, num_token_labels)

        self.sent_boomer = Boom(hidden_size * 2, dim_feedforward=hidden_size*2*4, dropout=dropout, shortcut=True)
        self.sent_linear = nn.Linear(self.hidden_size * 2, 1)

        self.doc_boomer = Boom(hidden_size, dim_feedforward=hidden_size*4, dropout=dropout, shortcut=True)
        self.doc_linear = nn.Linear(self.hidden_size, 1)

    def forward(self, embeddings): # embeddings ->[Sentences, SeqLen, BERT_Hidden]

        # In case we use the biGRU's output for token classification
        bigru1_all_hiddens, bigru1_last_hidden = self.bigru1(embeddings) # pass the output of bert through the first bigru to get sentence and token embeddings
        # bigru1_all_hiddens -> [Sentences, SeqLen, Hidden * 2]
        boomed_tokens = self.token_boomer(bigru1_all_hiddens) # boomed_tokens -> [Sentences, SeqLen, Hidden * 2]
        token_logits = self.token_linear(boomed_tokens) # token_logits -> [Sentences, SeqLen, num_token_labels]

        # In case we use BERT embeddings for token classification
        # bigru1_last_hidden = self.bigru1(embeddings)[1] # pass the output of bert through the first bigru to get sentence embeddings
        # boomed_tokens = self.token_boomer(embeddings)
        # token_logits = self.token_linear(boomed_tokens)

        sent_embeddings = bigru1_last_hidden[0, :, :] + bigru1_last_hidden[1, :, :] # here we add the output of two GRUs (forward and backward)
        # sent_embeddings -> [Sentences, HiddenSize]

        bigru2_output = self.bigru2(sent_embeddings.unsqueeze(0))

        boomed_sents = self.sent_boomer(bigru2_output[0])
        # boomed_sents -> [Sentences, HiddenSize]

        doc_embeddings = bigru2_output[1][0, :, :] + bigru2_output[1][1, :, :] # here we add the output of two GRUs (forward and backward)
        boomed_doc = self.doc_boomer(doc_embeddings)
        # boomed_doc -> [1, HiddenSize]

        sent_logits = self.sent_linear(boomed_sents).squeeze(0)
        doc_logit = self.doc_linear(boomed_doc)

        return token_logits, sent_logits, doc_logit # multi-task model training
        # return token_logits, sent_logits, doc_logit, sent_embeddings # coref_head_training aftergru1
        # return token_logits, sent_logits, doc_logit, bigru2_output[0].squeeze(0) # coref_head_training aftergru2


class Boom(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut=False):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if not shortcut: # No idea why this "shorcut" is used?
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.shortcut = shortcut
        self.act = nn.GELU()
        #self.act = nn.Tanh()

    def forward(self, input):
        x = self.act(self.linear1(input))
        if self.dropout: x = self.dropout(x)
        if self.shortcut:
            # Trim the end off if the size is different
            ninp = input.shape[-1]
            x = torch.narrow(x, -1, 0, x.shape[-1] // ninp * ninp)
            # Divide the hidden size evenly into chunks
            x = x.view(*x.shape[:-1], x.shape[-1] // ninp, ninp)
            # Collapse the chunks through summation
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)

        return z

class CorefHead(nn.Module):
    def __init__(self, d_model, dropout=0.1, shortcut=True):
        super(CorefHead, self).__init__()
        self.coref_mlp1 = Boom(d_model, dim_feedforward=d_model*4, dropout=dropout, shortcut=shortcut)
        # self.coref_mlp2 = Boom(d_model, dim_feedforward=d_model*4, dropout=dropout, shortcut=shortcut)
        self.biaffine = BiAffine(d_model)

    def forward(self, x):
        mlp1_out = self.coref_mlp1(x) # [Sentences, HiddenSize]
        # mlp2_out = self.coref_mlp2(x) # [Sentences, HiddenSize]
        coref_logits = self.biaffine(mlp1_out, mlp1_out) # [Sentences, Sentences]

        return coref_logits

# No need for an extra output dimension
class BiAffine(nn.Module):
    def __init__(self, d_model):
        super(BiAffine, self).__init__()
        self.U = nn.Parameter(torch.FloatTensor(d_model, d_model))
        nn.init.xavier_uniform(self.U)

    def forward(self, a, b):
        out = a @ self.U @ b.transpose(0,1)
        return out
