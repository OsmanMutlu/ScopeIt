from torch import nn
import torch.nn.functional as F
import torch

# this implementation separates scopeit model from the embedder (bert). 
# We are doing this to avoid using torch.nn.dataparallel since it has some problems.
# see: https://github.com/pytorch/pytorch/issues/7092#issuecomment-385357970
# this problem isn't solved when flatten_parameters is used too.  
class ScopeIt(nn.Module):
    def __init__(self, bert, hidden_size, num_layers=1, dropout=0.1):
        super(ScopeIt, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = bert.config.hidden_size
        self.bigru1 = nn.GRU(self.embedding_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.bigru2 = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)

        self.sent_boomer = Boom(hidden_size * 2, dim_feedforward=hidden_size*2*4, dropout=dropout, shortcut=True)
        self.sent_linear = nn.Linear(self.hidden_size * 2, 1)

        self.doc_boomer = Boom(hidden_size, dim_feedforward=hidden_size*4, dropout=dropout, shortcut=True)
        self.doc_linear = nn.Linear(self.hidden_size, 1)

    def forward(self, embeddings): 
        # embeddings ->[Sentences, SeqLen, BERT_Hidden]
        
        bigru1_last_hidden = self.bigru1(embeddings)[1] # pass the output of bert through the first bigru to get sentence embeddings
        sent_embeddings = bigru1_last_hidden[0, :, :] + bigru1_last_hidden[1, :, :] # here we add the output of two GRUs (forward and backward) 
        # sent_embeddings -> [Sentences, HiddenSize]

        bigru2_output = self.bigru2(sent_embeddings.unsqueeze(0))
        
        boomed_sents = self.sent_boomer(bigru2_output[0])
        # boomed_sents -> [Sentences, HiddenSize]

        doc_embeddings = bigru2_output[1][0, :, :] + bigru2_output[1][1, :, :] # here we add the output of two GRUs (forward and backward) 
        boomed_doc = self.doc_boomer(doc_embeddings)
        # boomed_sents -> [1, HiddenSize]

        # final output -> [Sentences, logit]
        sent_logits = self.sent_linear(boomed_sents).squeeze(0)
        doc_logit = self.doc_linear(boomed_doc)

        return torch.cat((sent_logits, doc_logit))


class Boom(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, shortcut=False):
        super(Boom, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) if dropout else None
        if not shortcut:
            self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.shortcut = shortcut
        self.act = nn.GeLU()
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
            #h = h + self.drop(x).sum(dim=-2)
            z = x.sum(dim=-2)
        else:
            z = self.linear2(x)

        return z