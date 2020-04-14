from torch import nn
import torch.nn.functional as F
import torch

# this implementation separates scopeit model from the embedder (bert). 
# We are doing this to avoid using torch.nn.dataparallel since it has some problems.
# see: https://github.com/pytorch/pytorch/issues/7092#issuecomment-385357970
# this problem isn't solved when flatten_parameters is used too.  
class ScopeIt(nn.Module):
    def __init__(self, bert, hidden_size, num_layers=1):
        super(ScopeIt, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = bert.config.hidden_size
        self.bigru1 = nn.GRU(self.embedding_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.bigru2 = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size * 2, 1)

        # TODO:
        # try adding boom layer.
        # test using multiple layers in GRU units.

    def forward(self, embeddings): # embeddings ->[Sentences, SeqLen, BERT_Hidden]
        # BIGRU_1 get sentence embeddings -> [Sentences, HiddenSize]
        bigru1_last_hidden = self.bigru1(embeddings)[1] # pass the output of bert through the first bigru to get sentence embeddings
        sent_embeddings = bigru1_last_hidden[0, :, :] + bigru1_last_hidden[1, :, :] # here we add the output of two GRUs (forward and backward) 

        # BIGRU_2 inter sentence aggregator -> [Sentences, HiddenSize]
        bigru2_output = self.bigru2(sent_embeddings.unsqueeze(0))[0]

        # final output -> [Sentences, logit]
        logits = self.linear(bigru2_output).squeeze(0)
        return logits