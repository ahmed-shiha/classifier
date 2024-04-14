import json
import numpy as np
from torch import nn
import torch
# from transformers import BertModel
from transformers import AutoModelForSequenceClassification
with open("config.json") as json_file:
    config = json.load(json_file)

int2Label = {0: 'anxiety',
            1: 'depression',
            2: 'normal',
            3: 'ocd',
            4: 'other',
            5: 'suicide'
} 


class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained('assets', num_labels=n_classes)
        # self.drop = nn.Dropout(p=0.3)
        # self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        logits = self.bert(input_ids=input_ids, attention_mask=attention_mask)['logits']
        # _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # output = self.drop(pooled_output)
        # output = np.argmax(logits.cpu(), axis=1).flatten()
        _, top_indices = torch.sort(logits, descending=True)
        top_indices = top_indices[:, :2]
        # print(int2Label[output.item()])
        # print(top_indices[0][0])
        # return int2Label[output.item()]
        return top_indices[0]
