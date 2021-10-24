# Ahahhaha scratch.py is back with another installment of ML frickery!
# get ready to be fricked by ML!

from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from collections import defaultdict

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import inaugural

from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from transformers import AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import re

# Simplicity evalutaion
def unique_words(sentence):
    words = [i.lower() for i in word_tokenize(sentence)]
    seen = []

    wc = 0
    for word in words:
        if word not in seen:
            wc += 1
            seen.append(word)

    return wc

# Find out the sentence similarity between two sentences
# using a model.
def semantic_similarity(a,b,model):
    a,b = model.encode([a,b])
    return util.pytorch_cos_sim(a,b)[0][0].item()

# Mixed simplification reward 
def reward(src,tgt,reduce_factor,model,similarity_mix=0.7):
    scaled_similarity = similarity_mix*semantic_similarity(src,tgt,model)

    a = unique_words(src)
    b = unique_words(tgt)
    target_vocab_size = a*reduce_factor

    vocab_reduction = 1-abs(b-target_vocab_size)/b
    scaled_vocab_reduction = vocab_reduction *(1-similarity_mix)

    return scaled_similarity+scaled_vocab_reduction

##############################
### zach you converted me ####
##############################

detokenizer = TreebankWordDetokenizer()
dataset = [detokenizer.detokenize(i) for i in inaugural.sents()]

dataset_train = dataset[:int(len(dataset)*0.9)]

bart_config = BartConfig.from_pretrained("facebook/bart-base")
# TODO any max length stuff
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", config=bart_config)

class Critic(nn.Module):
    def __init__(self, vocab_size, max_length):
        super(Critic,self).__init__()

        self.d1 = nn.Linear(vocab_size, 1024)
        self.d2 = nn.Linear(1024, 512)
        self.d3 = nn.Linear(512, 32)
        self.flatten = nn.Flatten()
        self.d4 = nn.Linear(32*max_length, 1)

    def forward(self,x):
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))
        x = F.relu(self.flatten(x))
        x = F.sigmoid(self.d4(x))
        return x

MAX_LENGTH = 50
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
np2tens = lambda x: torch.tensor(x)

bart_model.to(DEVICE)
bart_model.train()

critic_model = Critic(bart_tokenizer.vocab_size, MAX_LENGTH)
critic_model.to(DEVICE)
critic_model.train()

similarity_model = SentenceTransformer('stsb-bert-base')

# for sentence in dataset_train:
batch = [dataset_train[0], dataset_train[1]]

sentence_encoded = [bart_tokenizer.encode(i)[:MAX_LENGTH] for i in batch]
sentence_padded = np2tens([i + [1 for _ in range(MAX_LENGTH-len(i))] for i in sentence_encoded]).to(DEVICE)
sentence_mask = np2tens([[1 for _ in range(len(i))] + [0 for _ in range(MAX_LENGTH-len(i))] for i in sentence_encoded]).to(DEVICE)

result = bart_model(sentence_padded, attention_mask=sentence_mask)
logits = result["logits"]
values = critic_model(logits)

logits_string = [bart_tokenizer.decode(i) for i in [torch.argmax(i, axis=1) for i in logits.detach()]]
logits_string = [re.sub("<s>", "", i.split("</s>")[0]) for i in logits_string]
rewards = [reward(i,j,0.7,similarity_model) for i,j in zip(batch, logits_string)]

advantages = 

# optim = AdamW(bart_model.parameters(), lr=config.learning_rate)
