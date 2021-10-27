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
from torch.distributions import Categorical

import numpy as np
import re

from tqdm import tqdm

import wandb

print("Hello and welcome. Let's get started.")

# Parametres

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hyperparametre_defaults = dict(
    max_length = 50,
    actor_lr = 1e-6,
    critic_lr = 5e-6,
    batch_size = 16,
    epochs = 1000
)

run = wandb.init(entity='inscriptio', project='leslie', config=hyperparametre_defaults)
# run = wandb.init(entity='inscriptio', project='leslie', config=hyperparametre_defaults, mode="disabled")
config = wandb.config

MAX_LENGTH = config.max_length
ACTOR_LR = config.actor_lr
CRITIC_LR = config.critic_lr
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs

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
def reward(src,tgt,reduce_factor,model,similarity_mix=0.5):
    scaled_similarity = similarity_mix*semantic_similarity(src,tgt,model)

    a = unique_words(src)
    b = unique_words(tgt)
    target_vocab_size = a*reduce_factor

    try: 
        vocab_reduction = 1-abs(b-target_vocab_size)/b
        scaled_vocab_reduction = vocab_reduction *(1-similarity_mix)
    except ZeroDivisionError:
        scaled_vocab_reduction = 0

           # subtract by 0.5 to be able to distribute negative reward
    return (scaled_similarity+scaled_vocab_reduction)-0.5

    # return semantic_similarity(src,tgt,model)

##############################
### zach you converted me ####
##############################

print("Establishing tokenizers and actor model.")
detokenizer = TreebankWordDetokenizer()
dataset = [detokenizer.detokenize(i) for i in inaugural.sents()]

dataset_train = dataset[:int(len(dataset)*0.9)]

bart_config = BartConfig.from_pretrained("facebook/bart-base")
# TODO any max length stuff
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", config=bart_config)

print("Creating critic model.")
class Critic(nn.Module):
    def __init__(self, max_length):
        super(Critic,self).__init__()

        self.batchnorm = nn.BatchNorm1d(max_length)
        self.d1 = nn.Linear(max_length, 32)
        self.d2 = nn.Linear(32, 1)

    def forward(self,x):
        x = self.batchnorm(x)
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return x

critic_model = Critic(MAX_LENGTH)

np2tens = lambda x: torch.tensor(x)

print("Creating optimizers and moving models.")
bart_model.to(DEVICE)
bart_model.train()
actor_optim = AdamW(bart_model.parameters(), lr=ACTOR_LR)

critic_model.to(DEVICE)
critic_model.train()
critic_optim = AdamW(critic_model.parameters(), lr=CRITIC_LR)

run.watch((bart_model, critic_model))

print("Instantiating similarity model.")
similarity_model = SentenceTransformer('stsb-bert-base')

print("Starting to train.")
for _ in range(EPOCHS):
    dataset_bar = tqdm(range(0, len(dataset_train)-BATCH_SIZE, BATCH_SIZE))
    for i, batch_id in enumerate(dataset_bar):
        batch = dataset_train[batch_id:batch_id+BATCH_SIZE]

        # Encode each sentence 
        sentence_encoded = [bart_tokenizer.encode(i)[:MAX_LENGTH] for i in batch]
        # Pad the encoded result to the MAX_LENGTH
        sentence_padded = np2tens([i + [1 for _ in range(MAX_LENGTH-len(i))] for i in sentence_encoded]).to(DEVICE)
        # Mask the attention such that only non-padded values are available
        sentence_mask = np2tens([[1 for _ in range(len(i))] + [0 for _ in range(MAX_LENGTH-len(i))] for i in sentence_encoded]).to(DEVICE)

        # Put it through the model!
        result = bart_model(sentence_padded, attention_mask=sentence_mask)
        # Logit-shame the state. We support logit-shaming
        # to figure out what the value of the state is 
        values = torch.flatten(critic_model(sentence_padded.float()))
        # Get the logits
        logits = result["logits"]

        # Select for the predicted outputs
        actions = torch.stack([torch.argmax(i, axis=1) for i in logits.detach()])
        # Stringify the outputs
        logits_string = [bart_tokenizer.decode(i) for i in actions]
        # Return the final string
        logits_string = [re.sub("<s>", "", i.split("</s>")[0]) for i in logits_string]
        # Calculate reward value for these strings
        rewards = np2tens([reward(i,j,0.5,similarity_model,0.5) for i,j in zip(batch, logits_string)]).to(DEVICE)

        # Calculate the advantages of the model
        advantages = rewards - values

        # Calculate the log probabilites of the model output by softmaxing it
        logits_probs = F.softmax(logits, dim=2)

        # Gather the log probability values of the selected actions
        action_log_probs = logits_probs.gather(2, torch.unsqueeze(actions, 2)).log()

        # Add em up, multiply by the advantage, and calculate the mean
        actor_loss = torch.stack(
                        [a*l for a,l in zip(advantages.detach(),action_log_probs)]
                     ).mean() * -1
        # The critic wants to match the actual rewards as much as possible
        # So it's just MSE loss 
        critic_loss = advantages.pow(2).mean()

        dataset_bar.set_description(f"Sample: {batch_id}, Actor: {round(actor_loss.item(),3)}, Critic: {round(critic_loss.item(),3)}, Reward: {round(rewards[0].item(),3)}")

        if batch_id % 500 == 0:
            print(f"Sample: {batch_id}; Output: {logits_string[0]}")

        # Add the losses
        loss = actor_loss+critic_loss

        if i % 50 == 0:
            run.log({"actor_loss": actor_loss.item(),
                     "critic_loss": critic_loss.item(),
                     "advantage": advantages.mean().item(),
                     "loss": loss.item(),
                     "reward": rewards[0].item(),
                     "sample": wandb.Html(logits_string[0])})

        # Backprop!
        loss.backward()

        critic_optim.step()
        critic_optim.zero_grad()

        actor_optim.step()
        actor_optim.zero_grad()
        # optim = AdamW(bart_model.parameters(), lr=config.learning_rate)
