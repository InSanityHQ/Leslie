# Ahahhaha scratch.py is back with another installment of ML frickery!
# get ready to be fricked by ML!

from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from collections import defaultdict

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import brown

from transformers import BartConfig, BartTokenizer, BartForConditionalGeneration
from transformers import AdamW

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import statistics

import numpy as np
import re

from tqdm import tqdm
import math

import wandb

print("Hello and welcome. Let's get started.")

# Parametres

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

hyperparametre_defaults = dict(
    max_length = 50,
    actor_lr = 5e-5,
    batch_size = 16,
    epochs = 1000,
    replay_buffer = 50
)

run = wandb.init(entity='inscriptio', project='leslie', config=hyperparametre_defaults)
# run = wandb.init(entity='inscriptio', project='leslie', config=hyperparametre_defaults, mode="disabled")
config = wandb.config

MAX_LENGTH = config.max_length
ACTOR_LR = config.actor_lr
# CRITIC_LR = config.critic_lr
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
REPLAY = config.replay_buffer

print("Setting up dataset and reward.")
detokenizer = TreebankWordDetokenizer()
dataset_sents = [detokenizer.detokenize(i) for i in brown.sents()]
dataset_words = [i.lower() for i in brown.words()]
usage = defaultdict(int)

# We count the usage of each word
for word in dataset_words:
    usage[word] += 1

# We get the mean and stddev usage and normalize the
# usages by them
usage_mean = statistics.mean(usage.values())
usage_stddev = statistics.stdev(usage.values())

# Finally, we normalize every value based on this
# difference. We encourage results to be higher
# than mean so we don't abs value. Also we will
# take the tanh of the output to normalize it
# between -1 and 1

for key in usage.keys():
    usage[key] = np.tanh((usage[key]-usage_mean)/usage_stddev)

# Find out the sentence similarity between two sentences
# using a model.
def semantic_similarity(a,b,model):
    a,b = model.encode([a,b])
    return util.pytorch_cos_sim(a,b)[0][0].item()

# Mixed simplification reward 
def reward(src,tgt,model):
    words_src = [i.lower() for i in word_tokenize(src)]
    words_tgt = [i.lower() for i in word_tokenize(tgt)]

    try: 
        usage_src = sum([usage[i] for i in words_src])/len(words_src)
        usage_tgt = sum([usage[i] for i in words_tgt])/len(words_tgt)
        simplification_rating = ((usage_tgt-usage_src)/usage_src)
    except ZeroDivisionError:
        simplification_rating = -2

    scaled_similarity = semantic_similarity(src,tgt,model)
    similarity_rating = scaled_similarity

                                                      # rescale "doing nothing" to 0
    return (simplification_rating + similarity_rating)-1

print("Instantiating similarity model.")
similarity_model = SentenceTransformer('stsb-bert-base')

##############################
### zach you converted me ####
##############################

print("Establishing tokenizers and actor model.")

dataset_train = dataset_sents[:int(len(dataset_sents)*0.9)]

bart_config = BartConfig.from_pretrained("facebook/bart-base")
# TODO any max length stuff
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", config=bart_config)

np2tens = lambda x: torch.tensor(x)

print("Creating optimizers and moving models.")
bart_model.to(DEVICE)
bart_model.train()
actor_optim = AdamW(bart_model.parameters(), lr=ACTOR_LR)

run.watch(bart_model)

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

        # Get the logits
        logits = result["logits"]

        # Select for the predicted outputs
        actions = torch.stack([torch.argmax(i, axis=1) for i in logits.detach()])
        # Stringify the outputs
        logits_string = [bart_tokenizer.decode(i) for i in actions]
        # Return the final string
        logits_string = [re.sub("<s>", "", i.split("</s>")[0]) for i in logits_string]
        # Calculate reward value for these strings
        rewards = np2tens([reward(i,j,similarity_model) for i,j in zip(batch, logits_string)]).to(DEVICE)

        # # Calculate the advantages of the model
        # advantages = rewards -

        # Get the logits' probs by normalizing with softmax
        logits_probs = F.softmax(logits, dim=2)

        # Gather the log probability values of the selected actions
        action_log_probs = logits_probs.gather(2, torch.unsqueeze(actions, 2)).log()

        # Add em up, multiply by the advantage, and calculate the mean
        actor_loss = (torch.stack([a*l for a,l in zip(rewards,action_log_probs)]).mean())/REPLAY

        # The critic wants to match the actual rewards as much as possible
        # So it's just MSE loss 
        dataset_bar.set_description(f"Sample: {batch_id}, Actor: {round(actor_loss.item(),3)}, Reward: {round(rewards[0].item(),3)}")

        if i % 50 == 0:
            print(f"Sample: {batch_id}; Output: {logits_string[0]}")

        if i % 10 == 0:
            run.log({"actor_loss": actor_loss.item(),
                     # "critic_loss": critic_loss.item(),
                     # "advantage": advantages.mean().item(),
                     # "loss": loss.item(),
                     "reward": rewards[0].item(),
                     "sample": wandb.Html(logits_string[0])})

        # if we hit the replay buffer, then go ahead and update weights
        if i != 0 and i % REPLAY == 0:
            actor_optim.step()
            actor_optim.zero_grad()

        # Backprop!
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(bart_model.parameters(), 0.1)
