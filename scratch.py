# Ahahhaha scratch.py is back with another installment of ML frickery!
# get ready to be fricked by ML!

from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import word_tokenize
from collections import defaultdict

from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import inaugural



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

# Sentence similarity things
model = SentenceTransformer('stsb-bert-base')

reward("a sentence, indeed, is this one", "this is a sentence", 0.5, model)

##############################
### zach you converted me ####
##############################

detokenizer = TreebankWordDetokenizer()
dataset = [detokenizer.detokenize(i) for i in inaugural.sents()]
test[1s2]


