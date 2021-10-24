# Ahahhaha scratch.py is back with another installment of ML frickery!
# get ready to be fricked by ML!

from sentence_transformers import SentenceTransformer, util

# Sentence similarity things
model = SentenceTransformer('stsb-bert-base')

a,b,c = model.encode(["Hello, my name is a chicken.",
                      "A chicken is my name.",
                      "His face is a sam."])

similarityA = util.pytorch_cos_sim(a,b)
similarityB = util.pytorch_cos_sim(a,c)
similarityC = util.pytorch_cos_sim(c,c)
similarityA
similarityB
similarityC

# Simplicity evaluation

