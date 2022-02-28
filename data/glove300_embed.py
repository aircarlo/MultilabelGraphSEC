import numpy as np
import json
import pandas as pd

# download GloVe pre-trained embedding (_Wikipedia 2014_ + _Gigaword 5_ (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download) with
# wget -nc -O data.zip "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip" and unzip it

# Parse txt: read file and create a dictionary with words as keys and embedding vectors as values
embed_dict = {}
with open('glove.6B.300d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], 'float32')
        embed_dict[word] = vector

# parse FSD50K labels
with open('FSD50K_lbl_map.json', 'r') as f:
    lbl = json.load(f)

# to explore:
# pd.DataFrame(list(lbl.items()), columns=['label', 'label_ID'])
lbl_list = list(lbl.keys())

# Get embeddings: to avoid encoding misbehaviors some common words are removed from the label description, as well as the text between parentheses.
# Then labels with multiple words are encoded averaging the single embeddings.
emb_tot = []
for l1 in range(200):
    emb = 0
    lbl = lbl_list[l1].lower().split('_')
    lbl = [x for x in lbl if x not in ['and', 'from', 'or', 'by']]  # remove common words
    lbl = [x for x in lbl if x[0] != '(']  # remove words between ()
    lbl = [x for x in lbl if x[-1] != ')']
    for l2 in range(len(lbl)):
        try:
            emb += embed_dict[lbl[l2]]
        except:
            print(f'Not found: {l1}/{l2} - {lbl[l2]}')

    print(f'Class {l1}, {len(lbl)} word(s) embedded')
    emb_tot.append(emb / len(lbl))  # average

emb_array = np.vstack(emb_tot)
np.save('GloVe_300_embedding.npy', emb_array)
print('done')
