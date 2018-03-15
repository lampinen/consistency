from __future__ import print_function
from __future__ import division 

import sys
from collections import Counter

from pycocotools.coco import COCO

from util import caption_to_words

## config
config = {
    "coco_data_dir": "/home/lampinen/Documents/data/coco/",
    "coco_data_type": "train2014",
    "vocab_appearance_cutoff": 5 # words that appear less than this many times will not be included in vocab
}

##


########## Data ###############################################################
ann_filename = "{}/annotations/captions_{}.json".format(config["coco_data_dir"],
                                                        config["coco_data_type"])

coco = COCO(ann_filename)

########## Data processing ####################################################


def get_vocab():
    """gets vocabulary from train examples"""
    vocab = Counter()
    lengths = Counter()
    images = coco.loadImgs(coco.getImgIds()) 
    
    for img in images:
        
        img_ann_ids = coco.getAnnIds(imgIds=img["id"])
        img_anns = coco.loadAnns(img_ann_ids)
        for ann in img_anns:
            words = caption_to_words(ann['caption'])
            lengths.update([len(words)])

            vocab.update(words)
    print(lengths.most_common())
    return vocab

vocabulary = get_vocab()
print(len(vocabulary))
print(len([k for (k, v) in vocabulary.most_common() if v < 5]))
print(len([k for (k, v) in vocabulary.most_common() if v == 1]))
print(vocabulary.most_common(10))

print("saving vocab with cutoff = %i" % config["vocab_appearance_cutoff"])

vocab_appearance_cutoff = config["vocab_appearance_cutoff"]
with open("vocabulary.csv", "w") as v_file:
    v_file.write("<UNK>\n")
    v_file.write("<PAD>\n")
    v_file.write("<START>\n")
    for word, count in vocabulary.most_common():
        if count < vocab_appearance_cutoff:
            break
        v_file.write(word+"\n")
