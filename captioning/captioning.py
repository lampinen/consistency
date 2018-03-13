from __future__ import print_function
from __future__ import division 

import sys
import numpy as np
import skimage.io as io
import skimage.transform as transform

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.framework import arg_scope

from pycocotools.coco import COCO

## config
config = {
    "no_consistency": False,
    "coco_data_dir": "/home/lampinen/Documents/data/coco/",
    "coco_data_type": "train2014",
    "coco_val_data_type": "val2014",
    "image_width": 299,
    "seq_length": 5,
    "word_embedding_dim": 128,
    "hidden_dim": 256,
    "rnn_num_layers": 3,
    "full_train_every": 1, # a full training example is given once every _ training examples
    "init_lr": 0.001,
    "lr_decay": 0.85,
    "lr_decays_every": 50,
    "loss_weights": {
    },
    "test_every_k": 5,
    "batch_size": 1 # batches larger than 1 are not supported, this is just to get rid of the "magic constant" feel where it has to be specified
}

##

np.random.seed(0)
tf.set_random_seed(0)

########## Data ###############################################################
ann_filename = "{}/annotations/captions_{}.json".format(config["coco_data_dir"],
                                                        config["coco_data_type"])
val_ann_filename = "{}/annotations/captions_{}.json".format(config["coco_data_dir"],
                                                            config["coco_val_data_type"])

coco = COCO(ann_filename)
coco_val = COCO(val_ann_filename)

## dummy example of how data loading works
#image_ids = coco.getImgIds()
#img_meta = coco.loadImgs(image_ids[0])[0]
#img = io.imread("{}/images/{}".format(config["coco_data_dir"],
#                                      img_meta["file_name"]))
#img_ann_ids = coco.getAnnIds(imgIds=img_meta["id"])
#img_anns = coco.loadAnns(img_ann_ids)

########## Data processing ####################################################

def rescale_image(image):
    """Scales from [0, 255] to [-1, 1] for inception."""
    return 2. * (image/255.) - 1

def resize_image(image):
    """Resizes to [299, 299] for inception"""
    return transform.resize(image, [config["image_width"], config["image_width"]])

def get_examples():
    data = []
    images = coco.loadImgs(coco.getImgIds()) 
    
    for img in images:
        img_ann_ids = coco.getAnnIds(imgIds=img_meta["id"])
        img_anns = coco.loadAnns(img_ann_ids)
        this_datum = {
            "image": img = io.imread("{}/images/{}".format(config["coco_data_dir"],
                                                           img_meta["file_name"])),
            "id": img_meta["id"],
            "captions": [x["caption"] for x in img_anns] 
        }
        data.append(this_datum)
    return data

train_data = get_examples()

########## Model building #####################################################

class captioning_model(object):
    def __init__(self, no_consistency=False):
        pass
