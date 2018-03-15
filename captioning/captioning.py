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

from util import *

## config
config = {
    "no_consistency": False,
    "coco_data_dir": "/home/lampinen/Documents/data/coco/",
    "vocabulary_filename": "./vocabulary.csv",
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

vocab, vocab_size = load_vocabulary_to_index(config["vocabulary_filename"])

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
        img_ann_ids = coco.getAnnIds(imgIds=img["id"])
        img_anns = coco.loadAnns(img_ann_ids)
        this_datum = {
            "image_name": "{}/images/{}".format(config["coco_data_dir"],
                                                img["file_name"]),
            "id": img["id"],
            "captions": [words_to_indices(pad_or_trim(caption_to_words(x["caption"]), config["seq_length"]), vocab) for x in img_anns] 
        }
        data.append(this_datum)
    return data

train_data = get_examples()

    

########## Model building #####################################################

class captioning_model(object):
    def __init__(self, no_consistency=False):
        self.image_ph = tf.placeholder(tf.float32, [config["batch_size"], config["image_width"], config["image_width"], 3]) 
        self.caption_ph = tf.placeholder(tf.int32, [config["batch_size"], config["sequence_length"]]) 
        self.caption2_ph = tf.placeholder(tf.int32, [config["batch_size"], config["sequence_length"]]) 
        
        # perception
        def _build_perception_network(visual_input)
            with tf.variable_scope('perception'):
                with arg_scope(inception.inception_v3_arg_scope(use_fused_batchnorm=False)):
                    inception_features, _ = inception.inception_v3_base(visual_input)

        self.image_rep = _build_perception_network(self.image_ph)


        # captioning
	embedding_size = config['word_embedding_dim']
	input_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
							 -0.1/embedding_size, 0.1/embedding_size))
	output_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
							  -0.1/embedding_size, 0.1/embedding_size))

        def build_captioning_net(image_rep, reuse=True, keep_ph=None):
            """Solves problem from problem embedding"""
            with tf.variable_scope('captioning', reuse=reuse):
                if keep_ph is not None:
                    cell = lambda: tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim']), output_keep_prob=keep_ph)
                else:
                    cell = lambda: tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim'])

                stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config['rnn_num_layers'])])
                start_token = tf.nn.embedding_lookup(output_embeddings, vocab_dict["<START>"])
                char_logits = []

                state = stacked_cell.zero_state(config['batch_size'], tf.float32)
                state = tuple([tf.contrib.rnn.LSTMStateTuple(image_rep, state[0][1])] + [state[i] for i in range(1, len(state))])

                emb_output = tf.reshape(start_token, [config['batch_size'], -1])

                with tf.variable_scope("recurrence", reuse=reuse):
                    output_to_emb_output = tf.get_variable(
                        "output_to_emb_output",
                        [config['image_embedding_dim'], config['word_embedding_dim']],
                        tf.float32)
                    for step in range(config['seq_length']):
                        (output, state) = stacked_cell(emb_output, state)
                        emb_output = tf.matmul(output, output_to_emb_output)
                        this_char_logits = tf.matmul(emb_output, tf.transpose(output_embeddings))
                        char_logits.append(this_char_logits)
                        tf.get_variable_scope().reuse_variables()

                char_logits = tf.stack(char_logits, axis=1)
                return char_logits



        # TODO: this for all perception
        # set to initialize vision network from checkpoint
        tf.contrib.framework.init_from_checkpoint(
            model_config['vision_checkpoint_location'],
            {'InceptionV3/': 'perception/InceptionV3/'})
 

    def __example_to_feeddict(self, example):
        img = io.imread(example["image_name"]) 
        img = resize_image(rescale_image(img))
        captions = example["captions"] 
        np.random.shuffle(captions)
        feed_dict = {
             self.image_ph: img, 
             self.caption_ph: captions[0],
             self.caption2_ph: captions[1]
        }
        return feed_dict

    def run_train_exemplar(self, exemplar):
       pass 
