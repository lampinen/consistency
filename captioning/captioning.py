
from __future__ import division 

import sys
import numpy as np
import skimage.io as io
import skimage.transform as transform

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import inception
from tensorflow.contrib.framework import arg_scope

from pycocotools.coco import COCO

from util import *

## config
config = {
    "no_consistency": False,
    "coco_data_dir": "/home/lampinen/Documents/data/coco/",
    "vision_checkpoint_location": "./inception_v3.ckpt",
    "vocabulary_filename": "./vocabulary.csv",
    "coco_data_type": "train2014",
    "coco_val_data_type": "val2014",
    "image_width": 299,
    "sequence_length": 30,
    "word_embedding_dim": 128,
    "shared_word_embeddings": True, # whether input and output embeddings are the same or different
                                    # weird things will happen in the consistency training if this isn't true
    "hidden_size": 256,
    "rnn_num_layers": 3,
    "full_train_every": 1, # a full training example is given once every _ training examples
    "init_lr": 0.001,
    "lr_decay": 0.85,
    "lr_decays_every": 50,
    "loss_weights": {
    },
    "test_every_k": 5,
    "keep_prob": 0.5, # dropout keep probability
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

vocab, backward_vocab, vocab_size = load_vocabulary_to_index(config["vocabulary_filename"])

########## Data processing ####################################################

def rescale_image(image):
    """Scales from [0, 255] to [-1, 1] for inception."""
    return 2. * (image/255.) - 1

def resize_image(image):
    """Resizes to [299, 299] for inception"""
    return transform.resize(image, [config["image_width"], config["image_width"]])

def get_examples(n=None):
    """Gets n examples from coco data, or all if n is None"""
    data = []
    images = coco.getImgIds()
    if n is not None:
        images = images[:n]
    images = coco.loadImgs(images) 
    
    for img in images:
        img_ann_ids = coco.getAnnIds(imgIds=img["id"])
        img_anns = coco.loadAnns(img_ann_ids)
        captions =  [pad_or_trim(caption_to_words(x["caption"]), config["sequence_length"]) for x in img_anns]
        captions = [{"caption": words_to_indices(x["words"], vocab), "mask": x["mask"]} for x in captions]
        this_datum = {
            "image_name": "{}/images/{}".format(config["coco_data_dir"],
                                                img["file_name"]),
            "id": img["id"],
            "captions": captions 
        }
        data.append(this_datum)
    return data


    

########## Model building #####################################################

class captioning_model(object):
    def __init__(self, no_consistency=False):
        self.curr_lr = config["init_lr"]
        self.image_ph = tf.placeholder(tf.float32, [config["batch_size"], config["image_width"], config["image_width"], 3]) 
        self.caption_ph = tf.placeholder(tf.int32, [config["batch_size"], config["sequence_length"]]) 
        self.mask_ph =  tf.placeholder(tf.bool, [config["batch_size"], config["sequence_length"]]) 
        self.caption2_ph = tf.placeholder(tf.int32, [config["batch_size"], config["sequence_length"]]) 
        self.mask2_ph =  tf.placeholder(tf.bool, [config["batch_size"], config["sequence_length"]]) 
        self.neg_caption_ph = tf.placeholder(tf.int32, [config["batch_size"], config["sequence_length"]]) 
        self.lr_ph = tf.placeholder(tf.float32)
        self.keep_ph = tf.placeholder(tf.float32)

        self.optimizer = tf.train.AdamOptimizer(self.lr_ph) 
        
        # perception
        def _build_perception_network(visual_input):
            with tf.variable_scope('perception'):
                with arg_scope(inception.inception_v3_arg_scope()):
                    inception_features, _ = inception.inception_v3_base(visual_input)
                net = slim.layers.avg_pool2d(inception_features, [8, 8])
                net = slim.flatten(tf.stop_gradient(net))
                net = slim.layers.fully_connected(net, config["hidden_size"], activation_fn=tf.nn.relu)
                return net

        self.image_rep = _build_perception_network(self.image_ph)

        # captioning
        with tf.variable_scope("basic_net"):
            embedding_size = config['word_embedding_dim']
            input_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                                                             -0.1/embedding_size, 0.1/embedding_size))
            if config["shared_word_embeddings"]:
                output_embeddings = input_embeddings
            else:
                output_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                                                                  -0.1/embedding_size, 0.1/embedding_size))

            def _build_captioning_net(image_rep, reuse=True, keep_ph=None):
                """Creates caption from visual embedding"""
                with tf.variable_scope('captioning', reuse=reuse):
                    if keep_ph is not None:
                        cell = lambda: tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(config['hidden_size']), output_keep_prob=keep_ph)
                    else:
                        cell = lambda: tf.contrib.rnn.BasicLSTMCell(config['hidden_size'])

                    stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config['rnn_num_layers'])])
                    start_token = tf.nn.embedding_lookup(output_embeddings, vocab["<START>"])
                    word_logits = []
                    emb_outputs = []

                    state = stacked_cell.zero_state(config['batch_size'], tf.float32)
                    state = tuple([tf.contrib.rnn.LSTMStateTuple(image_rep, state[0][1])] + [state[i] for i in range(1, len(state))])

                    emb_output = tf.reshape(start_token, [config['batch_size'], -1])

                    with tf.variable_scope("recurrence", reuse=reuse):
                        output_to_emb_output = tf.get_variable(
                            "output_to_emb_output",
                            [config['hidden_size'], config['word_embedding_dim']],
                            tf.float32)
                        for step in range(config['sequence_length']):
                            (output, state) = stacked_cell(emb_output, state)
                            emb_output = tf.matmul(output, output_to_emb_output)
                            emb_outputs.append(emb_output)
                            this_word_logits = tf.matmul(emb_output, tf.transpose(output_embeddings))
                            word_logits.append(this_word_logits)
                            tf.get_variable_scope().reuse_variables()

                    word_logits = tf.stack(word_logits, axis=1)
                    emb_outputs = tf.stack(emb_outputs, axis=1)
                    return word_logits, emb_outputs

            self.caption_logits, own_caption_embs = _build_captioning_net(self.image_rep, reuse=False, keep_ph=self.keep_ph) 
            self.caption_hardmax = tf.argmax(self.caption_logits, axis=-1)

            masked_logits = tf.boolean_mask(self.caption_logits, self.mask_ph)

            masked_labels = tf.boolean_mask(tf.one_hot(self.caption_ph, depth=vocab_size), self.mask_ph)

            self.caption_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=masked_logits, labels=masked_labels)
            self.caption_loss = tf.reduce_sum(self.caption_loss)
            self.caption_train = self.optimizer.minimize(self.caption_loss)

        def _initialize_stuff():
            # init, etc.
            # set to initialize vision network from checkpoint
            tf.contrib.framework.init_from_checkpoint(
                config['vision_checkpoint_location'],
                {'InceptionV3/': 'perception/InceptionV3/'})

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=sess_config) 

            self.sess.run(tf.global_variables_initializer())

        if config["no_consistency"]:
            _initialize_stuff()

        # caption reading and caption consistent net
        with tf.variable_scope("consistency_evaluation"):

            def _build_caption_processing_net(embedded_input, reuse=True, keep_ph=None):
                """Processes caption from char embeddings"""
                with tf.variable_scope('caption/reading', reuse=reuse):
                    if keep_ph is not None:
                        cell = lambda: tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(config['hidden_size']), output_keep_prob=keep_ph)
                    else:
                        cell = lambda: tf.contrib.rnn.BasicLSTMCell(config['hidden_size'])
                    stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config['rnn_num_layers'])])

                    state = stacked_cell.zero_state(config['batch_size'], tf.float32)
                    with tf.variable_scope("recurrence", reuse=reuse):
                        for step in range(config['sequence_length']):
                            (output, state) = stacked_cell(embedded_input[:, step, :], state)
                            tf.get_variable_scope().reuse_variables()

                return output

            def _build_caption_reading_net(caption_input, reuse=True, keep_ph=None):
                """Reads caption and processes"""
                with tf.variable_scope('caption/reading', reuse=reuse):
                    embedded_input = tf.nn.embedding_lookup(input_embeddings, caption_input)
                output = _build_caption_processing_net(embedded_input, reuse=reuse, keep_ph=keep_ph)
                return output

            def _build_caption_consistency_net(caption_1_rep, caption_2_rep, reuse=True, keep_ph=None):
                """Evaluates whether captions represent the same image"""
                net = slim.layers.fully_connected(tf.concat([caption_1_rep, caption_2_rep], -1), config["hidden_size"], activation_fn=tf.nn.relu)
                net = slim.layers.fully_connected(net, 1, activation_fn=tf.nn.sigmoid)
                return net


            self.caption_rep = _build_caption_reading_net(tf.reverse(self.caption_ph, [-1]), reuse=False, keep_ph=self.keep_ph)
            self.caption2_rep = _build_caption_reading_net(tf.reverse(self.caption2_ph, [-1]), reuse=True, keep_ph=self.keep_ph)
            self.neg_caption_rep = _build_caption_reading_net(tf.reverse(self.neg_caption_ph, [-1]), reuse=True, keep_ph=self.keep_ph)
            self.own_caption_rep = _build_caption_processing_net(tf.reverse(own_caption_embs, [-1]), reuse=True, keep_ph=self.keep_ph)

            self.caption_consistency_loss = tf.square(1-_build_caption_consistency_net(self.caption_rep, self.caption2_rep, False, self.keep_ph))
            self.caption_consistency_loss += tf.square(_build_caption_consistency_net(self.caption_rep, self.neg_caption_rep, False, self.keep_ph))

            self.own_caption_consistency_loss = tf.square(1-_build_caption_consistency_net(self.caption2_rep, self.own_caption_rep, False, self.keep_ph))

        # images from captions


        self.main_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "basic_net/") 
        self.other_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "consistency_evaluation/") 
#        print(self.main_train_vars)
#        print(self.other_train_vars)
        _initialize_stuff()
 

    def _example_to_feeddict(self, example, negative_example=None):
        img = io.imread(example["image_name"]) 
        img = resize_image(rescale_image(img))
        captions = example["captions"] 
        np.random.shuffle(captions)
        feed_dict = {
             self.image_ph: np.expand_dims(img, axis=0), 
             self.caption_ph: np.expand_dims(captions[0]["caption"], 0),
             self.mask_ph: captions[0]["mask"],
             self.caption2_ph: np.expand_dims(captions[1]["caption"], 0),
             self.mask2_ph: captions[1]["mask"]
        }
        if negative_example is not None: # use a random caption from other as neg
            feed_dict[self.neg_caption_ph]= np.expand_dims(
                negative_example["captions"][np.random.randint(5)]["caption"], 0),
            
        return feed_dict

    def run_train_example(self, example, basic=False, negative_example=None):
        feed_dict = self._example_to_feeddict(example)
        feed_dict[self.keep_ph] = config["keep_prob"] 
        feed_dict[self.lr_ph] = self.curr_lr 
        self.sess.run(self.caption_train, feed_dict=feed_dict)


    def run_test_example(self, example, return_loss=True, return_words=False):
        """Runs an example, returns loss and optionally the words the model outputs"""
        feed_dict = self._example_to_feeddict(example)
        feed_dict[self.keep_ph] = 1. 
        res =  {}
        if return_words:
            if return_loss:
                loss, indices = self.sess.run([self.caption_loss, self.caption_hardmax], feed_dict=feed_dict)
                res["loss"] = loss
            else: 
                indices = self.sess.run(self.caption_hardmax, feed_dict=feed_dict)
            res["words"] = indices_to_words(indices[0], backward_vocab)
        else:
            res["loss"] = self.sess.run([self.caption_loss], feed_dict=feed_dict)

        return res







# Run some stuff!
model = captioning_model()
train_data = get_examples(10)
print(model.run_test_example(train_data[0], True, True))
model.run_train_example(train_data[0])
print(model.run_test_example(train_data[0], True, True))


