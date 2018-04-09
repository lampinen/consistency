
from __future__ import division 

import sys
import time

import numpy as np
import skimage.io as io
import skimage.transform as transform
import skimage.color as color

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
    "word_embedding_dim": 256,
    "shared_word_embeddings": True, # whether input and output embeddings are the same or different
                                    # weird things will happen in the consistency training if this isn't true
    "hidden_size": 512,
    "rnn_num_layers": 4,
    "num_epochs": 200,
    "test_every": 1,
    "output_every": 10,
    "init_lr": 0.0005,
    "lr_decay": 0.85,
    "lr_decays_every": 2,
    "loss_weights": {
        "caption_consistency_training_loss": 100., # chosen by heuristic of making losses about 
        "image_reconstruction_training_loss": 7.,  # the same magnitude
        "own_caption_consistency_loss": 100.,
        "image_reconstruction_consistency_loss": 7.
    },
    "keep_prob": 1.0, # dropout keep probability
    "freeze_cnn": False, # whether to freeze CNN weights or train them
    "batch_size": 20 
}

##

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

def get_examples(n=None, coco=coco):
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

        self.optimizer = tf.train.RMSPropOptimizer(self.lr_ph) 
        
        # perception
        def _build_perception_network(visual_input):
            with tf.variable_scope('perception'):
                with arg_scope(inception.inception_v3_arg_scope()):
                    inception_features, _ = inception.inception_v3_base(visual_input)
                net = slim.layers.avg_pool2d(inception_features, [8, 8])
                if config["freeze_cnn"]:
                    net = tf.stop_gradient(net)
                net = slim.flatten(net)
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

                    emb_output = tf.stack([start_token for i in range(config['batch_size'])], axis=0)

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
            self.caption_train = self.optimizer.minimize(self.caption_loss)
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
                net = slim.dropout(net, keep_prob=keep_ph)
                net = slim.layers.fully_connected(net, 1, activation_fn=tf.nn.sigmoid)
                return net


            self.caption_rep = _build_caption_reading_net(tf.reverse(self.caption_ph, [-1]), reuse=False, keep_ph=self.keep_ph)
            self.caption2_rep = _build_caption_reading_net(tf.reverse(self.caption2_ph, [-1]), reuse=True, keep_ph=self.keep_ph)
            self.neg_caption_rep = _build_caption_reading_net(tf.reverse(self.neg_caption_ph, [-1]), reuse=True, keep_ph=self.keep_ph)
            self.own_caption_rep = _build_caption_processing_net(tf.reverse(own_caption_embs, [-1]), reuse=True, keep_ph=self.keep_ph)

            self.caption_consistency_training_loss = tf.square(1-_build_caption_consistency_net(self.caption_rep, self.caption2_rep, False, self.keep_ph))
            self.caption_consistency_training_loss += tf.square(_build_caption_consistency_net(self.caption_rep, self.neg_caption_rep, False, self.keep_ph))

            self.own_caption_consistency_loss = tf.square(1-_build_caption_consistency_net(self.caption2_rep, self.own_caption_rep, False, self.keep_ph))

            # images from captions
            def _build_caption_to_image_rep_net(caption_rep, reuse=True, keep_ph=None):
                """Tries to reconstruct image rep from caption"""
                net = caption_rep
                net = slim.layers.fully_connected(net, config["hidden_size"], activation_fn=tf.nn.relu)
                net = slim.dropout(net, keep_prob=keep_ph)
                net = slim.layers.fully_connected(net, config["hidden_size"], activation_fn=tf.nn.relu)
                net = slim.dropout(net, keep_prob=keep_ph)
                net = slim.layers.fully_connected(net, config["hidden_size"], activation_fn=tf.nn.relu)
                return net

            self.reconstructed_image_rep = _build_caption_to_image_rep_net(self.caption_rep, reuse=False, keep_ph=self.keep_ph)
            self.image_reconstruction_training_loss = tf.nn.l2_loss(self.reconstructed_image_rep - self.image_rep)

            self.own_caption_reconstructed_image_rep = _build_caption_to_image_rep_net(self.own_caption_rep, reuse=True, keep_ph=self.keep_ph)
            self.image_reconstruction_consistency_loss = tf.nn.l2_loss(self.own_caption_reconstructed_image_rep - self.image_rep)

        # aggregated losses and training
        self.main_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "basic_net/") 
        self.other_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "consistency_evaluation/") 
#        print(self.main_train_vars)
#        print(self.other_train_vars)
        self.caption_train = self.optimizer.minimize(self.caption_loss, var_list=self.main_train_vars)

        # loss for training consistency and reconstruction nets
        self.train_c_agg_loss = config["loss_weights"]["caption_consistency_training_loss"] * self.caption_consistency_training_loss + config["loss_weights"]["image_reconstruction_training_loss"] * self.image_reconstruction_training_loss
        # loss for using them to improve captioning
        self.use_c_agg_loss = self.caption_loss + config["loss_weights"]["own_caption_consistency_loss"] * self.own_caption_consistency_loss + config["loss_weights"]["image_reconstruction_consistency_loss"] * self.image_reconstruction_consistency_loss 

        # for outputting only
        self.all_losses = [self.caption_loss, self.caption_consistency_training_loss, self.image_reconstruction_training_loss, self.own_caption_consistency_loss, self.image_reconstruction_consistency_loss]

        self.consistency_train = self.optimizer.minimize(self.train_c_agg_loss, var_list=self.other_train_vars)
        self.caption_train_with_consistency = self.optimizer.minimize(self.use_c_agg_loss, var_list=self.main_train_vars)

        # for convenience
        self.full_train = [self.consistency_train, self.caption_train_with_consistency]

        #
        _initialize_stuff()
 

    def _examples_to_feeddict(self, examples, negative_examples=None):
        images = []
        caption1s = []
        mask1s = []
        caption2s = []
        mask2s = []
        for example in examples:
            img = io.imread(example["image_name"]) 
            if len(np.shape(img)) == 2: # grayscale
                img = color.gray2rgb(img) 
            img = resize_image(rescale_image(img))
            images.append(img)
            captions = example["captions"] 
            np.random.shuffle(captions)
            caption1s.append(captions[0]["caption"])
            mask1s.append(captions[0]["mask"][0])
            caption2s.append(captions[1]["caption"])
            mask2s.append(captions[1]["mask"][0])

        feed_dict = {
             self.image_ph: np.array(images), 
             self.caption_ph: np.array(caption1s), 
             self.mask_ph: np.array(mask1s), 
             self.caption2_ph: np.array(caption2s), 
             self.mask2_ph: np.array(mask2s), 
        }
        if negative_examples is not None: # use a random caption from other as neg
            negative_captions = []
            for negative_example in negative_examples:
                negative_captions.append(
                    negative_example["captions"][np.random.randint(5)]["caption"])
            feed_dict[self.neg_caption_ph] = np.array(negative_captions) 
            
        return feed_dict
 

    def run_train_examples(self, examples, basic=False, negative_examples=None):
        feed_dict = self._examples_to_feeddict(examples, negative_examples=negative_examples)
        feed_dict[self.keep_ph] = config["keep_prob"] 
        feed_dict[self.lr_ph] = self.curr_lr 
        if basic:
            self.sess.run(self.caption_train, feed_dict=feed_dict)
        else:
            self.sess.run(self.full_train, feed_dict=feed_dict)


    def run_test_examples(self, examples, return_loss=True, return_words=False, negative_examples=None, return_all_losses=False):
        """Runs an example, returns loss and optionally the words the model outputs or returns all losses"""
        feed_dict = self._examples_to_feeddict(examples, negative_examples)
        feed_dict[self.keep_ph] = 1. 
        to_run = {}

        if return_loss:
            to_run["loss"] = self.caption_loss
        if return_words:
            to_run["words"] = self.caption_hardmax
        if return_all_losses:
            to_run["all_losses"] = self.all_losses

        res = self.sess.run(to_run, feed_dict=feed_dict)
        if return_words:
            res["words"] = [indices_to_words(res["words"][i], backward_vocab) for i in range(config['batch_size'])]

        return res


    def eval(self, test_dataset):
        """Runs test dataset, returns loss"""
        loss = 0.
        batch_size = config["batch_size"]
        num_batches = len(test_dataset)//batch_size
        for i in range(num_batches): 
            examples = [test_dataset[j] for j in range(i*batch_size, (i+1)*batch_size)]
            loss += np.sum(self.run_test_examples(examples)["loss"])
        return loss/len(test_dataset)


    def output_results(self, test_dataset, output_path=None):
        batch_size = config["batch_size"]
        num_batches = len(test_dataset)//batch_size
        if output_path is not None:
            results = []
        for i in range(num_batches): 
            examples = [test_dataset[j] for j in range(i*batch_size, (i+1)*batch_size)]
            res = self.run_test_examples(examples, return_loss=False, return_words=True)
            captions = res["words"]
            captions = [words_to_string(caption) for caption in captions]
            these_results = [{"image_id": examples[j]["id"], "caption": captions[j]} for j in range(batch_size)]
            if output_path is None:
                print(these_results)
            else:
                results += these_results 

        if output_path is not None:
            with open(output_path, "w") as fout:
                fout.write(results.__repr__())
        

    def train(self, dataset, nepochs=1000, test_dataset=None, logfile_path=None, output_path=None):
        batch_size = config['batch_size']
        if logfile_path is not None:
            with open(logfile_path, "w") as fout:
                fout.write("epoch, loss\n")
        num_batches = len(dataset)//batch_size
        last_batch_i = num_batches - 1
        start_time = time.time()
        for epoch in range(nepochs):
            order = np.random.permutation(len(dataset))
            for i in range(num_batches): 
                examples = [dataset[j] for j in order[i*batch_size:(i+1)*batch_size]]
                if i < last_batch_i:
                    negative_examples = [dataset[j] for j in order[(i+1)*batch_size:(i+2)*batch_size]]
                else:
                    negative_examples = [dataset[j] for j in order[:batch_size]]
                self.run_train_examples(examples, negative_examples)
                
            if epoch % config["test_every"] == 0 and test_dataset is not None: 
                curr_loss = self.eval(test_dataset)
                print("epoch: %i, current loss: %f, elapsed_time: %.1f" %(epoch, curr_loss, time.time() - start_time))
                if logfile_path is not None:
                    with open(logfile_path, "a") as fout:
                        fout.write("%i, %f\n" %(epoch, curr_loss))

            if epoch % config["output_every"] == 0 and test_dataset is not None: 
                if output_path is not None:
                    curr_output_path = output_path + "-%i" % epoch
                    print("Saving results to %s..." % curr_output_path)
                    self.output_results(test_dataset, curr_output_path)
                    print("Done.")

            if epoch % config["lr_decays_every"] == 0:
                self.curr_lr *= config["lr_decay"]



# Run some stuff!
np.random.seed(0)
tf.set_random_seed(0)

model = captioning_model()
#train_data = get_examples(10)
#print(model.run_test_examples([train_data[0], train_data[1]], True, True, [train_data[2], train_data[3]], True))
##print(model.run_test_examples(train_data[1], True, True, train_data[0], True))
##print(model.run_test_examples(train_data[2], True, True, train_data[3], True))
##print(model.run_test_examples(train_data[4], True, True, train_data[5], True))
##print(model.run_test_examples(train_data[6], True, True, train_data[7], True))
#model.run_train_examples([train_data[0],train_data[1]], [train_data[2],train_data[3]])
#print(model.run_test_examples([train_data[0], train_data[1]], True, True, [train_data[2], train_data[3]], True))
#exit()
#



train_data = get_examples()
print(len(train_data))
test_data = get_examples(n=10000, coco=coco_val) 
#print(test_data[0])
print("Initial loss: %f" % model.eval(test_data))
model.train(train_data, nepochs=config["num_epochs"], test_dataset=test_data, logfile_path="./results/log.csv", output_path="./results/outputs.json")
print("Final loss: %f" % model.eval(test_data))

