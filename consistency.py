from __future__ import print_function
from __future__ import division 

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

## config
config = {
    "seq_length": 5,
    "output_seq_length": 2,
    "char_embedding_dim": 20,
    "vision_embedding_dim": 100,
    "problem_embedding_dim": 100,
    "loss_weights": {
        "direct_solution_loss": 1.,
        "direct_visual_solution_loss": 1.,
        "reconstructed_solution_loss": 1.,
        "imagined_visual_solution_loss": 1.,

        "true_visual_problem_reconstruction_closs": 1.,
        "true_visual_visual_reconstruction_closs": 1.,
        "imagined_visual_problem_reconstruction_closs": 1.,
        "imagined_visual_visual_reconstruction_closs": 1.,

        "direct_solution_direct_visual_solution_closs": 1.,
        "direct_solution_imagined_visual_solution_closs": 1.,
        "reconstructed_solution_direct_visual_solution_closs": 1.
    },
    "batch_size": 1 # batches larger than 1 are not supported, this is just to get rid of the "magic constant" feel where it has to be specified
}

##

number_vocab = map(str, range(10))
op_vocab = ["+", "-", "*", "/"]
control_vocab = ["<PAD>", "<START>"] 
vocab = number_vocab + op_vocab + control_vocab 
vocab_dict = dict(zip(vocab, range(len(vocab)))) # index lookup

def text_to_indices(text):
    return [vocab_dict[char] for char in text]

def make_visual_array(n, m=None, op="+", dim=10):
    x = numpy.zeros((2, dim, dim))
    if m is not None:
        if op == "*":
            if m > dim or n > dim:
                raise ValueError("One of n or m is greater than dim")
            x[0, :m, :n] = 1.
        elif op == "-":
            if m > dim**2 or n > dim**2:
                raise ValueError("One of m or n is greater than dim**2")
            q, r = divmod(n, 10)
            x[0, :q, :] = 1.
            x[0, q, :r] = 1.
            q, r = divmod(m, 10)
            x[1, :q, :] = 1.
            x[1, q, :r] = 1.
        elif op == "+":
            q, r = divmod(n, 10)
            q2, r2 = divmod(m, 10)
            if q + q2 + 2 > dim:
                raise ValueError("m + n is too large")
            x[0, :q, :] = 1.
            x[0, q, :r] = 1.
            x[0, q+1:q+1+q2, :] = 1.
            x[0, q+q2+2, :r] = 1.
        elif op == "/":
            a, b = divmod(n, m)
            if b != 0 or a > dim or m > dim:
                raise ValueError("Uneven remainder, or one of n/m or m is greater than dim")
            x[0, :a, :m] = 1.
    else:
        if n > dim**2:
            raise ValueError("n is greater than dim**2")
        q, r = divmod(n, 10)
        x[0, :q, :] = 1.
        x[0, q, :r] = 1.
    return x

def right_pad_seq(seq, n=config['output_seq_length'], pad_symbol="<PAD>"):
    return seq + [pad_symbol] * (n- len(seq))

def left_pad_seq(seq, n=config['seq_length'], pad_symbol="<PAD>"):
    return [pad_symbol] * (n- len(seq)) + seq



class consistency_model(object):
    def __init__(self, no_consistency=False, no_visual=False):
        self.no_consistency = no_consistency
        self.no_visual = no_visual

        self.vocab_size = vocab_size = len(vocab)

        with tf.variable_scope('problem'):
            embedding_size = config['char_embedding_dim']
            input_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1/embedding_size, 0.1/embedding_size)) 
            output_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1/embedding_size, 0.1/embedding_size)) 

        def build_problem_processing_net(embedded_input, reuse=True):
            """Processes problem from char embeddings"""
            with tf.variable_scope('problem/reading', reuse=reuse):
                cell = tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim'])
            
                state = cell.zero_state(config['batch_size'], tf.float32)
                with tf.variable_scope("recurrence", reuse=reuse):
                    for step in xrange(config['seq_length']):
                        (output, state) = cell(embedded_input[:, step, :], state)
                        tf.get_variable_scope().reuse_variables()

            return output
        
        def build_problem_reading_net(problem_input, reuse=True):
            """Reads problem and processes"""
            with tf.variable_scope('problem/reading', reuse=reuse):
                embedded_input = tf.nn.embedding_lookup(input_embeddings, problem_input)
            output = build_problem_processing_net(embedded_input, reuse=reuse)
            return output

        def build_problem_solution_net(problem_embedding, reuse=True):
            """Solves problem from problem embedding"""
            with tf.variable_scope('problem/solution', reuse=reuse):

                cell = tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim'])
                start_token = tf.nn.embedding_lookup(output_embeddings, vocab_dict["<START>"])
                char_logits = []

                state = tf.contrib.rnn.LSTMStateTuple(problem_embedding, cell.zero_state(config['batch_size'], tf.float32)[1])
                emb_output = tf.reshape(start_token, [config['batch_size'], -1])

                with tf.variable_scope("recurrence", reuse=reuse):
                    output_to_emb_output = tf.get_variable(
                        "output_to_emb_output",
                        [config['problem_embedding_dim'], config['char_embedding_dim']],
                        tf.float32)
                    for step in xrange(config['output_seq_length']):
                        (output, state) = cell(emb_output, state)
                        emb_output = tf.matmul(output, output_to_emb_output) 
                        this_char_logits = tf.matmul(emb_output, tf.transpose(output_embeddings))
                        char_logits.append(this_char_logits)
                        tf.get_variable_scope().reuse_variables()

                char_logits = tf.stack(char_logits, axis=1)
                return char_logits 

        self.lr_ph  = tf.placeholder(tf.float32)
        self.problem_input_ph = tf.placeholder(tf.int32,
                                               [None, config['seq_length']])
        self.solution_input_ph = tf.placeholder(tf.int32,
                                           [None, config['output_seq_length']])
        ground_truth_solution = tf.one_hot(self.solution_input_ph,
                                           vocab_size) 

        problem_embedding = build_problem_reading_net(self.problem_input_ph,
                                                      reuse=False)
        direct_solution_logits = build_problem_solution_net(problem_embedding,
                                                            reuse=False)
        direct_solution_softmax = tf.nn.softmax(direct_solution_logits)

        self.direct_solution_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_solution, logits=direct_solution_logits) 

        optimizer = tf.train.AdamOptimizer(self.lr_ph)

        this_weight = config["loss_weights"]["direct_solution_loss"]
        self.direct_solution_train = optimizer.minimize(
            this_weight * self.direct_solution_loss)

        # Visual
        if no_visual:
            return

        def build_imagination_net(problem_embedding, reuse=True):
            """Generates "imaginary" visual array from a problem embedding"""
            with tf.variable_scope('imagination', reuse=reuse):
                # fully connected 
                net = problem_embedding
                net = slim.layers.fully_connected(net, 2 * 2 * 64, activation_fn=tf.nn.leaky_relu)
                net = tf.reshape(net, [-1, 2, 2, 64])

                # 2
                net = tf.image.resize_bilinear(net, [4, 4])
                net = slim.layers.conv2d_transpose(net, 32, [2, 2], stride=2)

                # 1
                net = tf.image.resize_bilinear(net, [10, 10])
                net = slim.layers.conv2d_transpose(net, 2, [3, 3], stride=1)
                return net


        def build_perception_net(perception_input, reuse=True): 
            """Generates perceptual embedding of visual array"""
            with tf.variable_scope('perception', reuse=reuse):
                # 1
                net = slim.layers.conv2d(perception_input, 32, [2, 2], stride=1)
                net = slim.layers.avg_pool2d(net, [3, 3], stride=1)
                # 2
                net = slim.layers.conv2d(net, 64, [2, 2], stride=2)
                net = slim.layers.avg_pool2d(net, [2, 2], stride=2)
                # fc
                net = slim.flatten(net)
                representation = slim.layers.fully_connected(net, config["vision_embedding_dim"], activation_fn=tf.nn.leaky_relu)
                return representation



        def build_perceptual_solution_net(vision_embedding, reuse=True):
            """Solves problem from visual embedding"""
            with tf.variable_scope('perception/solution', reuse=reuse):

                cell = tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim'])
                start_token = tf.nn.embedding_lookup(output_embeddings, vocab_dict["<START>"])
                char_logits = []

                state = tf.contrib.rnn.LSTMStateTuple(vision_embedding, cell.zero_state(config['batch_size'], tf.float32)[1])
                emb_output = tf.reshape(start_token, [config['batch_size'], -1])

                with tf.variable_scope("recurrence", reuse=reuse):
                    output_to_emb_output = tf.get_variable(
                        "output_to_emb_output",
                        [config['problem_embedding_dim'], config['char_embedding_dim']],
                        tf.float32)
                    for step in xrange(config['output_seq_length']):
                        (output, state) = cell(emb_output, state)
                        emb_output = tf.matmul(output, output_to_emb_output) 
                        this_char_logits = tf.matmul(emb_output, tf.transpose(output_embeddings))
                        char_logits.append(this_char_logits)
                        tf.get_variable_scope().reuse_variables()

                char_logits = tf.stack(char_logits, axis=1)
                return char_logits 


        # visual path
        self.vision_input_ph = tf.placeholder(tf.float32, [None, 10, 10, 2])
        visual_input_embedding = build_perception_net(self.vision_input_ph,
                                                      reuse=False)
        direct_visual_solution_logits = build_perceptual_solution_net(
            visual_input_embedding, reuse=False) 
        direct_visual_solution_softmax = tf.nn.softmax(
            direct_visual_solution_logits)

        self.direct_visual_solution_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_solution, logits=direct_solution_logits) 

        this_weight = config["loss_weights"]["direct_visual_solution_loss"]
        self.direct_visual_solution_train = optimizer.minimize(
            this_weight * self.direct_visual_solution_loss)

        # imagined path
        imagined_visual_scene = build_imagination_net(problem_embedding,
                                                      reuse=False) 
        imagined_visual_embedding = build_perception_net(imagined_visual_scene)
        imagined_visual_solution_logits = build_perceptual_solution_net(
            imagined_visual_embedding)

        imagined_visual_solution_softmax = tf.nn.softmax(
            imagined_visual_solution_logits)

        self.imagined_visual_solution_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_solution, logits=imagined_visual_solution_logits) 

        this_weight = config["loss_weights"]["imagined_visual_solution_loss"]
        self.imagined_visual_solution_train = optimizer.minimize(
            this_weight * self.imagined_visual_solution_loss)

        def build_perceptual_problem_reconstruction_net(vision_embedding, ground_truth, reuse=True):
            """Reconstructs problem from visual embedding"""
            with tf.variable_scope('perception/problem_reconstruction', reuse=reuse):

                cell = tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim'])
                start_token = tf.nn.embedding_lookup(output_embeddings, vocab_dict["<START>"])
                char_logits = []
                emb_outputs = []

                state = tf.contrib.rnn.LSTMStateTuple(vision_embedding, cell.zero_state(config['batch_size'], tf.float32)[1])
                emb_output = tf.reshape(start_token, [config['batch_size'], -1])

                with tf.variable_scope("recurrence", reuse=reuse):
                    output_to_emb_output = tf.get_variable(
                        "output_to_emb_output",
                        [config['problem_embedding_dim'], config['char_embedding_dim']],
                        tf.float32)
                    for step in xrange(config['seq_length']):
                        this_input = ground_truth[:, step-1, :]
                        (output, state) = cell(this_input, state)
                        emb_output = tf.matmul(output, output_to_emb_output) 
                        this_char_logits = tf.matmul(emb_output, tf.transpose(output_embeddings))
                        emb_outputs.append(emb_output)
                        char_logits.append(this_char_logits)
                        tf.get_variable_scope().reuse_variables()

                char_logits = tf.stack(char_logits, axis=1)
                emb_outputs = tf.stack(emb_outputs, axis=1)
                return char_logits, emb_outputs 

        # reconstructed problem path
        problem_input_embeddings = tf.nn.embedding_lookup(input_embeddings, self.problem_input_ph) # the choice to use output embeddings here reflects the fact that students might be verbally asked to produce such an answer, but a different choice could be made
        (true_visual_problem_reconstruction_logits,
         true_visual_problem_reconstruction) = build_perceptual_problem_reconstruction_net(
             visual_input_embedding, problem_input_embeddings, reuse=False)


        reconstructed_problem_embedding = build_problem_processing_net(
            true_visual_problem_reconstruction)
        reconstructed_direct_solution_logits = build_problem_solution_net(
            reconstructed_problem_embedding)
                                                            
        reconstructed_solution_softmax = tf.nn.softmax(direct_solution_logits)

        self.reconstructed_solution_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_solution, logits=reconstructed_direct_solution_logits) 

        this_weight = config["loss_weights"]["reconstructed_solution_loss"]
        self.reconstructed_solution_train = optimizer.minimize( 
            this_weight * self.reconstructed_solution_loss)


        # Consistency
        if no_consistency:
            return

        problem_input_onehot = tf.one_hot(self.problem_input_ph, vocab_size)

        # vision -> reconstructed problem = ground truth problem
        (true_visual_problem_reconstruction_logits,
         true_visual_problem_reconstruction) = build_perceptual_problem_reconstruction_net(
             visual_input_embedding, problem_input_embeddings)
        self.true_visual_problem_reconstruction_closs = tf.nn.softmax_cross_entropy_with_logits(
            labels=problem_input_onehot, logits=true_visual_problem_reconstruction_logits) 
        
        this_weight = config["loss_weights"]["true_visual_problem_reconstruction_closs"]
        self.true_visual_problem_reconstruction_ctrain = optimizer.minimize(
            this_weight * self.true_visual_problem_reconstruction_closs)
                
        # ground_truth_problem -> imagined visual -> reconstructed problem = ground truth problem
        (imagined_visual_problem_reconstruction_logits,
         imagined_visual_problem_reconstruction) = build_perceptual_problem_reconstruction_net(
             imagined_visual_embedding, problem_input_embeddings)
        self.imagined_visual_problem_reconstruction_closs = tf.nn.softmax_cross_entropy_with_logits(
            labels=problem_input_onehot, logits=imagined_visual_problem_reconstruction_logits) 

        this_weight = config["loss_weights"]["imagined_visual_problem_reconstruction_closs"]
        self.imagined_visual_problem_reconstruction_ctrain = optimizer.minimize(
            this_weight * self.imagined_visual_problem_reconstruction_closs)
        
        # ground truth problem -> imagined visual = true visual 
        self.imagined_visual_visual_reconstruction_closs = tf.nn.l2_loss(
            imagined_visual_scene - self.vision_input_ph)
        this_weight = config["loss_weights"]["imagined_visual_visual_reconstruction_closs"]
        self.imagined_visual_visual_reconstruction_ctrain = optimizer.minimize(
            this_weight * self.imagined_visual_visual_reconstruction_closs)

        # true visual -> reconstructed problem -> imagined visual = true visual 
        reconstructed_imagined_visual_scene = build_imagination_net(
            reconstructed_problem_embedding)
        self.true_visual_visual_reconstruction_closs = tf.nn.l2_loss(
            reconstructed_imagined_visual_scene - self.vision_input_ph)
        
        this_weight = config["loss_weights"]["true_visual_visual_reconstruction_closs"]
        self.true_visual_visual_reconstruction_ctrain = optimizer.minimize(
            this_weight * self.imagined_visual_visual_reconstruction_closs)

        # direct solution == direct_visual_solution
        self.direct_solution_direct_visual_solution_closs = 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=direct_solution_softmax, logits=direct_visual_solution_logits)  
        self.direct_solution_direct_visual_solution_closs += 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=direct_visual_solution_softmax, logits=direct_solution_logits)  

        this_weight = config["loss_weights"]["direct_solution_direct_visual_solution_closs"]
        self.direct_solution_direct_visual_solution_ctrain = optimizer.minimize(
            this_weight * self.direct_solution_direct_visual_solution_closs)

        # imagined_visual_solution == direct solution
        self.direct_solution_imagined_visual_solution_closs = 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=direct_solution_softmax, logits=imagined_visual_solution_logits)  
        self.direct_solution_imagined_visual_solution_closs += 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=imagined_visual_solution_softmax, logits=direct_solution_logits)  

        this_weight = config["loss_weights"]["direct_solution_imagined_visual_solution_closs"]
        self.direct_solution_imagined_visual_solution_ctrain = optimizer.minimize(
            this_weight * self.direct_solution_imagined_visual_solution_closs)

        # reconstructed problem direct solution == direct_visual_solution
        self.reconstructed_solution_direct_visual_solution_closs = 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=reconstructed_solution_softmax, logits=direct_visual_solution_logits)  
        self.reconstructed_solution_direct_visual_solution_closs += 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=direct_visual_solution_softmax, logits=reconstructed_direct_solution_logits)  

        this_weight = config["loss_weights"]["reconstructed_solution_direct_visual_solution_closs"]
        self.reconstructed_solution_direct_visual_solution_ctrain = optimizer.minimize(
            this_weight * self.reconstructed_solution_direct_visual_solution_closs)


        

consistency_model()

