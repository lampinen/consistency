from __future__ import print_function
from __future__ import division 

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

## config
config = {
    "seq_length": 5,
    "max_n": 64, 
    "add_max_n": 64, 
    "output_seq_length": 4,
    "char_embedding_dim": 32,
    "vision_embedding_dim": 128,
    "problem_embedding_dim": 128,
    "rnn_num_layers": 2,
    "full_train_every": 1, # a full training example is given once every _ training examples
    "num_train": 3968,
    "init_lr": 0.001,
    "lr_decay": 0.95,
    "lr_decays_every": 100,
    "loss_weights": {
        "direct_solution_loss": 1.,
        "direct_visual_solution_loss": 1.,
        "reconstructed_solution_loss": 0.5,
        "imagined_visual_solution_loss": 0.5,

        "true_visual_problem_reconstruction_closs": 0.5,
        "true_visual_visual_reconstruction_closs": 0.03, # heuristic, makes about the same size as the other losses
        "imagined_visual_problem_reconstruction_closs": 0.5,
        "imagined_visual_visual_reconstruction_closs": 0.03, # heuristic, makes about the same size as the other losses

        "direct_solution_direct_visual_solution_closs": 0.5,
        "direct_solution_imagined_visual_solution_closs": 0.5,
        "reconstructed_solution_direct_visual_solution_closs": 0.5
    },
    "batch_size": 1 # batches larger than 1 are not supported, this is just to get rid of the "magic constant" feel where it has to be specified
}

##

np.random.seed(0)
tf.set_random_seed(0)

number_vocab = map(str, range(10))
op_vocab = ["+", "-", "*", "/"]
control_vocab = ["<PAD>", "<START>"] 
vocab = number_vocab + op_vocab + control_vocab 
vocab_dict = dict(zip(vocab, range(len(vocab)))) # index lookup

def text_to_indices(text):
    return [vocab_dict[char] for char in text]

def make_visual_array(n, m=None, op="*", dim=(config["max_n"])):
    x = np.zeros((2, dim, dim))
    if m is not None:
        if op == "*":
            if m > dim or n > dim:
                raise ValueError("One of n or m is greater than dim")
            x[0, :m, :n] = 1.
        elif op == "-":
            if m > dim**2 or n > dim**2:
                raise ValueError("One of m or n is greater than dim**2")
            q, r = divmod(n, dim)
            x[0, :q, :] = 1.
            x[0, q, :r] = 1.
            q, r = divmod(m, dim)
            x[1, :q, :] = 1.
            x[1, q, :r] = 1.
        elif op == "+":
            q, r = divmod(n, dim)
            q2, r2 = divmod(m, dim)
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
        q, r = divmod(n, 20)
        x[0, :q, :] = 1.
        x[0, q, :r] = 1.
    
    return np.transpose(x, [1,2,0])

def right_pad_seq(seq, n=config['output_seq_length'], pad_symbol="<PAD>"):
    return seq + [pad_symbol] * (n- len(seq))

def left_pad_seq(seq, n=config['seq_length'], pad_symbol="<PAD>"):
    return [pad_symbol] * (n- len(seq)) + seq


## Data generation 

def make_multiplication_full_example(n, m):
    """Makes a full data exemplar for learning multiplication (full meaning
    problem, solution, and visual problem)."""


    sol = n * m

    problem = list(str(n)) + ["*"] + list(str(m))
    problem = np.array([text_to_indices(left_pad_seq(problem))])
    solution = list(str(sol))
    solution = np.array([text_to_indices(right_pad_seq(solution))])
    visual_array = np.array([make_visual_array(n, m, op="*")])

    return {"problem": problem, "solution": solution, "visual_array": visual_array}

def make_addition_full_example(n, m):
    """Makes a full data exemplar for learning addition (full meaning
    problem, solution, and visual problem)."""


    sol = n + m

    problem = list(str(n)) + ["*"] + list(str(m))
    problem = np.array([text_to_indices(left_pad_seq(problem))])
    solution = list(str(sol))
    solution = np.array([text_to_indices(right_pad_seq(solution))])
    visual_array = np.array([make_visual_array(n, m, op="+", dim=config["max_n"])])

    return {"problem": problem, "solution": solution, "visual_array": visual_array}

dataset = [make_multiplication_full_example(n,m) for n in xrange(config["max_n"] + 1) for m in xrange(config["max_n"] + 1)]# + [make_addition_full_example(n,m) for n in xrange(config["add_max_n"] + 1) for m in xrange(config["add_max_n"] + 1)]
np.random.shuffle(dataset)

train_dataset = dataset[:config["num_train"]]
test_dataset = dataset[config["num_train"]:]


## Model(s)
class consistency_model(object):
    def __init__(self, no_consistency=False, no_visual=False):
        self.no_consistency = no_consistency
        self.no_visual = no_visual
        self.full_train_every = config['full_train_every']

        self.curr_lr = config["init_lr"]
        self.lr_decay = config["lr_decay"]
        self.lr_decays_every = config["lr_decays_every"]

        self.vocab_size = vocab_size = len(vocab)

        with tf.variable_scope('problem'):
            embedding_size = config['char_embedding_dim']
            input_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1/embedding_size, 0.1/embedding_size)) 
            output_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -0.1/embedding_size, 0.1/embedding_size)) 

        def build_problem_processing_net(embedded_input, reuse=True):
            """Processes problem from char embeddings"""
            with tf.variable_scope('problem/reading', reuse=reuse):
                cell = lambda: tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim'])
                stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config['rnn_num_layers'])]) 
            
                state = stacked_cell.zero_state(config['batch_size'], tf.float32)
                with tf.variable_scope("recurrence", reuse=reuse):
                    for step in xrange(config['seq_length']):
                        (output, state) = stacked_cell(embedded_input[:, step, :], state)
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

                cell = lambda: tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim'])
                stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config['rnn_num_layers'])]) 
                start_token = tf.nn.embedding_lookup(output_embeddings, vocab_dict["<START>"])
                char_logits = []

                state = stacked_cell.zero_state(config['batch_size'], tf.float32)
                state = tuple([tf.contrib.rnn.LSTMStateTuple(problem_embedding, state[0][1])] + [state[i] for i in xrange(1, len(state))])

                emb_output = tf.reshape(start_token, [config['batch_size'], -1])

                with tf.variable_scope("recurrence", reuse=reuse):
                    output_to_emb_output = tf.get_variable(
                        "output_to_emb_output",
                        [config['problem_embedding_dim'], config['char_embedding_dim']],
                        tf.float32)
                    for step in xrange(config['output_seq_length']):
                        (output, state) = stacked_cell(emb_output, state)
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
        self.direct_solution_loss = tf.reduce_mean(
            self.direct_solution_loss)

        optimizer = tf.train.AdamOptimizer(self.lr_ph)

        this_weight = config["loss_weights"]["direct_solution_loss"]
        self.direct_solution_train = optimizer.minimize(
            this_weight * self.direct_solution_loss)

        # Visual
        if no_visual:
            # sesssion
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            return

        def build_imagination_net(problem_embedding, reuse=True):
            """Generates "imaginary" visual array from a problem embedding"""
            with tf.variable_scope('imagination', reuse=reuse):
                # fully connected 
                net = problem_embedding
                net = slim.layers.fully_connected(net, 1 * 1 * 128, activation_fn=tf.nn.leaky_relu)
                net = tf.reshape(net, [-1, 1, 1, 128])
                #print(net.get_shape())

                # 3
                net = tf.image.resize_bilinear(net, [2, 2])
                #print(net.get_shape())
                net = slim.layers.conv2d_transpose(net, 64, [4, 4], stride=2)
                #print(net.get_shape())

                # 2
                net = tf.image.resize_bilinear(net, [8, 8])
                #print(net.get_shape())
                net = slim.layers.conv2d_transpose(net, 32, [4, 4], stride=2)
                #print(net.get_shape())

                # 1
                net = tf.image.resize_bilinear(net, [32, 32])
                #print(net.get_shape())
                net = slim.layers.conv2d_transpose(net, 2, [4, 4], stride=2)
                #print(net.get_shape())
                return net


        def build_perception_net(perception_input, reuse=True): 
            """Generates perceptual embedding of visual array"""
            with tf.variable_scope('perception', reuse=reuse):
                # 1
                net = slim.layers.conv2d(perception_input, 32, [4, 4], stride=2)
                #print(net.get_shape())
                net = slim.layers.avg_pool2d(net, [2, 2], stride=2)
                #print(net.get_shape())
                # 2
                net = slim.layers.conv2d(net, 64, [4, 4], stride=2)
                #print(net.get_shape())
                net = slim.layers.avg_pool2d(net, [2, 2], stride=2)
                #print(net.get_shape())
                # 3
                net = slim.layers.conv2d(net, 128, [4, 4], stride=2)
                #print(net.get_shape())
                net = slim.layers.avg_pool2d(net, [2, 2], stride=2)
                # fc
                #print(net.get_shape())
                net = slim.flatten(net)
                representation = slim.layers.fully_connected(net, config["vision_embedding_dim"], activation_fn=tf.nn.leaky_relu)
                return representation


        def build_perceptual_solution_net(vision_embedding, reuse=True):
            """Solves problem from visual embedding"""
            with tf.variable_scope('perception/solution', reuse=reuse):

                cell = lambda: tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim'])
                stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config['rnn_num_layers'])]) 
                start_token = tf.nn.embedding_lookup(output_embeddings, vocab_dict["<START>"])
                char_logits = []

                state = stacked_cell.zero_state(config['batch_size'], tf.float32)
                state = tuple([tf.contrib.rnn.LSTMStateTuple(vision_embedding, state[0][1])] + [state[i] for i in xrange(1, len(state))])
                emb_output = tf.reshape(start_token, [config['batch_size'], -1])

                with tf.variable_scope("recurrence", reuse=reuse):
                    output_to_emb_output = tf.get_variable(
                        "output_to_emb_output",
                        [config['problem_embedding_dim'], config['char_embedding_dim']],
                        tf.float32)
                    for step in xrange(config['output_seq_length']):
                        (output, state) = stacked_cell(emb_output, state)
                        emb_output = tf.matmul(output, output_to_emb_output) 
                        this_char_logits = tf.matmul(emb_output, tf.transpose(output_embeddings))
                        char_logits.append(this_char_logits)
                        tf.get_variable_scope().reuse_variables()

                char_logits = tf.stack(char_logits, axis=1)
                return char_logits 


        # visual path
        self.vision_input_ph = tf.placeholder(tf.float32, [None, 64, 64, 2])
        visual_input_embedding = build_perception_net(self.vision_input_ph,
                                                      reuse=False)
        direct_visual_solution_logits = build_perceptual_solution_net(
            visual_input_embedding, reuse=False) 
        direct_visual_solution_softmax = tf.nn.softmax(
            direct_visual_solution_logits)

        self.direct_visual_solution_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_solution, logits=direct_solution_logits) 
        self.direct_visual_solution_loss = tf.reduce_mean(
            self.direct_visual_solution_loss)

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
        self.imagined_visual_solution_loss = tf.reduce_mean(
            self.imagined_visual_solution_loss)

        this_weight = config["loss_weights"]["imagined_visual_solution_loss"]
        self.imagined_visual_solution_train = optimizer.minimize(
            this_weight * self.imagined_visual_solution_loss)

        def build_perceptual_problem_reconstruction_net(vision_embedding, ground_truth, reuse=True):
            """Reconstructs problem from visual embedding"""
            with tf.variable_scope('perception/problem_reconstruction', reuse=reuse):

                cell = lambda: tf.contrib.rnn.BasicLSTMCell(config['problem_embedding_dim'])
                stacked_cell = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(config['rnn_num_layers'])]) 
                start_token = tf.nn.embedding_lookup(output_embeddings, vocab_dict["<START>"])
                char_logits = []
                emb_outputs = []

                state = stacked_cell.zero_state(config['batch_size'], tf.float32)
                state = tuple([tf.contrib.rnn.LSTMStateTuple(vision_embedding, state[0][1])] + [state[i] for i in xrange(1, len(state))])
                emb_output = tf.reshape(start_token, [config['batch_size'], -1])

                with tf.variable_scope("recurrence", reuse=reuse):
                    output_to_emb_output = tf.get_variable(
                        "output_to_emb_output",
                        [config['problem_embedding_dim'], config['char_embedding_dim']],
                        tf.float32)
                    for step in xrange(config['seq_length']):
                        this_input = ground_truth[:, step-1, :]
                        (output, state) = stacked_cell(this_input, state)
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
        self.reconstructed_solution_loss = tf.reduce_mean(
            self.reconstructed_solution_loss)

        this_weight = config["loss_weights"]["reconstructed_solution_loss"]
        self.reconstructed_solution_train = optimizer.minimize( 
            this_weight * self.reconstructed_solution_loss)


        # Consistency
        if no_consistency:
            # sesssion
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            return

        problem_input_onehot = tf.one_hot(self.problem_input_ph, vocab_size)

        # vision -> reconstructed problem = ground truth problem
        (true_visual_problem_reconstruction_logits,
         true_visual_problem_reconstruction) = build_perceptual_problem_reconstruction_net(
             visual_input_embedding, problem_input_embeddings)
        self.true_visual_problem_reconstruction_closs = tf.nn.softmax_cross_entropy_with_logits(
            labels=problem_input_onehot, logits=true_visual_problem_reconstruction_logits) 
        self.true_visual_problem_reconstruction_closs = tf.reduce_mean(
            self.true_visual_problem_reconstruction_closs)
        
        this_weight = config["loss_weights"]["true_visual_problem_reconstruction_closs"]
        self.true_visual_problem_reconstruction_ctrain = optimizer.minimize(
            this_weight * self.true_visual_problem_reconstruction_closs)
                
        # ground_truth_problem -> imagined visual -> reconstructed problem = ground truth problem
        (imagined_visual_problem_reconstruction_logits,
         imagined_visual_problem_reconstruction) = build_perceptual_problem_reconstruction_net(
             imagined_visual_embedding, problem_input_embeddings)
        self.imagined_visual_problem_reconstruction_closs = tf.nn.softmax_cross_entropy_with_logits(
            labels=problem_input_onehot, logits=imagined_visual_problem_reconstruction_logits) 
        self.imagined_visual_problem_reconstruction_closs = tf.reduce_mean(
            self.imagined_visual_problem_reconstruction_closs)

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
        self.direct_solution_direct_visual_solution_closs = tf.reduce_mean(
            self.direct_solution_direct_visual_solution_closs)

        this_weight = config["loss_weights"]["direct_solution_direct_visual_solution_closs"]
        self.direct_solution_direct_visual_solution_ctrain = optimizer.minimize(
            this_weight * self.direct_solution_direct_visual_solution_closs)

        # imagined_visual_solution == direct solution
        self.direct_solution_imagined_visual_solution_closs = 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=direct_solution_softmax, logits=imagined_visual_solution_logits)  
        self.direct_solution_imagined_visual_solution_closs += 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=imagined_visual_solution_softmax, logits=direct_solution_logits)  
        self.direct_solution_imagined_visual_solution_closs = tf.reduce_mean(
            self.direct_solution_imagined_visual_solution_closs)

        this_weight = config["loss_weights"]["direct_solution_imagined_visual_solution_closs"]
        self.direct_solution_imagined_visual_solution_ctrain = optimizer.minimize(
            this_weight * self.direct_solution_imagined_visual_solution_closs)

        # reconstructed problem direct solution == direct_visual_solution
        self.reconstructed_solution_direct_visual_solution_closs = 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=reconstructed_solution_softmax, logits=direct_visual_solution_logits)  
        self.reconstructed_solution_direct_visual_solution_closs += 0.5*tf.nn.softmax_cross_entropy_with_logits(
            labels=direct_visual_solution_softmax, logits=reconstructed_direct_solution_logits)  
        self.reconstructed_solution_direct_visual_solution_closs = tf.reduce_mean(
            self.reconstructed_solution_direct_visual_solution_closs)

        this_weight = config["loss_weights"]["reconstructed_solution_direct_visual_solution_closs"]
        self.reconstructed_solution_direct_visual_solution_ctrain = optimizer.minimize(
            this_weight * self.reconstructed_solution_direct_visual_solution_closs)

        # aggregated losses

        self.visual_full_basic_loss = (self.direct_solution_loss + 
                                 config["loss_weights"]["imagined_visual_solution_loss"] * self.imagined_visual_solution_loss)
        self.visual_full_basic_train = optimizer.minimize(self.visual_full_basic_loss)

        self.visual_full_loss = (self.visual_full_basic_loss + 
                                 config["loss_weights"]["direct_visual_solution_loss"] * self.direct_visual_solution_loss + 
                                 config["loss_weights"]["reconstructed_solution_loss"] * self.reconstructed_solution_loss)
        self.visual_full_train = optimizer.minimize(self.visual_full_loss)

        self.consistency_full_basic_loss = (self.visual_full_basic_loss +
                                            config["loss_weights"]["imagined_visual_problem_reconstruction_closs"] * self.imagined_visual_problem_reconstruction_closs + 
                                            config["loss_weights"]["direct_solution_imagined_visual_solution_closs"] * self.direct_solution_imagined_visual_solution_closs) 
        self.consistency_full_basic_train = optimizer.minimize(self.consistency_full_basic_loss)

        self.consistency_full_loss = (self.visual_full_loss +
                                      config["loss_weights"]["imagined_visual_problem_reconstruction_closs"] * self.imagined_visual_problem_reconstruction_closs + 
                                      config["loss_weights"]["direct_solution_imagined_visual_solution_closs"] * self.direct_solution_imagined_visual_solution_closs + 
                                      config["loss_weights"]["true_visual_problem_reconstruction_closs"] * self.true_visual_problem_reconstruction_closs + 
                                      config["loss_weights"]["direct_solution_direct_visual_solution_closs"] * self.direct_solution_direct_visual_solution_closs + 
                                      config["loss_weights"]["reconstructed_solution_direct_visual_solution_closs"] * self.reconstructed_solution_direct_visual_solution_closs + 
                                      config["loss_weights"]["imagined_visual_visual_reconstruction_closs"] * self.imagined_visual_visual_reconstruction_closs + 
                                      config["loss_weights"]["true_visual_visual_reconstruction_closs"] * self.true_visual_visual_reconstruction_closs) 
        self.consistency_full_train = optimizer.minimize(self.consistency_full_loss)

        self.unlabelled_full_loss = (config["loss_weights"]["imagined_visual_problem_reconstruction_closs"] * self.imagined_visual_problem_reconstruction_closs + 
                                     config["loss_weights"]["direct_solution_imagined_visual_solution_closs"] * self.direct_solution_imagined_visual_solution_closs + 
                                     config["loss_weights"]["true_visual_problem_reconstruction_closs"] * self.true_visual_problem_reconstruction_closs + 
                                     config["loss_weights"]["direct_solution_direct_visual_solution_closs"] * self.direct_solution_direct_visual_solution_closs + 
                                     config["loss_weights"]["reconstructed_solution_direct_visual_solution_closs"] * self.reconstructed_solution_direct_visual_solution_closs + 
                                     config["loss_weights"]["imagined_visual_visual_reconstruction_closs"] * self.imagined_visual_visual_reconstruction_closs + 
                                     config["loss_weights"]["true_visual_visual_reconstruction_closs"] * self.true_visual_visual_reconstruction_closs) 
        self.unlabelled_full_train = optimizer.minimize(self.unlabelled_full_loss)

        # sesssion
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def run_basic_train_example(self, train_exemplars):
        """runs a train step without visual input -- nobody gets visual examples all the time"""
        if self.no_visual:
            self.sess.run(
                self.direct_solution_train,
                feed_dict={self.problem_input_ph: train_exemplars[0]["problem"],
                           self.solution_input_ph: train_exemplars[0]["solution"],
                           self.lr_ph: self.curr_lr})
        elif self.no_consistency:
            self.sess.run(
                self.visual_full_basic_train,
                feed_dict={self.problem_input_ph: train_exemplars[0]["problem"],
                           self.solution_input_ph: train_exemplars[0]["solution"],
                           self.lr_ph: self.curr_lr})
        else:
            self.sess.run(
                self.consistency_full_basic_train,
                feed_dict={self.problem_input_ph: train_exemplars[0]["problem"],
                           self.solution_input_ph: train_exemplars[0]["solution"],
                           self.lr_ph: self.curr_lr})


    def run_full_train_example(self, train_exemplars):
        if self.no_visual:
            raise NotImplementedError("A model with no_visual = True cannot run full examples")
        elif self.no_consistency:
            self.sess.run(
                self.visual_full_train,
                feed_dict={self.problem_input_ph: train_exemplars[-1]["problem"],
                           self.vision_input_ph: train_exemplars[-1]["visual_array"],
                           self.solution_input_ph: train_exemplars[-1]["solution"],
                           self.lr_ph: self.curr_lr})
        else:
            self.sess.run(
                self.consistency_full_train,
                feed_dict={self.problem_input_ph: train_exemplars[-1]["problem"],
                           self.vision_input_ph: train_exemplars[-1]["visual_array"],
                           self.solution_input_ph: train_exemplars[-1]["solution"],
                           self.lr_ph: self.curr_lr})


    def run_unlabelled_train_example(self, train_exemplars):
        if self.no_visual or self.no_consistency:
            raise NotImplementedError("A model with no_consistency = True cannot run unlabeled examples")

        self.sess.run(
            self.unlabelled_full_train,
            feed_dict={self.problem_input_ph: train_exemplars[-7]["problem"],
                       self.vision_input_ph: train_exemplars[-7]["visual_array"],
                       self.lr_ph: self.curr_lr})

    def run_train_dataset(self, train_dataset):
        for i in xrange(len(train_dataset)):
            if i < 10: 
                train_exemplars = train_dataset[i-10:] + train_dataset[:i+1] 
            else: 
                train_exemplars = train_dataset[i-10:i+1]

            #assert(len(train_exemplars) == 11)
            if self.no_visual and (i % self.full_train_every == 0):
                self.run_basic_train_example(train_exemplars)
            elif (i % self.full_train_every == 0):
                self.run_full_train_example(train_exemplars)
            elif not (self.no_consistency):
                self.run_unlabelled_train_example(train_exemplars)
           

    def run_test_example(self, test_exemplar, test_only_main=False):
        if self.no_visual or test_only_main:
            this_exemplar_losses = self.sess.run(
                [self.direct_solution_loss],
                feed_dict={self.problem_input_ph: test_exemplar["problem"],
                           self.solution_input_ph: test_exemplar["solution"]})


        elif self.no_consistency:
            this_exemplar_losses = self.sess.run(
                [self.direct_solution_loss, 
                 self.imagined_visual_solution_loss,
                 self.direct_visual_solution_loss,
                 self.reconstructed_solution_loss],
                feed_dict={self.problem_input_ph: test_exemplar["problem"],
                           self.vision_input_ph: test_exemplar["visual_array"],
                           self.solution_input_ph: test_exemplar["solution"]})

        else:
            this_exemplar_losses = self.sess.run(
                [self.direct_solution_loss, 
                 self.imagined_visual_solution_loss,
                 self.imagined_visual_problem_reconstruction_closs,
                 self.direct_solution_imagined_visual_solution_closs,
                 self.direct_visual_solution_loss,
                 self.reconstructed_solution_loss,
                 self.true_visual_problem_reconstruction_closs,
                 self.direct_solution_direct_visual_solution_closs,
                 self.reconstructed_solution_direct_visual_solution_closs,
                 self.imagined_visual_visual_reconstruction_closs,
                 self.true_visual_visual_reconstruction_closs],
                feed_dict={self.problem_input_ph: test_exemplar["problem"],
                           self.vision_input_ph: test_exemplar["visual_array"],
                           self.solution_input_ph: test_exemplar["solution"]})

        return this_exemplar_losses 

    def run_test_dataset(self, test_dataset, test_only_main=False):
        if self.no_visual or test_only_main:
            test_direct_solution_loss = 0. 
            for test_exemplar in test_dataset:
                (this_direct_solution_loss,) = self.run_test_example(
                    test_exemplar, test_only_main=test_only_main)
                test_direct_solution_loss += this_direct_solution_loss
            num_test = len(test_dataset)
            test_direct_solution_loss /= num_test 
            return [test_direct_solution_loss,]

        elif self.no_consistency:
            [test_direct_solution_loss, 
             test_imagined_visual_solution_loss,
             test_direct_visual_solution_loss,
             test_reconstructed_solution_loss] = [0.] * 4
            for test_exemplar in test_dataset:
                (this_direct_solution_loss, 
                 this_imagined_visual_solution_loss,
                 this_direct_visual_solution_loss,
                 this_reconstructed_solution_loss) = self.run_test_example(
                     test_exemplar) 

                test_direct_solution_loss += this_direct_solution_loss 
                test_imagined_visual_solution_loss += this_imagined_visual_solution_loss
                test_direct_visual_solution_loss += this_direct_visual_solution_loss
                test_reconstructed_solution_loss += this_reconstructed_solution_loss

            num_test = len(test_dataset)
            test_direct_solution_loss /= num_test 
            test_imagined_visual_solution_loss /= num_test
            test_direct_visual_solution_loss /= num_test
            test_reconstructed_solution_loss /= num_test

            return [test_direct_solution_loss, 
                    test_imagined_visual_solution_loss,
                    test_direct_visual_solution_loss,
                    test_reconstructed_solution_loss]

        else:
            [test_direct_solution_loss, 
             test_imagined_visual_solution_loss,
             test_imagined_visual_problem_reconstruction_closs,
             test_direct_solution_imagined_visual_solution_closs,
             test_direct_visual_solution_loss,
             test_reconstructed_solution_loss,
             test_true_visual_problem_reconstruction_closs,
             test_direct_solution_direct_visual_solution_closs,
             test_reconstructed_solution_direct_visual_solution_closs,
             test_imagined_visual_visual_reconstruction_closs,
             test_true_visual_visual_reconstruction_closs] = [0.] * 11
            for test_exemplar in test_dataset:
                (this_direct_solution_loss, 
                 this_imagined_visual_solution_loss,
                 this_imagined_visual_problem_reconstruction_closs,
                 this_direct_solution_imagined_visual_solution_closs,
                 this_direct_visual_solution_loss,
                 this_reconstructed_solution_loss,
                 this_true_visual_problem_reconstruction_closs,
                 this_direct_solution_direct_visual_solution_closs,
                 this_reconstructed_solution_direct_visual_solution_closs,
                 this_imagined_visual_visual_reconstruction_closs,
                 this_true_visual_visual_reconstruction_closs) = self.run_test_example(
                    test_exemplar) 

                test_direct_solution_loss += this_direct_solution_loss 
                test_imagined_visual_solution_loss += this_imagined_visual_solution_loss
                test_imagined_visual_problem_reconstruction_closs += this_imagined_visual_problem_reconstruction_closs
                test_direct_solution_imagined_visual_solution_closs += this_direct_solution_imagined_visual_solution_closs
                test_direct_visual_solution_loss += this_direct_visual_solution_loss
                test_reconstructed_solution_loss += this_reconstructed_solution_loss
                test_true_visual_problem_reconstruction_closs += this_true_visual_problem_reconstruction_closs
                test_direct_solution_direct_visual_solution_closs += this_direct_solution_direct_visual_solution_closs
                test_reconstructed_solution_direct_visual_solution_closs += this_reconstructed_solution_direct_visual_solution_closs
                test_imagined_visual_visual_reconstruction_closs += this_imagined_visual_visual_reconstruction_closs
                test_true_visual_visual_reconstruction_closs += this_true_visual_visual_reconstruction_closs

            num_test = len(test_dataset)
            test_direct_solution_loss /= num_test 
            test_imagined_visual_solution_loss /= num_test
            test_imagined_visual_problem_reconstruction_closs /= num_test
            test_direct_solution_imagined_visual_solution_closs /= num_test
            test_direct_visual_solution_loss /= num_test
            test_reconstructed_solution_loss /= num_test
            test_true_visual_problem_reconstruction_closs /= num_test
            test_direct_solution_direct_visual_solution_closs /= num_test
            test_reconstructed_solution_direct_visual_solution_closs /= num_test
            test_imagined_visual_visual_reconstruction_closs /= num_test
            test_true_visual_visual_reconstruction_closs /= num_test

            return [test_direct_solution_loss, 
                    test_imagined_visual_solution_loss,
                    test_imagined_visual_problem_reconstruction_closs,
                    test_direct_solution_imagined_visual_solution_closs,
                    test_direct_visual_solution_loss,
                    test_reconstructed_solution_loss,
                    test_true_visual_problem_reconstruction_closs,
                    test_direct_solution_direct_visual_solution_closs,
                    test_reconstructed_solution_direct_visual_solution_closs,
                    test_imagined_visual_visual_reconstruction_closs,
                    test_true_visual_visual_reconstruction_closs]


    def run_training(self, train_dataset, test_dataset, nepochs=1000, test_only_main=False):
        train_losses = []
        test_losses = []
        train_losses.append(self.run_test_dataset(train_dataset,test_only_main=True))
        test_losses.append(self.run_test_dataset(test_dataset,test_only_main=True))
        print("Pre train")
        print(train_losses[-1])
        print("Pre test")
        print(test_losses[-1])
        for epoch in xrange(nepochs):
#            np.random.shuffle(train_dataset)
            self.run_train_dataset(train_dataset)
            if epoch % 10 == 0:
                train_losses.append(self.run_test_dataset(train_dataset, test_only_main=True))
                test_losses.append(self.run_test_dataset(test_dataset, test_only_main=True))
                print("Epoch %i train:" % epoch)
                print(train_losses[-1])
                print("test:")
                print(test_losses[-1])
            if epoch % self.lr_decays_every == 0 and epoch != 0:
                self.curr_lr *= self.lr_decay
        print("Post train")
        print(train_losses[-1])
        print("Post test")
        print(test_losses[-1])

np.random.seed(0)
tf.set_random_seed(0)
cm = consistency_model()
cm.run_training(train_dataset, test_dataset, 2000, test_only_main=True)
