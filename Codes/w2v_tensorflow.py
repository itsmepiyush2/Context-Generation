from w2v_numpy import get_wiki, get_context, test_model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import json

def modified_unigram_dist(sentences):
    word_freq = {}
    word_count = sum(len(sentence) for sentence in sentences)
    for sentence in sentences:
        for word in sentence:
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
  
    # vocab size
    V = len(word_freq)

    p_neg = np.zeros(V)
    for j in range(V):
        p_neg[j] = word_freq[j]**0.75

    # normalize it
    p_neg = p_neg / p_neg.sum()

    assert(np.all(p_neg > 0))
    return p_neg


def train_model(savedir):
    sentences, word2idx = get_wiki()
    V = len(word2idx)
    
    window_size = 10
    lr = 0.025
    final_lr = 0.0001
    # number of negative samples per input word required
    num_neg = 5
    epochs = 20
    samples_per_epoch = int(1e5)
    # embedding size
    D = 50
    # to linearly decrease the learning rate from max->min
    d_lr = (lr - final_lr) / epochs
    
    p_neg = modified_unigram_dist(sentences)
    
    # initialising the weights
    W1 = np.random.randn(V, D).astype(np.float32)
    W2 = np.random.randn(D, V).astype(np.float32)
    
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)
    
    tf_input = tf.placeholder(tf.int32, shape = (None,))
    tf_negword = tf.placeholder(tf.int32, shape = (None,))
    tf_context = tf.placeholder(tf.int32, shape = (None,))
    tfW1 = tf.Variable(W1)
    tfW2 = tf.Variable(W2.T)
    
    def dot(A, B):
        C = A * B
        return tf.reduce_sum(C, axis = 1)
    
    emb_input = tf.nn.embedding_lookup(tfW1, tf_input)
    emb_output = tf.nn.embedding_lookup(tfW2, tf_context)
    
    correct_output = dot(emb_input, emb_output)
    
    pos_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones(tf.shape(correct_output)), 
                                                       logits = correct_output)
     
    emb_input = tf.nn.embedding_lookup(tfW1, tf_negword)
    incorrect_output = dot(emb_input, emb_output)
    neg_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros(tf.shape(incorrect_output)),
                                                       logits=incorrect_output)
    loss = tf.reduce_mean(pos_loss) + tf.reduce_mean(neg_loss)
    
    train_op = tf.train.MomentumOptimizer(0.1, momentum=0.9).minimize(loss)
    
    session = tf.Session()
    init_op = tf.global_variables_initializer()
    session.run(init_op)
    
    costs = []
    total_words = sum(len(sentence) for sentence in sentences)
    print("Total number of words: ", total_words)
    
    for epoch in range(epochs):
        np.random.shuffle(sentences)
        cost = 0
        counter = 0
        
        inputs = []
        targets = []
        negwords = []
        
        t0 = datetime.now()
        
        for sentence in sentences:
            sentence = [w for w in sentence \
                        if np.random.random() < (1 - p_drop[w])
                        ]
            if len(sentence) < 2:
                continue
            
            randomly_ordered_pos = np.random.choice(len(sentence), size=len(sentence), replace=False)
            
            for j, pos in enumerate(randomly_ordered_pos):
                word = sentence[pos]
                context_words = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(V, p = p_neg)
                
                n = len(context_words)
                inputs += [word]*n
                negwords += [neg_word]*n
                targets += context_words
                
            if len(inputs) >= 128:
                _, c = session.run((train_op, loss),
                                  feed_dict={
                                            tf_input: inputs,
                                            tf_negword: negwords,
                                            tf_context: targets
                                            })
                cost += c

                inputs = []
                targets = []
                negwords = []
            counter += 1
            if counter % 100 == 0:
                sys.stdout.write("processed %s / %s\r" % (counter, len(sentences)))
                sys.stdout.flush()
        
        dt = datetime.now() - t0
        print("Epoch: ", epoch, "loss: ", cost, "time elapsed: ", dt)
        
        costs.append(cost)
        lr -= d_lr
    
    plt.plot(costs)
    plt.show()
    
    W1, W2T = session.run((tfW1, tfW2))
    W2 = W2T.T
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open('%s/word2idx.json' % savedir, 'w') as f:
        json.dump(word2idx, f)

    np.savez('%s/weights.npz' % savedir, W1, W2)
    return word2idx, W1, W2

if __name__ == "__main__":
    word2idx, W1, W2 = train_model('w2v_tensorflow')
    test_model(word2idx, W1, W2)