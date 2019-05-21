import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob
from scipy.special import expit as sigmoid
import sys
import os
sys.path.append(os.path.abspath('..'))
import json
import string
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import pairwise_distances

def remove_punct(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def get_wiki():
  V = 20000
  files = glob('../Wiki/enwiki*.txt')
  all_word_counts = {}
  print("Starting to count the word frequency")
  for f in files:
    for line in open(f):
      if line and line[0] not in '[*-|=\{\}':
        s = remove_punct(line).lower().split()
        if len(s) > 1:
          for word in s:
            if word not in all_word_counts:
              all_word_counts[word] = 0
            all_word_counts[word] += 1
  print("Finished counting the words")

  V = min(V, len(all_word_counts))
  all_word_counts = sorted(all_word_counts.items(), key = lambda x: x[1], reverse = True)

  top_words = [w for w, count in all_word_counts[:V-1]] + ['UNK']
  word2idx = {w:i for i, w in enumerate(top_words)}
  unk = word2idx['UNK']

  sents = []
  for f in files:
    for line in open(f):
      if line and line[0] not in '[*-|=\{\}':
        s = remove_punct(line).lower().split()
        if len(s) > 1:
          sent = [word2idx[w] if w in word2idx else unk for w in s]
          sents.append(sent)
  return sents, word2idx

def modified_unigram_dist(sentences, V):
    word_freq = np.zeros(V)
    for sentence in sentences:
        for word in sentence:
            word_freq[word] += 1
    
    p_neg = word_freq ** 0.75 # smooth
    p_neg = p_neg / p_neg.sum() # normalise
    
    # sanity check
    assert(np.all(p_neg) > 0)
    
    return p_neg 
 
    
def get_context(pos, sentence, window_size):
    # input: x x x c c c c c pos c c c c c x x x 
    # output: c c c c c c c c c c
    start = max(0, pos - window_size)
    end = min(len(sentence), pos + window_size)
    context = []
    for ctx_pos, ctx_word in enumerate(sentence[start:end], start = start):
        if ctx_pos != pos:
            context.append(ctx_word)
    return context


def SGD(word, targets, label, lr, W1, W2):
    activation = W1[word].dot(W2[:,targets])
    prob = sigmoid(activation)
    
    gW2 = np.outer(W1[word], prob - label)
    gW1 = np.sum((prob - label) * W2[:,targets], axis = 1)
    
    W2[:,targets] -= lr * gW2
    W1[word] -= lr * gW1
    
    # loss -> binary crossentropy
    J = label * np.log(prob + 1e-10) + (1 - label) * np.log(prob + 1e-10)
    J = J.sum()
    return J


def train_model(savedir):
    sentences, word2idx = get_wiki()
    V = len(word2idx)
    
    window_size = 5
    lr = 0.025
    final_lr = 0.0001
    # number of negative samples per input word required
    num_neg = 5
    epochs = 20
    # embedding size
    D = 50
    # to linearly decrease the learning rate from max->min
    d_lr = (lr - final_lr) / epochs
    
    
    # initialising the parameters
    W1 = np.random.randn(V, D)
    W2 = np.random.randn(D, V)
    
    # calculating the negative sampling distance
    p_neg = modified_unigram_dist(sentences, V)
    
    # calculating the subsampling distance to drop random words
    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)
    
    losses = []
    total_words = sum(len(sentence) for sentence in sentences)
    print("Total number of words: ", total_words)
    
    # train the model
    for epoch in range(epochs):
        # shuffle
        np.random.shuffle(sentences)
        
        loss = 0
        counter = 0
        t0 = datetime.now()
        for sentence in sentences:
            # remove the negatively sampled words
            sentence = [w for w in sentence \
                        if np.random.random() < (1 - p_drop[w])
                        ]
            if len(sentence) < 2:
                continue
        
            # randomly ordering the words in the sentence to help training
            randomly_ordered_pos = np.random.choice(len(sentence), size=len(sentence), replace=False)
        
            for pos in randomly_ordered_pos:
                word = sentence[pos]
                context_words = get_context(pos, sentence, window_size)
                # sample the negative words from p_neg
                neg_word = np.random.choice(V, p = p_neg)
                targets = np.array(context_words)
            
                # stochastic gradient descent
                c = SGD(word, targets, 1, lr, W1, W2)
                loss += c
                c = SGD(neg_word, targets, 0, lr, W1, W2)
                loss += c
            
            counter += 1
            if counter % 100 == 0:
                sys.stdout.write("processed %s / %s\r" % (counter, len(sentences)))
                sys.stdout.flush()
        
        dt = datetime.now() - t0
        print("Epoch: ", epoch, "loss: ", loss, "time elapsed: ", dt)
        
        losses.append(loss)
        lr -= d_lr
    
    plt.plot(losses)
    plt.show()
    
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    with open('%s/word2idx.json' % savedir, 'w') as f:
        json.dump(word2idx, f)

    np.savez('%s/weights.npz' % savedir, W1, W2)
    
    return word2idx, W1, W2

def analogy(pos1, neg1, pos2, neg2, word2idx, idx2word, W):
    V, D = W.shape
    print("testing: %s - %s = %s - %s" % (pos1, neg1, pos2, neg2))
    for w in (pos1, neg1, pos2, neg2):
        if w not in word2idx:
            print("Sorry, %s not in word2idx" % w)
            return
    p1 = W[word2idx[pos1]]
    n1 = W[word2idx[neg1]]
    p2 = W[word2idx[pos2]]
    n2 = W[word2idx[neg2]]
    
    vec = p1 - n1 + n2
    distances = pairwise_distances(vec.reshape(1, D), W, metric='cosine').reshape(V)
    idx = distances.argsort()[:10]
    
    best_idx = -1
    keep_out = [word2idx[w] for w in (pos1, neg1, neg2)]
    for i in idx:
        if i not in keep_out:
            best_idx = i
            break
    
    print("got: %s - %s = %s - %s" % (pos1, neg1, idx2word[best_idx], neg2))
    print("closest 10:")
    for i in idx:
        print(idx2word[i], distances[i])

    print("dist to %s:" % pos2, cosine(p2, vec))

def test_model(word2idx, W1, W2):
    # taking average of both the weight vectors to find the embedding
    W = (W1 + W2.T) / 2
    
    idx2word = {i:j for j, i in word2idx.items()}
    for We in (W1, W):
        analogy('king', 'man', 'queen', 'woman', word2idx, idx2word, We)
        analogy('king', 'prince', 'queen', 'princess', word2idx, idx2word, We)
        analogy('miami', 'florida', 'dallas', 'texas', word2idx, idx2word, We)
        analogy('einstein', 'scientist', 'picasso', 'painter', word2idx, idx2word, We)
        analogy('japan', 'sushi', 'germany', 'bratwurst', word2idx, idx2word, We)
        analogy('man', 'woman', 'he', 'she', word2idx, idx2word, We)
        analogy('man', 'woman', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('man', 'woman', 'brother', 'sister', word2idx, idx2word, We)
        analogy('man', 'woman', 'husband', 'wife', word2idx, idx2word, We)
        analogy('man', 'woman', 'actor', 'actress', word2idx, idx2word, We)
        analogy('man', 'woman', 'father', 'mother', word2idx, idx2word, We)
        analogy('heir', 'heiress', 'prince', 'princess', word2idx, idx2word, We)
        analogy('nephew', 'niece', 'uncle', 'aunt', word2idx, idx2word, We)
        analogy('france', 'paris', 'japan', 'tokyo', word2idx, idx2word, We)
        analogy('france', 'paris', 'china', 'beijing', word2idx, idx2word, We)
        analogy('february', 'january', 'december', 'november', word2idx, idx2word, We)
        analogy('france', 'paris', 'germany', 'berlin', word2idx, idx2word, We)
        analogy('week', 'day', 'year', 'month', word2idx, idx2word, We)
        analogy('week', 'day', 'hour', 'minute', word2idx, idx2word, We)
        analogy('france', 'paris', 'italy', 'rome', word2idx, idx2word, We)
        analogy('paris', 'france', 'rome', 'italy', word2idx, idx2word, We)
        analogy('france', 'french', 'england', 'english', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'china', 'chinese', word2idx, idx2word, We)
        analogy('china', 'chinese', 'america', 'american', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'italy', 'italian', word2idx, idx2word, We)
        analogy('japan', 'japanese', 'australia', 'australian', word2idx, idx2word, We)
        analogy('walk', 'walking', 'swim', 'swimming', word2idx, idx2word, We)
    
if __name__ == '__main__':
    word2idx, W1, W2 = train_model('w2v_model')
    test_model(word2idx, W1, W2)  