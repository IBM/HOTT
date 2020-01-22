from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import time

from data import loader
from knn_classifier import knn

import distances
import hott

# Download datasets used by Kusner et al from
# https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
# and put them into
data_path = './data/'


# Download GloVe 6B tokens, 300d word embeddings from
# https://nlp.stanford.edu/projects/glove/
# and put them into
embeddings_path = './data/glove.6B/glove.6B.300d.txt'

# Pick a dataset (uncomment the line you want)
data_name = 'bbcsport-emd_tr_te_split.mat'
# data_name = 'twitter-emd_tr_te_split.mat'
# data_name = 'r8-emd_tr_te3.mat'
# data_name = 'amazon-emd_tr_te_split.mat'
# data_name = 'classic-emd_tr_te_split.mat'
# data_name = 'ohsumed-emd_tr_te_ix.mat'

# p=1 for W1 and p=2 for W2
p = 1
data = loader(data_path + data_name, embeddings_path, p=p)
cost_E = data['cost_E']
cost_T = data['cost_T']

bow_data, y = data['X'], data['y']
topic_proportions = data['proportions']

seed = 0
bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(bow_data, topic_proportions, y, random_state=seed)

# Pick a method among RWMD, WMD, WMD-T20, HOTT, HOFTT
methods = {'HOTT': hott.hott,
           'HOFTT': hott.hoftt,
           'RWMD': distances.rwmd,
           'WMD-T20': lambda p, q, C: distances.wmd(p, q, C, truncate=20),
           'WMD': distances.wmd}
    
for method in methods.keys():
    
    t_s = time.time()
    # Get train/test data representation and transport cost
    if method in ['HOTT', 'HOFTT']:
        # If method is HOTT or HOFTT train LDA and compute topic-topic transport cost
        X_train, X_test = topic_train, topic_test
        C = data['cost_T']
    else:
        # Normalize BOW and compute word-word transport cost
        X_train, X_test = normalize(bow_train, 'l1'), normalize(bow_test, 'l1')
        C = data['cost_E']
    
    # Compute test error
    test_error = knn(X_train, X_test, y_train, y_test, methods[method], C, n_neighbors=7)
    print(method + ' test error is %f; took %.2f seconds' % (test_error, time.time()-t_s))

# Done!
