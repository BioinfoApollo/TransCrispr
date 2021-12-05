from keras.preprocessing.text import Tokenizer
import pandas as pd

class Nuc_NtsTokenizer(Tokenizer):

    def __init__(self):
        Tokenizer.__init__(self)
        self.dic = ['SOS']
        self.dic += [a for a in 'ATCG']
        self.fit_on_texts(self.dic)

class Dimer_NtsTokenizer(Tokenizer):

    def __init__(self):
        Tokenizer.__init__(self)
        self.dic = ['SOS']
        self.dic += ['SEP1']

        self.dic += [a for a in 'ATCG']
        self.dic += [a + b for a in 'ATCG' for b in 'ATCG']
        self.dic += [a + '0' for a in 'ATCG']

        self.fit_on_texts(self.dic)

def split_seqs(seq):
    t = Nuc_NtsTokenizer()

    result = 'SOS'
    lens = len(seq)

    for i in range(lens):
        result += ' ' + seq[i].upper()

    seq_result = t.texts_to_sequences([result])

    nuc_seq = pd.Series(seq_result[0]) - 1
    pos_seq = pd.Series(i for i in range(lens + 1))
    seq = pd.concat([nuc_seq, pos_seq], axis=0, ignore_index=True)

    return seq

def Dimer_split_seqs(seq):
    t = Dimer_NtsTokenizer()

    result = 'SOS'
    lens = len(seq)

    for i in range(lens):
        result += ' ' + seq[i].upper()

    # dimer_encode
    result += ' '
    result += 'SEP1'

    seq += '0'
    wt = 2
    for i in range(lens):
        result += ' ' + seq[i:i + wt].upper()

    seq_result = t.texts_to_sequences([result])

    nuc_seq = pd.Series(seq_result[0]) - 1
    pos_seq = pd.Series(i for i in range(lens + 1))

    seq = pd.concat([nuc_seq, pos_seq], axis=0, ignore_index=True)

    return seq

def DimerOnly_split_seqs(seq):
    t = Dimer_NtsTokenizer()

    result = 'SOS'
    lens = len(seq)

    seq += '0'
    wt = 2
    for i in range(lens):
        result += ' ' + seq[i:i + wt].upper()

    seq_result = t.texts_to_sequences([result])

    nuc_seq = pd.Series(seq_result[0]) - 1
    pos_seq = pd.Series(i for i in range(lens + 1))

    seq = pd.concat([nuc_seq, pos_seq], axis=0, ignore_index=True)

    return seq

class Result(object):
    Best = -1

from sklearn.metrics import mean_squared_error, r2_score
import scipy as sp

def get_score_at_test(model, input, result, label, issave=True, savepath=None):
    pred_label = model.predict([input])
    mse = mean_squared_error(label, pred_label)
    spearmanr = sp.stats.spearmanr(label, pred_label)[0]
    r2 = r2_score(label, pred_label)
    y_test1 = label.reshape(-1, )
    y_test_pre1 = pred_label.reshape(-1, )
    pearson = sp.stats.pearsonr(y_test1, y_test_pre1)[0]

    if result.Best < spearmanr:
        result.Best = spearmanr
        if issave:
            model.save_weights(savepath) 
        print('best')
    return 'MES:' + str(mse), 'Spearman:' + str(spearmanr), 'Pearson:' + str(pearson), 'r2:' + str(r2), 'best:' + str(result.Best)
    


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, BatchNormalization


def mlp(inputs, output_layer_activation, output_dim, output_use_bias,
        hidden_layer_num, hidden_layer_units_num, hidden_layer_activation, dropout,
        name=None, output_regularizer=None):
    if output_layer_activation == 'sigmoid' or output_layer_activation == 'tanh':
        hidden_layer_num -= 1
    x = inputs
    for l in range(hidden_layer_num):
        x = Dense(hidden_layer_units_num, activation=hidden_layer_activation)(inputs)
        x = Dropout(dropout)(x)
    if output_layer_activation == 'sigmoid' or output_layer_activation == 'tanh':
        x = Dense(hidden_layer_units_num)(x)

        x = tf.keras.layers.concatenate([x, inputs])
        x = Activation(hidden_layer_activation)(x)
        x = Dense(output_dim, use_bias=output_use_bias,
                  kernel_regularizer='l2', activity_regularizer=output_regularizer)(x)
        x = Activation(output_layer_activation, name=name)(x)
        return x
    x = Dense(output_dim, activation=output_layer_activation,
              kernel_regularizer='l2', activity_regularizer=output_regularizer,
              use_bias=output_use_bias, name=name)(x)
    return x