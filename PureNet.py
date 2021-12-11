import numpy as np
import pandas as pd
from Transformer import Transformer
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Reshape, Lambda, Permute, Flatten, Dropout
from tensorflow.keras.layers import Embedding, Concatenate, Add
from tensorflow.keras.layers import Conv1D, AveragePooling1D
from tensorflow.keras import Model, Input
import tensorflow as tf
import os
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def transformer_ont(params):
    dropout_rate = params['dropout_rate']
    input = Input(shape=(48,))
    input_nuc = input[:, :24]
    input_pos = input[:, 24:]

    embedded_nuc = Embedding(30, params['nuc_embedding_outputdim'], input_length=24)(input_nuc)
    conv1_nuc = Conv1D(params['conv1d_filters_num'], params['conv1d_filters_size'], padding='same', activation="relu", name="conv1_nuc")(embedded_nuc)

    pool1_nuc = AveragePooling1D(1, padding='same')(conv1_nuc)
    drop1_nuc = Dropout(dropout_rate)(pool1_nuc)

    emd_pos = Embedding(30, params['conv1d_filters_num'], input_length=24)(input_pos)
    pool1 = Add()([pool1_nuc, emd_pos])
    drop1 = Add()([drop1_nuc, emd_pos])

    conv2 = Conv1D(params['conv1d_filters_num'], params['conv1d_filters_size'], activation="relu", name="conv2")(pool1)
    conv3 = Conv1D(params['conv1d_filters_num'], params['conv1d_filters_size'], activation="relu", name="conv3")(drop1)

    sample_transformer = Transformer(params['transformer_num_layers'], params['transformer_final_fn'], 50, params['conv1d_filters_num'], 8, params['transformer_ffn_1stlayer'], rate=0.1)
    x, attention_weights = sample_transformer(conv3, conv2, training=False, encoding_padding_mask=None,
                                              decoder_mask=None, encoder_decoder_padding_mask=None)

    my_concat = Lambda(lambda x: Concatenate(axis=1)([x[0], x[1]]))
    weight_1 = Lambda(lambda x: x * 0.2)
    weight_2 = Lambda(lambda x: x * 0.8)

    flat1 = Flatten()(pool1)
    flat2 = Flatten()(x)
    flat = my_concat([weight_1(flat1), weight_2(flat2)])

    dense1 = Dense(params['dense1'],
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu",
                   name="dense1")(flat)
    drop3 = Dropout(dropout_rate)(dense1)

    dense2 = Dense(params['dense2'],
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu",
                   name="dense2")(drop3)
    drop4 = Dropout(dropout_rate)(dense2)

    dense3 = Dense(params['dense3'], activation="relu", name="dense3")(drop4)
    drop5 = Dropout(dropout_rate)(dense3)

    output = Dense(1, activation="linear", name="output")(drop5)

    model = Model(inputs=[input], outputs=[output])
    return model


if __name__ == "__main__":
    # train(params,x_train, y_train, x_test, y_test)
    from ParamsDetail import Params as params
    params = params['ModelParams']
    model = transformer_ont(params)

    print("Loading weights")
    model.load_weights("models/BestModel_xCas.h5")
        
    data_path = "./testsets/test_xCas9.csv"
    data = pd.read_csv(data_path)
    seq_column = 'Input_Sequence'
    nts = data.loc[:, seq_column].apply(
                    lambda seq: split_seqs(seq[0:23]))
    x_test = np.array(nts)
    y_pred = model.predict([x_test])

    print(y_pred)

