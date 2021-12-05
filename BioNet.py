from Transformer import Transformer
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Reshape, Lambda, Permute, Flatten, Dropout
from tensorflow.keras.layers import Embedding, Concatenate, Add
from tensorflow.keras.layers import Conv1D, AveragePooling1D
from tensorflow.keras import Model, Input
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def transformer_ont(params):
    dropout_rate = params['dropout_rate']
    input = Input(shape=(83,))
    input_nuc = input[:, :24]
    input_dimer = input[:, 24:48]
    input_pos = input[:, 48:72]
    input_biofeat = input[:, 72:]

    embedded_nuc = Embedding(30, params['nuc_embedding_outputdim'], input_length=24)(input_nuc)
    embedded_dimer = Embedding(30, params['nuc_embedding_outputdim'], input_length=24)(input_dimer)

    conv1_nuc = Conv1D(params['conv1d_filters_num'], params['conv1d_filters_size'], padding='same', activation="relu", name="conv1_nuc")(embedded_nuc)
    conv1_dimer = Conv1D(params['conv1d_filters_num'], params['conv1d_filters_size'], padding='same', activation="relu", name="conv1_dimer")(embedded_dimer)

    pool1_nuc = AveragePooling1D(1, padding='same')(conv1_nuc)
    drop1_nuc = Dropout(dropout_rate)(pool1_nuc)
    pool1_dimer = AveragePooling1D(1, padding='same')(conv1_dimer)
    drop1_dimer = Dropout(dropout_rate)(pool1_dimer)

    pool_seq = Add()([pool1_nuc, pool1_dimer])
    drop_seq = Add()([drop1_nuc, drop1_dimer])

    emd_pos = Embedding(30, params['conv1d_filters_num'], input_length=24)(input_pos)

    pool1 = Add()([pool_seq, emd_pos])
    drop1 = Add()([drop_seq, emd_pos])

    pool1 = Add()([pool1_dimer, emd_pos])
    drop1 = Add()([drop1_dimer, emd_pos])

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

    x_bio = mlp(input_biofeat,
                output_layer_activation='tanh', output_dim=256, output_use_bias=True,
                hidden_layer_num=2, hidden_layer_units_num=150,
                hidden_layer_activation='relu', dropout=0.05,
                name='biofeat_embedding')

    output_bio = tf.keras.layers.concatenate([flat, x_bio])

    dense1 = Dense(params['dense1'],
                   kernel_regularizer=regularizers.l2(1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activation="relu",
                   name="dense1")(output_bio)
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


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback,LambdaCallback
from tensorflow.keras.optimizers import *
from hyperopt import STATUS_OK

def train(params,train_input,train_label,test_input,test_label,issave=False):
    result = Result()
    m = transformer_ont(params)
    batch_size = params['train_batch_size']
    learningrate = params['train_base_learning_rate']
    epochs = params['train_epochs_num']
    m.compile(loss='mse', optimizer=Adam(lr=learningrate))

    batch_end_callback = LambdaCallback(on_epoch_end=
                                        lambda batch,logs:
                                        print(get_score_at_test(m,test_input,result,test_label,
                                                                issave=issave,savepath=params['model_save_file'])))

    m.fit(train_input,train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_split=0.1,
          callbacks=[batch_end_callback])
    return {'loss': -1*result.Best, 'status': STATUS_OK}

if __name__ == "__main__":
    from ParamsDetail import Params as params
    import numpy as np
	import pandas as pd
	from utils import *
	from sklearn import preprocessing
	params = params['ModelParams']

    crispr = pd.read_csv("./eSpCas9_withbiofeat.csv")
	seq_column = 'Input_Sequence'
	# Sequence encoding with dimer
	nts = crispr.loc[:, seq_column].apply(
	                lambda seq: Dimer_split_seqs(seq[0:23]))
	nts = np.array(nts)
	# Biofeature
	bio_feature = crispr.iloc[:,2:]
	bio_feature = np.array(bio_feature)
	bio_feature = preprocessing.scale(bio_feature)

	nts = np.concatenate((nts,bio_feature),axis=1)

	print(nts)
	print(nts.shape)

	y_column = 'Indel_Norm'
	y = crispr[y_column]
	y = np.array(y)
	print(y.shape)

	from sklearn.model_selection import train_test_split

	random_state=40
	test_size = 0.15

	x_train, x_test, y_train, y_test = train_test_split(nts, y, test_size=test_size, random_state=random_state)
	print("=" * 10 + "x_train" +"=" * 10)
	print(x_train.shape)
	print("=" * 10 + "x_test" +"=" * 10)
	print(x_test.shape)
	print("=" * 10 + "y_train" +"=" * 10)
	print(y_train.shape)
	print("=" * 10 + "y_test" +"=" * 10)
	print(y_test.shape)

    train(params,x_train, y_train, x_test, y_test)