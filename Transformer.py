import tensorflow as tf
import keras
import numpy as np
import pandas as pd

def feed_forward_network(d_model,diff):
    # diff:dim of feed network
    return tf.keras.Sequential([
        tf.keras.layers.Dense(diff,activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


def scaled_dot_product_attention(q,k,v,mask):
    '''
    Args:
    -q : shape==(...,seq_len_q,depth)
    -k : shape==(...,seq_len_k,depth)
    -v : shape==(...,seq_len_v,depth_v)
    - seq_len_k = seq_len_v
    - mask: shape == (...,seq_len_q,seq_len_k)
    return:
    output:weighted sum
    attention_weights:weights of attention
    '''
    # shape == (...,seq_len_q,seq_len_k) 
    matmul_qk =tf.matmul(q,k,transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1],tf.float32)
    scaled_attention_logits =matmul_qk/tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask*-1e9)
    # shape == (...,seq_len_q,seq_len_k) 
    attention_weights =tf.nn.softmax(scaled_attention_logits,axis=-1)
    # shape==(...,seq_len_q,depth_v)
    output = tf.matmul(attention_weights,v)
    return output,attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
 
        assert d_model % self.num_heads == 0
 
        self.depth = d_model // self.num_heads
 
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
 
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
 
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
 
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
 
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
 
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
 
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
 
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
 
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
 
        return output, attention_weights

class EncoderLayer(tf.keras.layers.Layer):
    '''
    block:
    x->self.attention->add&normalize&dropout->feed_forward->add&normalize&dropout
    '''
    def __init__(self,d_model,num_heads,diff,rate =0.1):
        super(EncoderLayer,self).__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.ffn = feed_forward_network(d_model,diff)
        self.layer_norm1 =tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    
    def call(self,x,training,encoder_padding_mask):
        # x.shape:(batch_size,seq_len,dim=dmodel)
        # attn_shape:(batch_size,seq_len,d_model)
        attn_output,_ = self.mha(x,x,x,encoder_padding_mask)
        
        attn_output = self.dropout1(attn_output,training=training)
        out1 = self.layer_norm1(x+attn_output)
        ffn_output = self.ffn(out1)
        
        # ffn_output(batch_size,seq_len,d_model)
        # out2.shape:(batch_size,seq_len,d_model)
        out2 = self.layer_norm2(out1+ffn_output)
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    '''
    x->self.Attention->add&norm&dropout->out1
    out1,encoding_outputs->self.attention_>add&norm&dropput->out2
    out2->ffn->add&norm&dropout->out3
    '''
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
 
        self.mha1 = MultiHeadAttention(d_model, num_heads) # self attention
        self.mha2 = MultiHeadAttention(d_model, num_heads) 
 
        self.ffn = feed_forward_network(d_model, dff)
 
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
 
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    
    def call(self, x, enc_output, training, decoder_mask, encoder_decoder_padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
 
        attn1, attn_weights_block1 = self.mha1(x, x, x, decoder_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
 
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, encoder_decoder_padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
 
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
 
        return out3, attn_weights_block1, attn_weights_block2

def get_angles(pos,i,d_model):
    angle_rates = 1/np.power(10000,(2*(i//2))/np.float32(d_model))
    return pos *angle_rates
 
def get_position_embedding(sentence_length,d_model):
    angle_rads =get_angles(np.arange(sentence_length)[:,np.newaxis],
                          np.arange(d_model)[np.newaxis,:],
                          d_model)
    # sines.shape:[sentence_length,d_model/2]
    sines =np.sin(angle_rads[:,0::2])
    cosines = np.cos(angle_rads[:,1::2])
    # [sentence_length,d_model]
    position_embedding  = np.concatenate([sines,cosines],axis=-1)
    # [1,sentence_length,d_model]
    position_embedding = position_embedding[np.newaxis,...]# 这里是np的一个trick
    return tf.cast(position_embedding,dtype=tf.float32)

class EncoderModel(tf.keras.layers.Layer):
    def __init__(self,num_layers,max_length,d_model,num_heads,dff,rate=0.1):
        super(EncoderModel,self).__init__()
        self.d_model = d_model
        self.num_layers =num_layers
        self.max_length = max_length
        # self.embedding = tf.keras.layers.Embedding(input_vocab_size,self.d_model)
        # shape:(1,max_len,d_model)
        # self.position_embedding = get_position_embedding(max_length,self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.encoder_layers = [EncoderLayer(d_model,num_heads,dff,rate) for _ in range(self.num_layers)]
    
    def call(self,x,training,encoder_padding_mask):
        # x,shape:(batchsize,input_seq_len)
        input_seq_len = tf.shape(x)[1]
        # assert input_seq_len<=self.max_length
        tf.debugging.assert_less_equal(input_seq_len,self.max_length,"input_seq_len should be less or equal to self.max_length! ")
        # (batch_size,input_seq_len,d_model)
        # x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model,tf.float32)) 
        # x +=self.position_embedding[:,:input_seq_len,:]
#         x = self.dropout(x,training)
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x,training,encoder_padding_mask)
        # x.shape:(batch_size,input_seq_len,d_model)
        return x


class DecoderModel(tf.keras.layers.Layer):
    def __init__(self,num_layers,max_length,d_model,num_heads,dff,rate=0.1):
        super(DecoderModel,self).__init__()
        self.num_layers =num_layers
        self.max_length = max_length
        self.d_model = d_model
        # self.embedding = tf.keras.layers.Embedding(target_vocab_size,d_model)
        # self.position_embedding = get_position_embedding(max_length,d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.decoder_layers = [DecoderLayer(d_model,num_heads,dff,rate) for _ in range(self.num_layers)]
    
    def call(self,x,encoding_outputs,training,decoder_mask,encoder_decoder_padding_mask):
        # x.shape:(batch_size,output_seq_len)
        output_seq_len = tf.shape(x)[1]
        tf.debugging.assert_less_equal(output_seq_len,self.max_length,'output_seq_len should less or equal to self.max_length! ')
        attention_weights = {}
        # x.shape(batch_size,output_seq_len,d_model)
        # x =self.embedding(x)
        x*= tf.math.sqrt(tf.cast(self.d_model,tf.float32))
        # x+= self.position_embedding[:,:output_seq_len,:]
        # x = self.dropout(x,training)
        for i in range(self.num_layers):
            x ,att1,att2= self.decoder_layers[i](x,encoding_outputs,training,decoder_mask,encoder_decoder_padding_mask)
            attention_weights['decoder_layer{}_att1'.format(i+1)] = att1
            attention_weights['decoder_layer{}_att2'.format(i+1)] = att2
            # x.shape(batch_size,output_seq_len,d_model)
        return x,attention_weights

class Transformer(tf.keras.Model):
    def __init__(self,num_layers,target_vocab_size,max_length,d_model,num_heads,dff,rate=0.1):
        super(Transformer,self).__init__()
        
        self.encoder_model = EncoderModel(num_layers,max_length,d_model,num_heads,dff,rate)
        self.decoder_model = DecoderModel(num_layers,max_length,d_model,num_heads,dff,rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size) 
        
    
    def call(self,inp,tar,training,encoding_padding_mask=None,decoder_mask=None,encoder_decoder_padding_mask=None):
        # (batch_size,input_seq_len,d_model)
        encoding_outputs = self.encoder_model(inp,training,encoding_padding_mask)
        # decoding_outputs:(batch_size,output_seq_len,d_model)
        decoding_outputs,attention_weights = self.decoder_model(tar,encoding_outputs,training,decoder_mask,encoder_decoder_padding_mask)
        # batch_size,output_seq_len,target_vocab_size
        predictions = self.final_layer(decoding_outputs)
        return predictions,attention_weights