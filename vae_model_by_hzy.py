# coding: utf-8
import keras
import os
from keras import backend as K
from keras import objectives
from keras.layers import Input, LSTM, Embedding, RepeatVector
from keras.layers.core import Dense, Lambda
from keras.layers.wrappers import TimeDistributed
from keras.models import Model
from keras.utils import plot_model
import parameters as para
import numpy as np
#from pg_vae.glove_embedding import load_golve_embedding
import parameters as para
import tensorflow as tf
import keras.losses as losses
import torch as t


#############################################################################################################################################################
#                                                                
#                                                                                       final_y   answerY
#                                                                 x_o_3                    |
#                                                               encoder3------h,c-------decoder4
#     (hzy_lstm_dim)                                                            |      |
# encoder1--h,c--encoder2-------------encoder_z--->vae------------------------->vae_z*单词数  x_p_4
# x_o_1          x_p_2
#
#
#         bigEncoder                        VAE_Z                          bigDecoder
#
#
#############################################################################################################################################################
# 1  定义整个encoder端 输入x_o x_p 输出最后末端的输出的z，h和c不要了 z的维度是《句子，词(数量和x_p_2一样)，hzy_lstm_dim维》
def create_bigEncoder():
    X_o_1 = Input(shape=(None, para.hzy_token_embedding), name="X_o_1") #《句子，词，token_embedding》
    X_p_2 = Input(shape=(None, para.hzy_token_embedding), name="X_p_2")
    encoder1 = LSTM(para.hzy_lstm_dim, return_sequences=True,return_state=True, name="encoder1") #LSTM中间编码的维度，这里是
    encoder2 = LSTM(para.hzy_lstm_dim, return_sequences=True,return_state=True, name="encoder2") 
    _, h, c = encoder1(X_o_1) 
    encoder_z, _, _ = encoder2(X_p_2, initial_state=[h, c])
    bigEncoder = Model([X_o_1, X_p_2], [encoder_z],name="bigEncoder")
    return bigEncoder

# 2--1 定义decoder端的解码器端 输入 h，c，维度都是《句子，hzy_lstm_dim维》，
def create_decoder4():  # bigDecoder的decoder4
    h = Input(shape=(para.hzy_lstm_dim,), name="h") #《句子，hzy_lstm_dim维》
    c = Input(shape=(para.hzy_lstm_dim,), name="c") #《句子，hzy_lstm_dim维》
    vae_z = Input(shape=(None,para.hzy_vae_dense_dim), name="encoder_z")#《句子，词，hzy_lstm_dim维》？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
    x_p_4 = Input(shape=(None,para.hzy_token_embedding), name="x_p_4") #《句子，词，token_embedding》

    decoder4 = LSTM(para.hzy_lstm_dim, return_sequences=True,return_state=True, name="decoder4")
    dec_concate =keras.layers.concatenate( [x_p_4, vae_z], axis=-1)    
    
   
    decode_y, h1, c1= decoder4(dec_concate, initial_state=[h, c]) #送进去的是《句子，词，token_embedding+z的hzy_lstm_dim维》》
    
    #这里应该把解码结果拉成二维，然后再变回三维?????????????????????????????????????????????????????????????????????????????????
    dec_dense_layer = Dense(para.hzy_xp_word_vocab_size,activation="softmax", name="dec_dense_layer") 
    final_y=dec_dense_layer(decode_y)
    decoder4model = Model([h, c, vae_z, x_p_4], [final_y,h1,c1],name="decoder4")
     
    return decoder4model

    # 2-2 定义整个decoder端
# 2--2 定义整个解码端
def create_bigDecoder():
    x_o_3 = Input(shape=(None, para.hzy_token_embedding), name="x_o_3")#《句子，词，token_embedding》
    # 定义decoder端的编码器端
    encoder3 = LSTM(para.hzy_lstm_dim, return_sequences=True,return_state=True, name="encoder3")
    _, h, c = encoder3(x_o_3)

    decoder4 = create_decoder4()
    x_p_4 = Input(shape=(None, para.hzy_token_embedding), name="x_p_4")#《句子，词，token_embedding》
    vae_z = Input(shape=(None,para.hzy_vae_dense_dim), name="z_from_encode")  #《句子，词，z para.hzy_lstm_dim》 
    final_y, _, _ = decoder4([h, c, vae_z, x_p_4])
    bigDecoder = Model([x_o_3, x_p_4, vae_z], [final_y],name="bigDecoder")
    return bigDecoder,decoder4,encoder3


# 3 定义VAE Z层 
def create_VAE_Z():
    
    encoder_z = Input(shape=(None, para.hzy_lstm_dim), name="z_from_encode")
    random_z=Input(shape=(None,para.hzy_vae_dense_dim),name="random_z") 

    context_to_mu = Dense(units=para.hzy_vae_dense_dim,name="Lstm_Layer_z_mean")
    context_to_logvar = Dense(units=para.hzy_vae_dense_dim,name="Lstm_Layer_z_log_sigma")  
    
    mu = context_to_mu(encoder_z)
    logvar = context_to_logvar(encoder_z)
    
    def get_z(x):
        mu1,logvar1=x
        std =keras.backend.exp(0.5 * logvar1)           
        z2 = random_z#keras.backend.random_normal(shape=(para.time_steps, para.latent_dim), mean=0., stddev=1.0)
        z3 = z2 * std + mu1
        return z3
   
    z1 = keras.layers.Lambda(get_z, output_shape=(para.hzy_vae_dense_dim,), name="Lambda_Layer_sampling_z")([mu, logvar])
    #z1 = keras.layers.RepeatVector(para.time_steps)(z1)
  

    def compute_kl_loss(x):
        mu1,logvar1=x
        kl_loss1 = - 0.5 * K.mean(1 + logvar1 - K.square(mu1) - K.exp(logvar1))
        return kl_loss1
    #kl_loss= keras.layers.Lambda(lambda x:x)(kl_loss)
    #这个是源代码的kld = (-0.5 * t.sum(logvar - t.pow(mu, 2) - t.exp(logvar) + 1, 1)).mean().squeeze() ############################################################
    #############################这里和论文不一样，论文是开根号，源代码里是平方#################划重点#################################################################
    ###################################################划重点#################################################################
    ###################################################划重点#################################################################
    kld=keras.layers.Lambda(compute_kl_loss,output_shape=(para.hzy_vae_dense_dim,), name="compute_kl_loss")([mu, logvar])
    print(kld)
    print("**************************************************")
    zmodel=Model([encoder_z,random_z],[z1,kld],name="VAE") 
    return zmodel


def create_lstm_vae():

    # 4  定义整个VAE
	#定义输入
    X_o_1 = Input(shape=(None, para.hzy_token_embedding), name="X_o_1")
    X_p_2 = Input(shape=(None, para.hzy_token_embedding), name="X_p_2")
    x_o_3 = Input(shape=(None, para.hzy_token_embedding), name="x_o_3")
    x_p_4 = Input(shape=(None, para.hzy_token_embedding), name="x_p_4")
    random_z=Input(shape=(None,para.hzy_vae_dense_dim),name="random_z") #???????
    alpha =Input(shape=(None,1), name="alpha")
    #编码
    bigEncoder = create_bigEncoder()   
    encoder_z=bigEncoder([X_o_1, X_p_2])
    #生成z

    vae_z_model=create_VAE_Z()	
    [vae_z,kld]=vae_z_model([encoder_z, random_z])
    
    #vae_z=RepeatVector(para.time_steps)(vae_z)
    
    #解码
    bigDecoder,decoder4,encoder3 =create_bigDecoder()
    final_y=bigDecoder([x_o_3, x_p_4, vae_z])
   
    #定义损失函数
    def vae_loss(y, y_predict):
        xent_loss = objectives.categorical_crossentropy(y, y_predict)        
        loss = 79 * xent_loss + alpha * kld
        return loss
    #生成模型
    vae=Model([X_o_1,X_p_2,x_o_3,x_p_4,alpha,random_z],final_y,name="VAE_G") 

    vae.compile(optimizer="adam", loss=vae_loss, metrics=["accuracy"])
    bigEncoder.compile(optimizer="adam", loss=vae_loss, metrics=["accuracy"])
    bigDecoder.compile(optimizer="adam", loss=vae_loss, metrics=["accuracy"])
    decoder4.compile(optimizer="adam", loss=vae_loss, metrics=["accuracy"])
    vae.summary()
    bigEncoder.summary()
    bigDecoder.summary()
    decoder4.summary()
    plot_model(vae, to_file='/home/hzy/Desktop/vae.png')
    plot_model(bigEncoder, to_file='/home/hzy/Desktop/bigEncoder.png')
    plot_model(bigDecoder, to_file='/home/hzy/Desktop/bigDecoder.png')
    plot_model(decoder4, to_file='/home/hzy/Desktop/decoder4.png')
    return vae, bigDecoder, decoder4
