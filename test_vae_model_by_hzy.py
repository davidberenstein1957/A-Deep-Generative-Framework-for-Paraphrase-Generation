import tensorflow.keras
from tensorflow.keras import Model,losses
from tensorflow.keras.layers import Input, LSTM,Dense
from tensorflow.keras.utils import plot_model
import numpy as np
import tensorflow as tf
import vae_model_by_hzy as vae
import parameters as para
def test_bigEncoder():
    X_o_1 = Input(shape=(None, 2), name="X_o_1") #句子，词，词的embedding
    X_p_2 = Input(shape=(None, 2), name="X_p_2")
    encoder1 = LSTM(3, return_sequences=True,return_state=True, name="encoder1") #LSTM中间编码的维度，这里是
    encoder2 = LSTM(3, return_sequences=True,return_state=True, name="encoder2") 
    _, h, c = encoder1(X_o_1) 
    encoder_z, h, c = encoder2(X_p_2, initial_state=[h, c])
    bigEncoder = Model([X_o_1, X_p_2], [encoder_z,h,c])

    bigEncoder.summary()
    bigEncoder.compile(optimizer="adam",loss=losses.categorical_crossentropy)
    plot_model(bigEncoder, to_file='/home/hzy/Desktop/bigEncoder.png')


    a=tf.convert_to_tensor([[[1.,2],[1.,2]],
                            [[1.,2],[1.,2]]])
    b=tf.convert_to_tensor([[[1.,2],[1.,2],[1.,2]],
                            [[1.,2],[1.,2],[1.,2]]])
    [z,h,c]=bigEncoder([a,b])
    print(a)
    print(b)
    print(z)
    print(h)
    print(c)

def test_decoder4():    
    para.embedding_dim=2
    para.intermediate_dim=3
    h=tf.convert_to_tensor([[1.,2,3],
                        [1,2,3]])#两个句子，每个句子3维度lstm维度
    c=tf.convert_to_tensor([[1,2,3],
                        [1.,2,3]])#两个句子，每个句子3维度lstm维度
    x_p_4=tf.convert_to_tensor([
                                [[1.,2],[1,2],[1,2]],
                                [[1,2],[1,2],[1,2]]
                                ])#两个句子，每个句子三个词，每个词2维度embeding
    vae_z=tf.convert_to_tensor([
                                [[1.,2,3],[1,2,3],[1,2,3]],
                                [[1,2,3],[1,2,3],[1,2,3]]
                                ])#两个句子，每个句子三个词，每个词2维度zzzzz                            
    decoder4=vae.create_decoder4()
    decode_y,h1,c1=decoder4([h,c,vae_z,x_p_4])
    print(decode_y)
    print(h1)
    print(c1)

def test_create_bigDecoder():
    para.embedding_dim=2
    para.intermediate_dim=3
    h=tf.convert_to_tensor([[1.,2,3],
                        [1,2,3]])#两个句子，每个句子3维度lstm维度
    c=tf.convert_to_tensor([[1,2,3],
                        [1.,2,3]])#两个句子，每个句子3维度lstm维度
    x_p_4=tf.convert_to_tensor([
                                [[1.,2],[1,2],[1,2]],
                                [[1,2],[1,2],[1,2]]
                                ])#两个句子，每个句子三个词，每个词2维度embeding
    vae_z=tf.convert_to_tensor([
                                [[1.,2,3],[1,2,3],[1,2,3]],
                                [[1,2,3],[1,2,3],[1,2,3]]
                                ])#两个句子，每个句子三个词，每个词3维度zzzzz                            
    x_o_3=tf.convert_to_tensor([
                                [[1.,2],[1,2],[1,2]],
                                [[1,2],[1,2],[1,2]]
                                ])#两个句子，每个句子三个词，每个词2维度embeding
    bigDecoder,_=vae.create_bigDecoder()
    final_y=bigDecoder([x_o_3, x_p_4, vae_z])
    print(final_y)

def test_VAE_Z():
    encoder_z=tf.convert_to_tensor(tf.ones(shape=(32,25,1100)) *1.0)
    random_z=tf.convert_to_tensor(tf.ones(shape=(32,25,600)) *1.0)
    z1=tf.convert_to_tensor(tf.ones(shape=(32,25,600)) *1.0)
    kld=tf.convert_to_tensor(tf.ones(shape=(32,25)) *1.0)

    aaa1=vae.create_VAE_Z()
    aaa1.compile(optimizer="adam", loss="mae")
    aaa1.summary()
    aaa1.train_on_batch([encoder_z,random_z],[z1,kld])
    print("train finash")

test_VAE_Z()

'''


h = Input(shape=(None, 3), name="h")
c = Input(shape=(None, 3), name="c")
encoder_z = Input(shape=(None, 4), name="encoder_z")
x_p_4 = Input(shape=(None, 2), name="x_p_4")
decoder4 = LSTM(3, return_sequences=True,return_state=True, name="decoder4")
decode_y, h, c = decoder4([x_p_4, encoder_z], initial_state=[h, c])
dec_dense_layer = Dense(3,activation="softmax", name="Dec_Dense_Layer")
final_y=dec_dense_layer(decode_y)
decoder4model = Model([h, c, encoder_z, x_p_4], [final_y, h, c])
h=tf.convert_to_tensor([[1.,2,3],[1.,2,3]])
c=tf.convert_to_tensor([[1.,2,3],[1.,2,3]])
x_p_4=tf.convert_to_tensor([[[1.,2],[1.,2]],
                            [[1.,2],[1.,2]]])
z=tf.convert_to_tensor([[[1.,2,3],[1.,2,3]],
                            [[1.,2,3],[1.,2,3]]])
[z,h,c]=decoder4model([h,c,z,x_p_4])
print(z)
print(h)
print(c)
'''