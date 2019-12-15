import vae_model_by_hzy as VAE
import tensorflow as tf
from keras.layers import Dense,Input
from keras.models import Model as Model
import parameters  as para
import keras as keras
import keras.losses as losses
from keras import backend as K





# LSTMLayer=Sequential()
# LSTMLayer.add(LSTM(para.decoder_rnn_size, return_sequences=True,return_state=True, name="decoder4"))
# LSTMLayer.add(LSTM(para.decoder_rnn_size, return_sequences=True,return_state=True, name="decoder4"))
vae,bigdecoder,docoder4=VAE.create_lstm_vae()
alpha=(tf.ones(shape=(32,25,1)))  


random_z=tf.convert_to_tensor(tf.ones(shape=(32,25,600)) *1.0)

x=tf.ones(shape=(32,25,825))
y=tf.ones(shape=(32,25,8250))

vae.train_on_batch([x,x,x,x,alpha,random_z],y)




