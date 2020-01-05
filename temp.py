import vae_model_by_hzy as VAE
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model as Model
import parameters  as para
import tensorflow.keras as keras
import tensorflow.keras.losses as losses
from tensorflow.keras import backend as K









# LSTMLayer=Sequential()
# LSTMLayer.add(LSTM(para.decoder_rnn_size, return_sequences=True,return_state=True, name="decoder4"))
# LSTMLayer.add(LSTM(para.decoder_rnn_size, return_sequences=True,return_state=True, name="decoder4"))
vae,bigdecoder,docoder4=VAE.create_lstm_vae_1()
alpha=(tf.ones(shape=(32,25,1)))  
random_z=tf.convert_to_tensor(tf.ones(shape=(32,25,600)) *1.0)
x=tf.ones(shape=(32,25,825))
y=tf.ones(shape=(32,25,825))

vae.train_on_batch([x,x,x,x,alpha,random_z],y)




