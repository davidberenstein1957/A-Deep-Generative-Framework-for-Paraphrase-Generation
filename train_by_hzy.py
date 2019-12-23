from utils.batch_loader import BatchLoader
import argparse
import os
from parameters import para,path
import torch as t
import vae_model_by_hzy as VAE
from embedding import Embedding as Embedding
from torch.autograd import Variable
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    if not os.path.exists(path+"data/word_embeddings.npy"):
        raise FileNotFoundError("word embeddings file was't found")
    if True :            
        parser = argparse.ArgumentParser(description='RVAE')
        parser.add_argument('--num-iterations', type=int, default=120000, metavar='NI',
                            help='num iterations (default: 120000)')
        parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                            help='batch size (default: 32)')
        parser.add_argument('--use-cuda', type=bool, default=True, metavar='CUDA',
                            help='use cuda (default: True)')
        parser.add_argument('--learning-rate', type=float, default=0.00005, metavar='LR',
                            help='learning rate (default: 0.00005)')
        parser.add_argument('--dropout', type=float, default=0.3, metavar='DR',
                            help='dropout (default: 0.3)')
        parser.add_argument('--use-trained', type=bool, default=False, metavar='UT',
                            help='load pretrained model (default: False)')
        parser.add_argument('--ce-result', default='', metavar='CE',
                            help='ce result path (default: '')')
        parser.add_argument('--kld-result', default='', metavar='KLD',
                            help='ce result path (default: '')')
        args = parser.parse_args()
    if True:
            
        ''' =================== Creating batch_loader for encoder-1 =========================================
        '''
        data_files = [path + 'data/train.txt',
                        path + 'data/test.txt']

        idx_files = [path + 'data/words_vocab.pkl',
                        path + 'data/characters_vocab.pkl']

        tensor_files = [[path + 'data/train_word_tensor.npy',
                            path + 'data/valid_word_tensor.npy'],
                            [path + 'data/train_character_tensor.npy',
                            path + 'data/valid_character_tensor.npy']]

        batch_loader = BatchLoader(data_files, idx_files, tensor_files, path)
        batch_loader.load_preprocessed(data_files, idx_files, tensor_files)
        parameters = para(batch_loader.max_word_len,
                                batch_loader.max_seq_len,
                                batch_loader.words_vocab_size,
                                batch_loader.chars_vocab_size)


        ''' =================== Doing the same for encoder-2 ===============================================
        '''
        data_files = [path + 'data/super/train_2.txt',
                        path + 'data/super/test_2.txt']

        idx_files = [path + 'data/super/words_vocab_2.pkl',
                        path + 'data/super/characters_vocab_2.pkl']

        tensor_files = [[path + 'data/super/train_word_tensor_2.npy',
                            path + 'data/super/valid_word_tensor_2.npy'],
                            [path + 'data/super/train_character_tensor_2.npy',
                            path + 'data/super/valid_character_tensor_2.npy']]
        batch_loader_2 = BatchLoader(data_files, idx_files, tensor_files, path)
        parameters_2 = para(batch_loader_2.max_word_len,
                                batch_loader_2.max_seq_len,
                                batch_loader_2.words_vocab_size,
                                batch_loader_2.chars_vocab_size)
        '''=================================================================================================
        '''
    embedding = Embedding(parameters, path)
    embedding_2 = Embedding(parameters_2, path,True)
    vae,bigdecoder,docoder4=VAE.create_lstm_vae()
    start_index = 0
    for iteration in range(args.num_iterations):
        if True:
            #This needs to be changed ##这一步必须保证不大于训练数据数量-每一批数据的大小，否则越界报错######################
            start_index =  (start_index+1)%(49999-args.batch_size)
            #start_index = (start_index+args.batch_size)%149163 #计算交叉熵损失，等 
            #=================================================== Input for Encoder-1,3 ========================================================
            input = batch_loader.next_batch(args.batch_size, 'train', start_index)
            input = [Variable(t.from_numpy(var)) for var in input]
            input = [var.long() for var in input]
            input = [var.cuda() if args.use_cuda else var for var in input]
            #这里是data/train.txt,转换变成embedding，用pand补齐， 
            #其中encoder_word_input, encoder_character_input是将 xo原始句输入倒过来前面加若干占位符， 
            # decoder_word_input, decoder_character_input是 xo原始句加了开始符号末端补齐
            # target，结束句子后面加了结束符，target是xo原始句加结束符后面加若干占位符
            [encoder_word_input, encoder_character_input, decoder_word_input, decoder_character_input, target] = input
            # =================================================== Input for Encoder-1,3 ========================================================
            ''' =================================================== Input for Encoder-2 ========================================================
            '''
            input_2 = batch_loader_2.next_batch(args.batch_size, 'train', start_index)
            input_2 = [Variable(t.from_numpy(var)) for var in input_2]
            input_2 = [var.long() for var in input_2]
            input_2 = [var.cuda() if args.use_cuda else var for var in input_2]           
            #这里是data/super/train.txt,转换变成embedding，用pand补齐， 
            #其中encoder_word_input_2, encoder_character_input_2是将 释义句xp输入倒过来前面加若干占位符， 
            # decoder_word_input_2, decoder_character_input是 释义句xp加了开始符号末端补齐
            # target，结束句子后面加了结束符，target是释义句xp加结束符后面加若干占位符
            [encoder_word_input_2, encoder_character_input_2, decoder_word_input_2, decoder_character_input_2, target] = input_2

            ''' =================================================== Input for Encoder-2 ============================================================        '''
            
        
        [batch_size, _] = encoder_word_input.size()
        encoder_input = embedding(encoder_word_input, encoder_character_input)
        [batch_size_2, _] = encoder_word_input_2.size()
        encoder_input_2 = embedding_2(encoder_word_input_2, encoder_character_input_2)

        #这里又几个变量 encoder_input 相当于x_o_1,x_o_3, encoder_input_2 相当于x_p_2,x_p_4
        [X_o_1,X_p_2,x_o_3,x_p_4]=[tf.convert_to_tensor(encoder_input.data),tf.convert_to_tensor(encoder_input_2.data),tf.convert_to_tensor(encoder_input.data),tf.convert_to_tensor(encoder_input_2.data)]
        alpha=(tf.ones(shape=(32,25,1)))  
        z=tf.ones(shape=(32,25,600)) 

        vae.train_on_batch([X_o_1,X_p_2,x_o_3,x_p_4,alpha,z],x_p_4)

        print("xunlian")


