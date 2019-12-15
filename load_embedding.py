import numpy as np
import tensorflow as tf
import _pickle as cPickle
from embedding import Embedding as Embedding
import argparse
import torch as t


from parameters import para as para


class hzy_embedding():
                       #数据文件   id到词的索引文件   词的embedding文件
    def __init__(self, data_files, idx_files, tensor_files,unknow_word):      
      
        self.unknow_word=unknow_word
        data = [open(file, "r").read() for file in data_files]
        data_words = [[line.split() for line in target.split('\n')] for target in data]
        self.max_seq_len = np.amax([len(line) for target in data_words for line in target])
        self.num_lines = [len(target) for target in data_words]
        #单词id到单词
        [self.idx_to_word, self.idx_to_char] = [cPickle.load(open(file, "rb")) for file in idx_files]

        [self.words_vocab_size, self.chars_vocab_size] = [len(idx) for idx in [self.idx_to_word, self.idx_to_char]]
        #单词到单词id
        [self.word_to_idx, self.char_to_idx] = [dict(zip(idx, range(len(idx)))) for idx in
                                                [self.idx_to_word, self.idx_to_char]]
      
        self.max_word_len = np.amax([len(word) for word in self.idx_to_word])
        #这里存储的是train以及test里每句话对应的词的id
        [self.word_tensor, self.character_tensor] = [np.array([np.load(target,allow_pickle=True) for target in input_type])
                                                        for input_type in tensor_files] 
        #整个词表
        self.just_words = [word for line in self.word_tensor[0] for word in line] 
    
    def onehot_word(self, idx):
        result = np.zeros(self.words_vocab_size)
        result[idx] = 1
        return result
    def get_wordid_from_one_word(self,word):
        if self.word_to_idx.get(word)==None:
            return self.word_to_idx.get(self.unknow_word)
        return self.word_to_idx.get(word)
    def get_charid_from_one_char(self,word):
        if self.char_to_idx.get(word)==None:
           return self.char_to_idx.get(self.unknow_word)
        return self.char_to_idx.get(word)

    # def get_wordembeding_from_one_word(self,word):  
    #     wordid=self.get_wordid_from_one_word(word)
        
    #     return self.word_tensor[wordid]

    def get_wordids_from_wordid_list(self,wordList):
        result=[]
        wordlist=wordList.split(" ")
        for word in enumerate(wordList):
            result.append(self.get_wordid_from_one_word(word))
        return tf.convert_to_tensor(result)
    def get_Embedding_from_batchsize_seqLen_wordid(self, word_input):
        [batch_size, seq_len] = word_input.size()        
        word_input = word_input.view(-1)   
        return result

class load_hzy_embedding():
    def __init__(self,path):
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

        self.args = parser.parse_args()

       
        self.blind_symbol = ''
        self.pad_token = '_'
        self.go_token = '>'
        self.end_token = '|'
        self.a_token = '?'
            
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


        self.hzy_embedding1=hzy_embedding(data_files,idx_files,tensor_files,"未登录词应该表示成什么？")#########需要处理需要处理需要处理需要处理需要处理##########################################################需要处理
        parameters = para(self.hzy_embedding1.max_word_len,
                                self.hzy_embedding1.max_seq_len,
                                self.hzy_embedding1.words_vocab_size,
                                self.hzy_embedding1.chars_vocab_size)
       

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


        self.hzy_embedding2=hzy_embedding(data_files,idx_files,tensor_files,"未登录词应该表示成什么？")#########需要处理需要处理需要处理需要处理需要处理##########################################################需要处理
        parameters_2 = para(self.hzy_embedding2.max_word_len,
                                self.hzy_embedding2.max_seq_len,
                                self.hzy_embedding2.words_vocab_size,
                                self.hzy_embedding2.chars_vocab_size)
        
        '''=================================================================================================
        '''
        self.embedding = Embedding(parameters, path)
        self.embedding_2 = Embedding(parameters_2, path,True)

    def get_wordembeding_from_one_word(self,word,xo_or_xp):
        if xo_or_xp=="xo":
            wordid=self.hzy_embedding1.get_wordid_from_one_word(word)
            the_word=[[wordid]]
            encoder_word_input=t.tensor(the_word)
            charlist=[]

            for char in word:
                charlist.append(self.hzy_embedding1.get_charid_from_one_char(char))
            encoder_character_input=[[charlist]]
          
            for i, line in enumerate(encoder_character_input):#把输入的句子倒过来 前面补齐pad-token             
                encoder_character_input[i] = line[::-1]
            encoder_character_input=t.tensor(encoder_character_input)
            embedding=self.embedding(encoder_word_input,encoder_character_input)
            #:param word_input: [batch_size, seq_len] tensor of Long type
            #:param character_input: [batch_size, seq_len, max_word_len] tensor of Long type
    
            print(embedding)


        #, 
        # #np.array(decoder_character_input), 
        # if xo_or_xp=="xp":
        #     wordid=self.hzy_embedding1.get_wordid_from_one_word(word)
        #     the_word=[[wordid]]
        #     encoder_word_input=t.tensor(the_word)
        #     charlist=[]
        #     for char in word:
        #         charlist.append(self.hzy_embedding1.get_charid_from_one_char(char))
        #     encoder_character_input=([[charlist]])
        #     input_seq_len = [len(line) for line in encoder_word_input]              
        #     wordid=self.hzy_embedding2.get_wordid_from_one_word(word)
        #     the_word=[[wordid]]
        #     for i, line in enumerate(decoder_character_input):# 后面补齐pad-token
        #         line_len = input_seq_len[i]
        #         to_add = max_input_seq_len - line_len
        #         #decoder_character_input[i] = line + [self.encode_characters(self.pad_token)] * to_addmax_input_seq_len = np.amax(input_seq_len)
        #         #decoder_character_input = [[hzy_embedding1.get_charid_from_one_char(self.go_token)] + line for line in encoder_character_input] #输入端加上开始标志    





path="/home/hzy/Desktop/hzyparaphraseGen/1130model/"
aaa=load_hzy_embedding(path)
aaa.get_wordembeding_from_one_word("you","xo")

