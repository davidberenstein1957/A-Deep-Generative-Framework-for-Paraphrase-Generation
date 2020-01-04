# -*- coding: UTF-8 -*-
class para:
    def __init__(self, max_word_len, max_seq_len, word_vocab_size, char_vocab_size):
        #     max_word_length,max_sentence_length, 
        self.max_word_len = int(max_word_len)
        self.max_seq_len = int(max_seq_len) + 1  # go or eos token
        #   word_vocabulary_size,char_vocabulary_size
        self.word_vocab_size = int(word_vocab_size)
        self.char_vocab_size = int(char_vocab_size)
        self.word_embed_size = 300
        self.char_embed_size = 15
        # kernel  
        self.sum_depth=525
        self.kernels = [(1, 25), (2, 50), (3, 75), (4, 100), (5, 125), (6, 150)]
       
        self.encoder_rnn_size = 600
        self.encoder_num_layers = 1

        self.latent_variable_size = 1100

        self.decoder_rnn_size = 600
        self.decoder_num_layers = 2

hzy_token_embedding=825
hzy_lstm_dim=1100
hzy_xp_word_vocab_size=825
hzy_xo_word_vocab_size=8250

hzy_vae_dense_dim=600

word_vocab_size=1000000#????????????????????????

embedding_dim = 300
char_embed_size = 15



latent_dim=600

word_embed_size = 300
char_embed_size = 15
encoder_rnn_size = 600
encoder_num_layers = 1

latent_variable_size = 1100

decoder_rnn_size = 600
decoder_num_layers = 2

path="D:/Dropbox/temp/1130model/"