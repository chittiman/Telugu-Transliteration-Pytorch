import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from . import transliteration_tokenizers

class Simple_seq2seq(nn.Module):
    """ Simple seq2seq based Transliteration model:
        - Bidirectional GRU Encoder
        - Unidirectional GRU Decoder
    """
    def __init__(self, embed_size, hidden_size, src_tokenizer ,tgt_tokenizer ):
        """ 
        Initialize seq2seq model
        @param embed_size(int): Embedding size (dimensionality)
        @param hidden_size(int): Hidden size of GRU
        @param src_vocab(Vocab): Source Vocabulary
        
        """
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_vocab_size = self.src_tokenizer.get_vocab_size()
        self.tgt_vocab_size = self.tgt_tokenizer.get_vocab_size()
        self.src_pad_id = self.src_tokenizer.token_to_id('<pad>')
        self.tgt_pad_id = self.tgt_tokenizer.token_to_id('<pad>')
        self.tgt_start_id = self.tgt_tokenizer.token_to_id('<s>')
        self.tgt_end_id = self.tgt_tokenizer.token_to_id('</s>')
        
        self.src_embedding = nn.Embedding(self.src_vocab_size,self.embed_size, padding_idx = self.src_pad_id)
        self.encoder = nn.GRU(self.embed_size, self.hidden_size, batch_first = True, bidirectional = True)
        self.encode_hid2decode_hid_init = nn.Linear(2*self.hidden_size, self.hidden_size,bias = False)
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size,self.embed_size, padding_idx = self.tgt_pad_id)
        self.decoder = nn.GRU(self.embed_size, self.hidden_size, batch_first = True)
        self.decode2vocab = nn.Linear(self.hidden_size, self.tgt_vocab_size,bias = False)
        
    def forward(self,batch):
        """ 
        For a given batch of source sentences, calculate scores of words in target vocabulary 
        
        @param batch(tuple): tuple of source sentences, target sentences, source sentence lengths, target sentence lengths
        
        @returns decode_vocab_scores(Tensor): Final scores of vocabulary words whose shape (batch, tgt_seq_len-1,tgt_vocab_size)
        """
        
        src_sents,tgt_sents,src_lens = batch
        del batch
        decode_hidden_init = self.encode(src_sents,src_lens)
        del src_sents,src_lens
        decode_vocab_scores = self.decode(tgt_sents,decode_hidden_init)
        del tgt_sents,decode_hidden_init
        return decode_vocab_scores
        
    def encode(self,src_sents,src_lens):
        """
        Apply the encoder to source words, obtain encoder hidden states and transform them to hidden states 
        which are used to initialize the decoder

        @param src_sents(Tensor) : Tensor of padded source words of shape (batch,src_seq_len)
        @param src_lens(list): Source word lengths
        
        @returns encoding(Tensor): returns encoding of words of shape(1,batch,hidden_size) which will be fed as 
        hidden state for decoder
                       
        src_seq_len varies with batch. Sents are ordered in descending length
        
        """
        batch_size,src_seq_len = src_sents.shape
        src_embeds = self.src_embedding(src_sents)
        del src_sents
        packed_embeds = pack_padded_sequence(src_embeds, src_lens, batch_first=True, enforce_sorted=True)
        del src_embeds,src_lens
        _,encode_hidden = self.encoder(packed_embeds)
        del packed_embeds
        encode_hidden_unpacked = encode_hidden.view(1, 2, batch_size, -1)
        del encode_hidden
        encode_fwd,encode_bwd = encode_hidden_unpacked[:,0,:,:].squeeze(0),encode_hidden_unpacked[:,1,:,:].squeeze(0)
        del encode_hidden_unpacked
        encode_concat = torch.cat([encode_fwd,encode_bwd],dim =1)
        del encode_fwd,encode_bwd
        decode_hidden_init = self.encode_hid2decode_hid_init(encode_concat).unsqueeze(dim = 0)
        del encode_concat
        return decode_hidden_init
    
    def decode(self,tgt_sents,decode_hidden_init):
        """
        Using encoding to initialze decoder hidden state, predict the scores for target vocabulary words

        @param tgt_sents(Tensor) : Tensor of padded target sentences of shape (batch,tgt_seq_len)
        @param tgt_lens(list): Target sentence lengths
        @param decode_hidden_init(Tensor): Final sents encoding of shape (1,batch,hidden_size) used to initialize decoder 
        hidden state
        
        @returns decode_vocab_scores(Tensor): Final scores of vocabulary words whose shape 
        (batch, tgt_seq_len-1,tgt_vocab_size)
        
        tgt_seq_len varies with batch. 
              
        """
        tgt_inputs = tgt_sents[:,:-1]
        del tgt_sents#last word need not be fed into decoder
        tgt_embeds = self.tgt_embedding(tgt_inputs)
        del tgt_inputs
        decode_out,decode_hidden = self.decoder(tgt_embeds,decode_hidden_init)
        del tgt_embeds,decode_hidden_init,decode_hidden
        decode_vocab_scores = self.decode2vocab(decode_out)
        del decode_out
        return decode_vocab_scores
    
    def decode_step(self,tgt_word_in,decode_hidden_in):
        """
        Take input words and perform one decoding step
        @param tgt_word_in(Tensor): Tensor of words input at a given step of shape (batch,1)
        @param decode_hidden_in(Tensor): Decoder Hidden state input of shape(1,batch,hidden_size)
        
        @returns tgt_word_out(Tensor): Tensor of words output at a given step of shape (batch,1)
        @returns decode_hidden_out(Tensor) : Decoder Hidden state output of shape(1,batch,hidden_size)
        """
        tgt_embeds_in = self.tgt_embedding(tgt_word_in)
        del tgt_word_in
        decode_out, decode_hidden_out = self.decoder(tgt_embeds_in,decode_hidden_in)
        del tgt_embeds_in,decode_hidden_in
        decode_vocab_scores = self.decode2vocab(decode_out)
        del decode_out
        tgt_words = torch.argmax(decode_vocab_scores,dim = -1)
        return tgt_words,decode_hidden_out,decode_vocab_scores
    
class Attention_seq2seq(nn.Module):
    """ Attention based seq2seq Translation model:
        - Bidirection LSTM Encoder
        - Unidirection LSTM Decoder
    """
    def __init__(self, embed_size, hidden_size, src_tokenizer ,tgt_tokenizer,dropout_rate = 0.2):
        """ 
        Initialize seq2seq model
        @param embed_size(int): Embedding size (dimensionality)
        @param hidden_size(int): Hidden size of GRU
        @param src_vocab(Vocab): Source Vocabulary
        @param dropout_rate(float): Dropout rate ..............
        
        """
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_vocab_size = self.src_tokenizer.get_vocab_size()
        self.tgt_vocab_size = self.tgt_tokenizer.get_vocab_size()
        self.src_pad_id = self.src_tokenizer.token_to_id('<pad>')
        self.tgt_pad_id = self.tgt_tokenizer.token_to_id('<pad>')
        self.tgt_start_id = self.tgt_tokenizer.token_to_id('<s>')
        self.tgt_end_id = self.tgt_tokenizer.token_to_id('</s>')
        self.dropout_rate = dropout_rate
        self.attention_softmax = nn.LogSoftmax(dim = 1)
        
        self.src_embedding = nn.Embedding(self.src_vocab_size,self.embed_size, self.src_pad_id)
        self.encoder = nn.LSTM(self.embed_size, self.hidden_size, batch_first = True, bidirectional = True)
        self.encode_hid2decode_hid_init = nn.Linear(2*self.hidden_size, self.hidden_size, bias = False)
        self.encode_cell2decode_cell_init = nn.Linear(2*self.hidden_size, self.hidden_size, bias = False)
        
        self.tgt_embedding = nn.Embedding(self.tgt_vocab_size, self.embed_size, self.tgt_pad_id)
        self.decoder = nn.LSTM(self.embed_size + self.hidden_size, self.hidden_size, batch_first = True)
        self.attention_projection = nn.Linear(2*self.hidden_size,self.hidden_size,bias = False)
        self.combined_output_projection = nn.Linear(3*self.hidden_size, self.hidden_size,bias = False)
        self.decode2vocab_projection = nn.Linear(self.hidden_size,self.tgt_vocab_size,bias = False)
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self,batch):
        src_sents,tgt_sents,src_lens = batch
        del batch
        encoder_hidden_states,decode_hidden_init,decode_cell_init,src_mask = self.encode(src_sents,src_lens)
        del src_sents,src_lens
        target_vocab_scores = self.decode(encoder_hidden_states,decode_hidden_init,decode_cell_init,src_mask,tgt_sents)
        del encoder_hidden_states,decode_hidden_init,decode_cell_init,src_mask,tgt_sents
        return target_vocab_scores
        
        
    def encode(self,src_sents,src_lens):
        """
        Apply the encoder to source sentences, obtain encoder hidden states and transform them to hidden states 
        which are used to initialize the decoder

        @param src_sents(Tensor) : Tensor of padded source sentences of shape (batch,src_seq_len)
        @param src_lens(list): Source sentence lengths
        
        @returns encoding(Tensor): returns encoding of sentences of shape(1,batch,hidden_size) which will be fed as hidden state for decoder
                       
        src_seq_len varies with batch. Sents are ordered in descending length
        
        """
        batch_size,src_seq_len = src_sents.shape
        src_mask = self.generate_masks(src_sents)
        src_embeds = self.src_embedding(src_sents)#(batch_size, src_seq_len, embedding_size)
        del src_sents
        packed_embeds = pack_padded_sequence(src_embeds, src_lens, batch_first=True, enforce_sorted=True)#Packed_sequence
        del src_embeds,src_lens
        hidden_states_packed,(encode_hidden, encode_cell) = self.encoder(packed_embeds)
        del packed_embeds
        hidden_states,src_lens_unpacked = pad_packed_sequence(hidden_states_packed,batch_first = True)
        del hidden_states_packed,src_lens_unpacked
        
        encode_hidden_unpacked = encode_hidden.view(1, 2, batch_size, -1)
        del encode_hidden
        encode_hidden_fwd,encode_hidden_bwd = encode_hidden_unpacked[:,0,:,:].squeeze(0),encode_hidden_unpacked[:,1,:,:].squeeze(0)
        del encode_hidden_unpacked
        encode_hidden_concat = torch.cat([encode_hidden_fwd,encode_hidden_bwd],dim =1)
        del encode_hidden_fwd,encode_hidden_bwd
        decode_hidden_init = self.encode_hid2decode_hid_init(encode_hidden_concat).unsqueeze(dim = 0)
        del encode_hidden_concat
        
        encode_cell_unpacked = encode_cell.view(1, 2, batch_size, -1)
        del encode_cell
        encode_cell_fwd,encode_cell_bwd = encode_cell_unpacked[:,0,:,:].squeeze(0),encode_cell_unpacked[:,1,:,:].squeeze(0)
        del encode_cell_unpacked
        encode_cell_concat = torch.cat([encode_cell_fwd,encode_cell_bwd],dim =1)
        del encode_cell_fwd,encode_cell_bwd
        decode_cell_init = self.encode_cell2decode_cell_init(encode_cell_concat).unsqueeze(dim = 0)
        del encode_cell_concat
        
        return hidden_states,decode_hidden_init,decode_cell_init,src_mask
    
    def generate_masks(self,src_sents):
        mask = (src_sents == self.src_pad_id).float().unsqueeze(dim = -1)
        return mask
        
    
    def decode(self, encoder_hidden_states,decode_hidden_init,decode_cell_init,src_mask,tgt_sents):
        """
        Using encoding to initialze decoder hidden state, predict the scores for target vocabulary words

        @param tgt_sents(Tensor) : Tensor of padded target sentences of shape (batch,tgt_seq_len)
        @param tgt_lens(list): Target sentence lengths
        @param decode_hidden_init(Tensor): Final sents encoding of shape (1,batch,hidden_size) used to initialize decoder hidden state
        
        @returns decode_vocab_scores(Tensor): Final scores of vocabulary words whose shape (batch, tgt_seq_len-1,tgt_vocab_size)
        
        tgt_seq_len varies with batch. 
              
        """
        batch_size,tgt_seq_len = tgt_sents.shape
        tgt_inputs = tgt_sents[:,:-1] #()last word need not be fed into decoder
        del tgt_sents
        tgt_embeds = self.tgt_embedding(tgt_inputs)#(batch_size,tgt_seq_len - 1, embed_size)
        del tgt_inputs
        combined_output_init = torch.zeros((batch_size,1, self.hidden_size)) #(batch_size, 1, hidden_size)
        decode_hidden,decode_cell,combined_output = decode_hidden_init,decode_cell_init,combined_output_init
        del decode_hidden_init,decode_cell_init,combined_output_init
        target_vocab_scores = torch.empty(size = (batch_size,0,self.tgt_vocab_size ))
        
        for tgt_embeds_step in torch.split(tgt_embeds, 1, dim = 1):
            combined_output,vocab_scores,decode_hidden,decode_cell = \
            self.decode_step(tgt_embeds_step,combined_output,decode_hidden,decode_cell,encoder_hidden_states,src_mask)
            target_vocab_scores = torch.cat([target_vocab_scores,vocab_scores], dim = 1 )
        
        return target_vocab_scores
          
            
    def decode_step(self, tgt_embeds_input, combined_output_prev, prev_hidden,prev_cell, encoder_hidden_states,src_mask):
        """
        
        """
        decoder_input = torch.cat([combined_output_prev, tgt_embeds_input], dim = 2)
        del combined_output_prev, tgt_embeds_input
        decoder_output, (next_hidden,next_cell) = self.decoder(decoder_input, (prev_hidden,prev_cell))
        del decoder_input,prev_hidden,prev_cell
        encoding_projections = self.attention_projection(encoder_hidden_states)
        dec_hidden_reshaped = next_hidden.permute((1,2,0))
        attention_outputs = torch.bmm(encoding_projections, dec_hidden_reshaped)
        del encoding_projections, dec_hidden_reshaped
        attention_masked = attention_outputs.masked_fill_(src_mask, -float("Inf"))
        del attention_outputs
        attention_scores = self.attention_softmax(attention_masked)
        del attention_masked
        attention_output = torch.sum(attention_scores*encoder_hidden_states, 1, keepdim=True)
        del attention_scores,encoder_hidden_states
        attention_hidden_concat = torch.cat([attention_output,decoder_output],dim = 2)
        del attention_output,decoder_output
        raw_combined = self.combined_output_projection(attention_hidden_concat)
        del attention_hidden_concat
        combined_output = self.dropout(torch.tanh(raw_combined))
        del raw_combined
        vocab_scores = self.decode2vocab_projection(combined_output)
        
        return combined_output,vocab_scores,next_hidden,next_cell    
    
