import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self , ds , tokenizer_src ,tokenizer_tgt ,src_lang ,tgt_lang ,seq_len):
        super().__init__()

        self.ds =ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang  = tgt_lang 
        """
        to make the all tensor are the same size of the padding ex( 24 ,20,8) --> if we use the seq_len 5 
        then 5 - 3 =2 so we add the two somthing maaybe to 0  ( 24 ,20,8 ,0 ,0)
        """
        self.sos_token =torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])],dtype = torch.int64)
        self.eos_token =torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])],dtype = torch.int64)
        self.pad_token =torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])],dtype = torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair =self.ds[index]
        src_text = src_target_pair['transulation'][self.src_lang]
        tgt_text = src_target_pair['transulation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.decode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens <0:
            raise ValueError('sentence is too long')

        #add sos and eos to the sourse text 
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens , dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens , dtype=torch.int64)
             
                
            ]
        )
        
        #adding soos  to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens,dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype=torch.int64)
             

            ]
        )

        #the labe is the decoder output whihc means the transformer output

        #add eos to the lable 
        lable = torch.cat(
            [
                torch.tensor(dec_input_tokens ,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype=torch.int64)
             
            ]
        )  

        assert encoder_input.size(0) == self.seq_Len
        assert decoder_input.size(0) == self.seq_Len
        assert lable.size(0) == self.seq_Len