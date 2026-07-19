import torch
import torch.nn as nn
from torch.utils.data import Dataset ,DataLoader , random_split

from dataset import BilingualDataset ,casual_mask
from model import Build_Transformer

from config import get_weight_file_path

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from tqdm import tqdm

import warnings
from config import get_config


def get_all_sentences(ds ,lang):
    for item in ds:
        yield item[lang]

def get_or_build_tokenizer(config , ds ,lang):
    # this lone will create somthing like config['tokenizer_file'] = '../tokenizers/tokenizer_{0}.json'
    tokenier_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenier_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer =Whitespace()
        trainer = WordLevelTrainer(special_tokens = ["[UNK]","[PAD]","[SOS]","[EOS]"],min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(ds , lang) , trainer=trainer)
        tokenizer.save(str(tokenier_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenier_path))
    return tokenizer

def get_ds(config):
    ds_raw = load_dataset('gopi30/english-tamil', split='train')

    #build tokenizer
    tokenizer_src = get_or_build_tokenizer(config , ds_raw , config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config , ds_raw , config['lang_tgt'])

    #keep 90% for tranning and 10% for the validation 
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_size ,val_ds_size = random_split(ds_raw,[train_ds_size ,val_ds_size])

    train_ds = BilingualDataset(train_ds_size , tokenizer_src , tokenizer_tgt ,config['lang_src'] , config['lang_tgt'] , config['seq_len'])
    val_ds = BilingualDataset(val_ds_size , tokenizer_src , tokenizer_tgt ,config['lang_src'] , config['lang_tgt'] , config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0 

    for item in ds_raw:
       
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
        max_len_src =max(max_len_src , len(src_ids))
        max_len_tgt = max(max_len_tgt , len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max lenght of target sentence: {max_len_tgt}')

    train_dataloader  = DataLoader(train_ds , batch_size = config['batch_size'] , shuffle = True)   
    val_dataloader = DataLoader(val_ds , batch_size=1 , shuffle=True)

    return train_dataloader ,val_dataloader ,tokenizer_src , tokenizer_tgt 


def get_model(config , vocab_src_len , vocab_tgt_len):
    model = Build_Transformer(vocab_src_len , vocab_tgt_len , config['seq_len'] , config['seq_len'], config['d_model'])
    return model

def train_model(config):
    #define device
    device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    Path(config['model_folder']).mkdir(parents=True , exist_ok=True)

    train_dataloader ,val_dataloader ,tokenizer_src , tokenizer_tgt  = get_ds(config)
    model = get_model(config , tokenizer_src.get_vocab_size() , tokenizer_tgt.get_vocab_size()).to(device)
    #tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer  = torch.optim.Adam(model.parameters() , lr = config['lr'] , eps = 1e-9)

    # used to resume the model if creashed while on tranning
    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weight_file_path(config , config['preload'])
        print(f'preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state ['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1 ).to(device)  

    for epoch in range(initial_epoch , config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader , desc =f'processing epoch{epoch:02d}')
        for batch in batch_iterator :
            encoder_input = batch['encoder_input'].to(device) #(B ,seq_len)
            decoder_input = batch['decoder_input'].to(device) #(b ,seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B ,1, 1 ,seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (b ,1 , 1 ,seq_len ,seq_len)

            #run  the tensor through the transfrmer 
            encoder_output = model.encoder(encoder_input ,encoder_mask) # (B , seq_len , d_model)
            decoder_output = model.decoder(encoder_output ,encoder_mask ,decoder_input , decoder_mask) #(B , seq_len ,d_model)
            project_output = model.project(decoder_output ) #(B ,seq_len ,tgt_vocab_size)

            lable = batch['lable'].to(device) # (B , seq_len) 

            #(B , seq_len , tgt_vocab_zise) --> (B  * seq_len , tgt_vocab_size)
            loss = loss_fn(project_output.view(-1 , tokenizer_tgt.get_vocab_size( )) , lable.view(-1) ) 
            batch_iterator.set_postfix({f'loss' : f'{loss.item():6.3f}'})

            #log the loss
            writer.add_scalar('train loss' , loss.item() ,global_step)
            writer.flush()

            # backpropagation 
            loss.backward()

            # update the  weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # save model
        model_filename = get_weight_file_path(config , f'{epoch:02d}')
        torch.save({
            'epoch' : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)

if __name__ =="__main__" :
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)

