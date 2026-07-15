import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path


def get_all_sentences(ds ,lang):
    for item in ds:
        yield item['transulation'][lang]

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

