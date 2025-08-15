from pathlib import Path
from typing import Callable
from tokenizers import Tokenizer
from tokenizers.models import BPE,WordPiece
from tokenizers.trainers import BpeTrainer,WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset,DataLoader,random_split
from dataset_config import BiLingualDataset,hidding_future
from config import Config, get_config
from datasets import load_from_disk
from model import Transformer, build_transformer

def get_all_sentences(ds, lang):
    for train_test_split in ds.keys():
        for item in ds[train_test_split]:
            yield item[lang]

def get_or_build_tokenizer(config:Config, ds:Dataset , lang:str):
    tokenizer_path = Path(config['tokenizer_file'].format(lang=lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(model = WordPiece(unk_token="[UNK]")) #type:ignore
        tokenizer.pre_tokenizer = Whitespace() #type:ignore 
        
        trainer = WordPieceTrainer(special_tokens =["[UNK]", "[PAD]", "[SOS]", "[EOS]"],#type:ignore 
                            min_frequency=2,  # type: ignore
                            # max_token_length=config['max_token_length'] # type: ignore
                            )  
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config:Config):
    dataset_path = Path(config['dataset_folder_name'])
    # if not dataset_path.exists():
    #     dataset_ = load_dataset(config['huggingface_dataset_download_input'])
    #     dataset_.save_to_disk(dataset_path) # type: ignore
    
    dataset_ = load_from_disk(dataset_path)
    source_tokenizer = get_or_build_tokenizer(config=config,ds=dataset_, # type: ignore
                                              lang=config['source_language'])
    target_tokenizer = get_or_build_tokenizer(config=config,ds=dataset_, # type: ignore
                                              lang=config['target_language'])
    
    train_dataset_size = int(config['train_validate']*len(dataset_['train']))
    validation_dataset_size = len(dataset_['train']) - train_dataset_size
    train_dataset,val_dataset = random_split(dataset=dataset_['train'], # type: ignore
                                   lengths=[train_dataset_size,validation_dataset_size])
    
    source_max_len =0
    target_max_len =0
    for data in train_dataset:
        source_ids = source_tokenizer.encode(data[config['source_language']]).ids
        target_ids = target_tokenizer.encode(data[config['target_language']]).ids
        source_max_len = max(source_max_len,len(source_ids))
        target_max_len = max(target_max_len,len(target_ids))
    for data in train_dataset:
        source_ids = source_tokenizer.encode(data[config['source_language']]).ids
        target_ids = target_tokenizer.encode(data[config['target_language']]).ids
        source_max_len = max(source_max_len,len(source_ids))
        target_max_len = max(target_max_len,len(target_ids))
    print(f"{source_max_len=}")
    print(f"{target_max_len=}")
    max_possible_length = max(source_max_len,target_max_len)
        
    train_dataset = BiLingualDataset(dataset_=train_dataset,
                                     source_tokenizer=source_tokenizer,
                                     target_tokenizer=target_tokenizer,
                                     max_seq_length=max_possible_length,
                                     source_language=config['source_language'],
                                     target_language=config['target_language']
                                    )
    val_dataset = BiLingualDataset(dataset_=train_dataset,
                                   source_tokenizer=source_tokenizer,
                                   target_tokenizer=target_tokenizer,
                                   max_seq_length=max_possible_length,
                                   source_language=config['source_language'],
                                   target_language=config['target_language']
                                  ) 
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True)  
    val_dataloader = DataLoader(dataset=val_dataset,
                                  batch_size=config['batch_size'],
                                  shuffle=True)  
    
    return train_dataloader,val_dataloader

def get_transformer(config : Config)->Transformer:
    d_model = config['d_model']
    dropout = config['dropout']
    num_attention_heads = config['num_atttention_heads']
    num_stacks = config['num_stacks']
    source_tokenizer_path = config['tokenizer_file'].format(lang = config['source_language'])
    source_tokenizer = Tokenizer(model = WordPiece(unk_token="[UNK]")) # type: ignore
    source_tokenizer = source_tokenizer.from_file(source_tokenizer_path)
    target_tokenizer_path = config['tokenizer_file'].format(lang = config['target_language'])
    target_tokenizer = Tokenizer(model = WordPiece(unk_token="[UNK]")) # type: ignore
    target_tokenizer = target_tokenizer.from_file(target_tokenizer_path) 
    input_vocab_size = source_tokenizer.get_vocab_size()   
    target_vocab_size = source_tokenizer.get_vocab_size() 
    feed_forward_num_out = config['feed_forward_num_out']  
    transformer = build_transformer(
                                        input_vocab_size = input_vocab_size,
                                        target_vocab_size = target_vocab_size,
                                        max_input_seq_lenght = 112,
                                        max_target_seq_lenght = 41,
                                        d_model = d_model,
                                        dropout = dropout,
                                        num_stacks = num_stacks,
                                        num_attention_heads = num_attention_heads,
                                        feed_forward_num_out = feed_forward_num_out,
                                   )
    return transformer




if __name__=="__main__":
    # get_dataset(config=get_config())
    pass