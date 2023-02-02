from torchtext.data.utils import get_tokenizer
from typing import List
from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab

from models.scratch_transformer_models import Seq2SeqTransformer

class ScratchTransfoNMT: 

    def __init__(self, dataset) -> None:
        self.SRC_LANGUAGE = 'en'
        self.TGT_LANGUAGE = 'fr'        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.token_transform = {}
        self.vocab_transform = {}
        self.build_vocab(dataset)
        self.get_text_transform()
        self.SRC_VOCAB_SIZE = len(self.vocab_transform[self.SRC_LANGUAGE])
        self.TGT_VOCAB_SIZE = len(self.vocab_transform[self.TGT_LANGUAGE])
        EMB_SIZE = 512
        NHEAD = 8
        FFN_HID_DIM = 512
        NUM_ENCODER_LAYERS = 3
        NUM_DECODER_LAYERS = 3
        self.model = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, self.SRC_VOCAB_SIZE, self.TGT_VOCAB_SIZE, FFN_HID_DIM).to(self.device)
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def train(self, train_dataset, num_epochs=10, batch_size=8):
        optimizer = torch.optim.AdamW(self.model.parameters())
        train_iter = DataLoader(train_dataset, batch_size=batch_size, collate_fn=self.collate_fn)

        for epoch in range(1, num_epochs+1):
            start_time = timer()
            train_loss = self.train_epoch(train_iter, optimizer)
            end_time = timer()
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        
    def save_model(self) -> None:
        torch.save(self.model.state_dict(), 'models/saved_models/scratch_transfo_weights.pth')  
    
    def load_model(self) -> None:
        self.model.load_state_dict(torch.load('models/saved_models/scratch_transfo_weights.pth'))

    def train_epoch(self, train_iter, optimizer):
        self.model.train()
        losses = 0
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        for src, tgt in train_iter:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)

            logits = self.model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

        return losses / len(train_iter)
    
    def evaluate(self, val_iter):
        self.model.eval()
        losses = 0
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        for src, tgt in val_iter:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)

            logits = self.model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(val_iter)
    
    # function to generate output sequence using greedy algorithm
    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            memory = memory.to(self.device)
            tgt_mask = (self.generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(self.device)
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.EOS_IDX:
                break
        return ys


    # actual function to translate input sentence into target language
    def translate(self, src_sentence: str):
        self.model.eval()
        src = self.text_transform[self.SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 5, start_symbol=self.BOS_IDX).flatten()
        return " ".join(self.vocab_transform[self.TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
    
    def predict(self, text_list):
        translated_list = []
        for text in text_list:
            translated_text = self.translate(text)
            translated_list.append(translated_text)
        return translated_list

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == self.PAD_IDX).transpose(0, 1)
        tgt_padding_mask = (tgt == self.PAD_IDX).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
    
    # helper function to club together sequential operations
    def sequential_transforms(self, *transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(self, token_ids: List[int]):
        return torch.cat((torch.tensor([self.BOS_IDX]),
                        torch.tensor(token_ids),
                        torch.tensor([self.EOS_IDX])))
    
    def collate_fn(self, batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[self.TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=self.PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.PAD_IDX)
        return src_batch, tgt_batch
    
    def build_vocab(self, dataset): 
        self.token_transform[self.SRC_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')
        self.token_transform[self.TGT_LANGUAGE] = get_tokenizer('spacy', language='fr_core_news_sm')

        # Define special symbols and indices
        self.UNK_IDX, self.PAD_IDX, self.BOS_IDX, self.EOS_IDX = 0, 1, 2, 3
        # Make sure the tokens are in order of their indices to properly insert them in vocab
        self.special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.en_tokenizer = get_tokenizer('basic_english')
        en_counter = Counter()
        self.fr_tokenizer = get_tokenizer('spacy', 'fr_core_news_sm')
        fr_counter = Counter()
        for i in range(len(dataset)):
            en_text, fr_text = list(dataset[i]['translation'].values())
            en_counter.update(self.en_tokenizer(en_text))
            fr_counter.update(self.fr_tokenizer(fr_text))
        en_vocab = vocab(en_counter, specials=('<unk>', '<pad>', '<bos>', '<eos>'))
        fr_vocab = vocab(fr_counter, specials=('<unk>', '<pad>', '<bos>', '<eos>'))
        self.vocab_transform[self.SRC_LANGUAGE] = en_vocab
        self.vocab_transform[self.TGT_LANGUAGE] = fr_vocab
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            self.vocab_transform[ln].set_default_index(self.UNK_IDX)
    
    def get_text_transform(self):
        self.text_transform = {}
        for ln in [self.SRC_LANGUAGE, self.TGT_LANGUAGE]:
            self.text_transform[ln] = self.sequential_transforms(self.token_transform[ln], #Tokenization
                                                self.vocab_transform[ln], #Numericalization
                                                self.tensor_transform) # Add BOS/EOS and create tensor