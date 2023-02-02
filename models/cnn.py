import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import vocab
import time

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models.cnn_models import CNNDecoder, CNNEncoder, ConvS2S

class ConvS2SMT:

    def __init__(self, dataset) -> None:
        self.get_vocab(dataset)
        self.INPUT_DIM = len(self.en_vocab)
        self.OUTPUT_DIM = len(self.fr_vocab)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

        ENC_EMB_DIM = 256
        DEC_EMB_DIM = 256
        ENC_HID_DIM = 512
        KERNEL_SIZE = 3
        DEC_HID_DIM = 512
        ATTN_DIM = 64
        ENC_DROPOUT = 0.1
        DEC_DROPOUT = 0.1
        N_LAYERS = 3
        self.PAD_IDX = self.fr_vocab.get_stoi()['<pad>']

        self.encoder = CNNEncoder(input_dim=self.INPUT_DIM, emb_dim=ENC_EMB_DIM, hid_dim=ENC_HID_DIM, n_layers=N_LAYERS,
                    kernel_size=KERNEL_SIZE, dropout=ENC_DROPOUT, device=self.device)
        self.decoder = CNNDecoder(output_dim=self.OUTPUT_DIM, emb_dim=DEC_EMB_DIM, hid_dim=DEC_HID_DIM, n_layers=N_LAYERS, 
                    kernel_size=KERNEL_SIZE, dropout=DEC_DROPOUT, trg_pad_idx=self.PAD_IDX, device=self.device)
        self.model = ConvS2S(self.encoder, self.decoder).to(self.device)
        self.model.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)

    def training_step(self, train_iter, clip:float=1):   
        optimizer = optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        self.model.train()

        epoch_loss = 0
        for _, (src, trg) in enumerate(train_iter):
            src, trg = src.to(self.device), trg.to(self.device)
            trg = trg.permute(1, 0)
            optimizer.zero_grad()

            output, _ = self.model(src, trg)
            output = output[:,1:].contiguous().view(-1, output.shape[-1])
            trg = trg[:,1:].contiguous().view(-1)     

            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_iter)
    
    def evaluate(self, iterator: torch.utils.data.DataLoader):
        criterion = nn.CrossEntropyLoss(ignore_index=self.PAD_IDX)
        self.model.eval()

        epoch_loss = 0

        with torch.no_grad():

            for _, (src, trg) in enumerate(iterator):
                src, trg = src.to(self.device), trg.to(self.device)

                output, _ = self.model(src, trg, 0) #turn off teacher forcing

                output = output[:,1:].contiguous().view(-1, output.shape[-1])
                trg = trg[:,1:].contiguous().view(-1)    

                loss = criterion(output, trg)

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)
    
    def epoch_time(self, start_time: int, end_time: int):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
    
    def train(self, train_dataset, n_epochs:int=10, batch_size:int=8):            
        data = self.preprocess_dataset(train_dataset)
        train_iter = DataLoader(data, batch_size=batch_size,
                            shuffle=True, collate_fn=self.generate_batch) 
        for epoch in range(n_epochs):
            start_time = time.time()

            train_loss = self.training_step(train_iter)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if epoch%1 == 0 or epoch == n_epochs - 1:
                print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
                print(f'\tTrain Loss: {train_loss:.3f}')          
        self.save_model()   

    def save_model(self) -> None:
        torch.save(self.encoder.state_dict(), 'models/saved_models/cnn_encoder_weights.pth')
        torch.save(self.decoder.state_dict(), 'models/saved_models/cnn_decoder_weights.pth')    
    
    def load_model(self) -> None:
        self.encoder.load_state_dict(torch.load('models/saved_models/cnn_encoder_weights.pth'))
        self.decoder.load_state_dict(torch.load('models/saved_models/cnn_decoder_weights.pth'))
        self.model = ConvS2S(self.encoder, self.decoder).to(self.device)

    def greedy_decode(self, src, max_len, start_symbol):
        src = src.to(self.device)
        conved, combined = self.model.encoder(src)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len-1):
            conved = conved.to(self.device)
            combined = combined.to(self.device)
            out, _ = self.model.decoder(ys, conved, combined)
            out = out.transpose(0, 1)
            prob = out[:, -1]
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.EOS_IDX:
                break
        return ys


    def translate(self, src_sentence: str):
        self.model.eval()
        src = self.en_tokenizer(src_sentence)
        token_src = torch.tensor([self.en_vocab.get_stoi()[s] for s in src]).view(-1, 1)
        num_tokens = token_src.shape[0]
        tgt_tokens = self.greedy_decode(token_src, max_len=num_tokens + 5, start_symbol=self.BOS_IDX).flatten()
        return " ".join(self.fr_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

    def predict(self, text_list):
        translated_list = []
        for text in text_list:
            translated_text = self.translate(text)
            translated_list.append(translated_text)
        return translated_list
    
    def get_vocab(self, dataset):
        self.en_tokenizer = get_tokenizer('basic_english')
        en_counter = Counter()
        self.fr_tokenizer = get_tokenizer('spacy', 'fr_core_news_sm')
        fr_counter = Counter()
        for i in range(len(dataset)):
            en_text, fr_text = list(dataset[i]['translation'].values())
            en_counter.update(self.en_tokenizer(en_text))
            fr_counter.update(self.fr_tokenizer(fr_text))
        self.en_vocab = vocab(en_counter, specials=('<unk>', '<pad>', '<bos>', '<eos>'))
        self.fr_vocab = vocab(fr_counter, specials=('<unk>', '<pad>', '<bos>', '<eos>'))
            
        self.PAD_IDX = self.fr_vocab['<pad>']
        self.BOS_IDX = self.fr_vocab['<bos>']
        self.EOS_IDX = self.fr_vocab['<eos>']
    
    def preprocess_dataset(self, dataset):
        data = []
        for i in range(len(dataset)):
            raw_en = dataset[i]["translation"]["en"]
            raw_fr = dataset[i]["translation"]["fr"]
            en_tensor_ = torch.tensor([self.en_vocab[token] for token in self.en_tokenizer(raw_en)],
                                            dtype=torch.long)
            fr_tensor_ = torch.tensor([self.fr_vocab[token] for token in self.fr_tokenizer(raw_fr)],
                                            dtype=torch.long)
            data.append((en_tensor_, fr_tensor_))
        return data

    def generate_batch(self, data_batch):
        en_batch, fr_batch = [], []
        for (en_item, fr_item) in data_batch:
            en_batch.append(torch.cat([torch.tensor([self.BOS_IDX]), en_item, torch.tensor([self.EOS_IDX])], dim=0))
            fr_batch.append(torch.cat([torch.tensor([self.BOS_IDX]), fr_item, torch.tensor([self.EOS_IDX])], dim=0))
        en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX)
        fr_batch = pad_sequence(fr_batch, padding_value=self.PAD_IDX)
        return en_batch, fr_batch