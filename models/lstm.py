import time 
import random
import torch
import torch.nn as nn

from torch import optim
from typing import List

from .lstm_utils import prepareData, tensorsFromPair, timeSince, tensorFromSentence, normalizeString, showPlot
from .lstm_models import EncoderRNN, AttnDecoderRNN

class LSTMMT:

    def __init__(self, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
            hidden_size:int=256, max_length:int=500) -> None:
        self.SOS_token = 0
        self.EOS_token = 1 
        self.device = device
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.input_lang = None 
        self.output_lang = None
        self.encoder = None
        self.decoder = None

    def train(self, dataset, n_iters:int=5000) -> None: 
        self.input_lang, self.output_lang, pairs = prepareData("en", "fr", dataset)
        self.encoder = EncoderRNN(self.input_lang.n_words, self.hidden_size, self.device).to(self.device)
        self.decoder = AttnDecoderRNN(self.hidden_size, self.output_lang.n_words, self.device, dropout_p=0.1).to(self.device)

        print_every = n_iters//10
        plot_every = n_iters//20
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adagrad(self.encoder.parameters())
        decoder_optimizer = optim.Adagrad(self.decoder.parameters())
        training_pairs = [tensorsFromPair(random.choice(pairs), self.input_lang, self.output_lang)
                        for i in range(n_iters)]
        criterion = nn.CrossEntropyLoss()

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self.train_step(input_tensor, target_tensor, 
                    encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)
    
    def train_step(self, input_tensor, target_tensor, encoder_optimizer, decoder_optimizer,
        criterion, teacher_forcing_ratio:float=0.5):

        encoder_hidden = self.encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length

    def predict(self, sentence_list:List[str]) -> List[str]:
        n_sentences = len(sentence_list)
        predictions = [""]*n_sentences
        for i in range(n_sentences):
            sent_to_trad = normalizeString(sentence_list[i])
            predictions[i] = self.translate_sentence(sent_to_trad)
        return predictions

    def translate_sentence(self, sentence:str) -> str:
        with torch.no_grad():
            input_tensor = tensorFromSentence(self.input_lang, sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                            encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([[self.SOS_token]], device=self.device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()
            decoded_sentence = ' '.join(decoded_words[:len(decoded_words)-1])
            return decoded_sentence

    def load_model(self) -> None:
        pass 