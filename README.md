# machine_translation
Repository for the Deep Learning Project that implements Neural Machine Translation methods and evaluates their sacrebleu scores on the Tatoeba and Opus Books Datasets.

The models that were planned to be used at first were a Transformer, a ConvS2S, and a Seq2Seq Encoder-Decoder model using LSTMs, all implemented from scratch in Pytorch. These models were meant to be compared to a huggingface transformer, serving as a baseline.

However, due to the low performances of the convS2S and transformer, results for them cannot be plotted. This study will therefore focus on comparing a Seq2Seq LSTM Model with a huggingface transformer.

# training 

To train the models, use "opus_train.py" and "tatoeba_train.py".
To evaluate the models' performance, use "opus_main.py" and "tatoeba_main.py".

In order to train the models that were not included in the results part of the report, one can still use the "failed_models_train.py" module.
