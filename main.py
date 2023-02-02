import os
import torch
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, ReadInstruction
from models.transformer import TransformerMT
from models.lstm import LSTMMT
from bleu_metric import sacrebleu_metric
from visualise_results import visualize

print(torch.device.is_available())
if __name__ == "__main__":
    dataset = load_dataset("tatoeba", lang1 = "en", lang2 = "fr", split=ReadInstruction("train",from_=0, to=100, unit="%", rounding="pct1_dropremainder"))
    #dataset = dataset["train"].train_test_split(test_size=0.2)
    lstm_model = LSTMMT()
    lstm_model.build_language(dataset)
    dataset = dataset.train_test_split(test_size=0.2)
    preloaded_dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = preloaded_dataset["train"]
    eval_dataset = preloaded_dataset["test"]
    test_dataset = dataset["test"]
    lstm_model.load_model()

    
    test = []
    for i in range(len(test_dataset)):
        test.append(test_dataset[i]["translation"]["en"])
    lstm_predictions = lstm_model.predict(test)
    
    lstm_score = sacrebleu_metric(test, lstm_predictions)

    transformer_model = TransformerMT()
    transformer_model.load_model()

    transformer_predictions = transformer_model.predict(test)
    transformer_score = sacrebleu_metric(test, transformer_predictions)
    perf_dict = {"lstm": lstm_score, "transformer":transformer_score}
    visualize(perf_dict)
    