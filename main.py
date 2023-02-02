import os
import torch
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, ReadInstruction
from models.transformer import TransformerMT
from models.lstm import LSTMMT
from models.lstm_utils import normalizeString
from bleu_metric import sacrebleu_metric
from visualise_results import visualize

print(torch.cuda.is_available())
if __name__ == "__main__":
    dataset = load_dataset("tatoeba", lang1 = "en", lang2 = "fr", split=ReadInstruction("train",from_=0, to=100, unit="%", rounding="pct1_dropremainder"))

    lstm_model = LSTMMT(hidden_size=1024, max_length=500)
    lstm_model.build_language(dataset)
    lstm_model.init_models()
    dataset = dataset.train_test_split(test_size=0.01)
    test_dataset = dataset["test"]
    lstm_model.load_model()
    
    test = []
    
    for i in range(len(test_dataset)):
        test.append(test_dataset[i]["translation"]["en"])
    normalized_test = []    
    for i in range(len(test_dataset)):
        normalized_test.append(normalizeString(test_dataset[i]["translation"]["en"]))
    lstm_predictions = lstm_model.predict(normalized_test)
    
    lstm_score = sacrebleu_metric(normalized_test, lstm_predictions)
    print(lstm_score)

    transformer_model = TransformerMT()
    transformer_model.load_model()

    transformer_predictions = transformer_model.predict(test)
    transformer_score = sacrebleu_metric(test, transformer_predictions)
    perf_dict = {"s2s_lstm": lstm_score['bleu'], "hugface_transformer":transformer_score['bleu']}
    visualize(perf_dict)
    