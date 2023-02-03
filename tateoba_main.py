import os
import torch
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, ReadInstruction
from models.transformer import TransformerMT
from models.lstm import LSTMMT
from models.lstm_utils import normalizeString
from bleu_metric import sacrebleu_metric
from visualise_results import visualize, show_boxplot

print(torch.cuda.is_available())
if __name__ == "__main__":
    dataset = load_dataset("tatoeba", lang1 = "en", lang2 = "fr", split=ReadInstruction("train",from_=0, to=100, unit="%", rounding="pct1_dropremainder"))
    perf_dict = {"model_name":[], "sacrebleu":[]}
    for _ in range(5):
        lstm_model = LSTMMT(hidden_size=1024, max_length=500)
        lstm_model.build_language(dataset)
        lstm_model.init_models()
        split_dataset = dataset.train_test_split(test_size=0.01)
        test_dataset = split_dataset["test"]
        lstm_model.load_model()
        
        test = []   
        normalized_test = []    
        fr_test = []
        fr_normalized_test = []
        for i in range(len(test_dataset)):
            elem = test_dataset[i] 
            test.append(elem["translation"]["en"])        
            normalized_test.append(normalizeString(elem["translation"]["en"]))
            fr_test.append(elem["translation"]["fr"])
            fr_normalized_test.append(normalizeString(elem["translation"]["fr"]))
        lstm_predictions = lstm_model.predict(normalized_test)
        
        lstm_score = sacrebleu_metric(fr_normalized_test, lstm_predictions)
        perf_dict["model_name"].append("s2s_lstm")
        perf_dict["sacrebleu"].append(lstm_score["bleu"])
        

        transformer_model = TransformerMT()
        transformer_model.load_model()

        transformer_predictions = transformer_model.predict(test)
        transformer_score = sacrebleu_metric(fr_test, transformer_predictions)
        perf_dict["model_name"].append("hugface_transformer")
        perf_dict["sacrebleu"].append(transformer_score["bleu"])

    print(perf_dict)
    #perf_dict = {"s2s_lstm": lstm_score['bleu'], "hugface_transformer":transformer_score['bleu']}
    #visualize(perf_dict)
    show_boxplot(perf_dict, x="model_name", y="sacrebleu")
    