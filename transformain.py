import os
import torch
print(torch.cuda.is_available())
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, ReadInstruction
from models.transformer import TransformerMT
from bleu_metric import sacrebleu_metric


if __name__ == "__main__":
    dataset = load_dataset("tatoeba", lang1 = "en", lang2 = "fr", split=ReadInstruction("train",from_=0, to=100, unit="%", rounding="pct1_dropremainder"))
    dataset = dataset.train_test_split(test_size=0.2)
    preloaded_dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = preloaded_dataset["train"]
    eval_dataset = preloaded_dataset["test"]
    train_dataset = dataset["test"]
    print(list(train_dataset[0]["translation"].values()))
    

    model = TransformerMT()
    model.train(train_dataset, eval_dataset)
    
    test = []
    for i in range(len(eval_dataset)):
        test.append(eval_dataset[i]["translation"]["en"])
    predictions = model.predict(test)
    print(sacrebleu_metric(test, predictions))