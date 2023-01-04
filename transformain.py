import os
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, ReadInstruction
from models.transformer import TransformerMT


if __name__ == "__main__":
    dataset = load_dataset("opus_books", "en-fr", split=ReadInstruction("train",from_=0, to=5, unit="%", rounding="pct1_dropremainder"))
    #dataset = dataset["train"].train_test_split(test_size=0.2)
    dataset = dataset.train_test_split(test_size=0.2)
    preloaded_dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = preloaded_dataset["train"]
    eval_dataset = preloaded_dataset["test"]
    test_dataset = dataset["test"]
    #remove everything after en-fr when using it for real
    #dataset = load_dataset("opus_books", "en-fr", split=ReadInstruction("train",from_=11, to=12, unit="%", rounding="pct1_dropremainder"))
    #dataset = dataset.train_test_split(test_size=0.2)
    print(list(train_dataset[0]["translation"].values()))
    

    model = TransformerMT()
    model.train(train_dataset, eval_dataset)
    
    #test_dataset = ["Legumes share resources with nitrogen-fixing bacteria."]
    test = []
    for i in range(len(test_dataset)):
        test.append(test_dataset[i]["translation"]["en"])
    print(model.predict(test))