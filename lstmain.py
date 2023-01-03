from datasets import load_dataset, ReadInstruction
from models.lstm import LSTMMT

if __name__ == "__main__":
    #dataset = load_dataset("wmt14", "fr-en")
    #remove everything after en-fr when using it for real
    dataset = load_dataset("opus_books", "en-fr", split=ReadInstruction("train",from_=11, to=12, unit="%", rounding="pct1_dropremainder"))
    dataset = dataset.train_test_split(test_size=0.2)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print(list(dataset["train"][0]["translation"].values()))
    
    model = LSTMMT()
    model.train(train_dataset, 5000)
    test_dataset = [train_dataset[0]["translation"]["en"]]
    print(test_dataset)
    print(model.predict(test_dataset))