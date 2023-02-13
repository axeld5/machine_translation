import os
os.environ["WANDB_DISABLED"] = "true"

from datasets import load_dataset, ReadInstruction

from models.lstm import LSTMMT
from models.transformer import TransformerMT
from bleu_metric import sacrebleu_metric
from models.lstm_utils import normalizeString

if __name__ == "__main__":
    dataset = load_dataset("opus_books", lang1 = "en", lang2 = "fr", split=ReadInstruction("train",from_=0, to=100, unit="%", rounding="pct1_dropremainder"))
    lstm_model = LSTMMT(hidden_size=1024, max_length=500)
    lstm_model.build_language(dataset)
    lstm_model.init_models()
    dataset = dataset.train_test_split(test_size=0.01)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    lstm_model.load_model(suffix="opus")
    lstm_model.train(train_dataset, 100000, suffix="opus")

    sentence = test_dataset[0]["translation"]["en"]
    lstm_predictions = lstm_model.predict([sentence])
    print(normalizeString(sentence))
    print(lstm_predictions)
    test = []
    version = []
    for i in range(len(test_dataset)):
        test.append(normalizeString(test_dataset[i]["translation"]["en"]))
        version.append(normalizeString(test_dataset[i]["translation"]["fr"]))
    lstm_predictions = lstm_model.predict(test)
    
    lstm_score = sacrebleu_metric(version, lstm_predictions)
    print(lstm_score)

    preloaded_dataset = dataset["train"].train_test_split(test_size=0.1)
    train_dataset = preloaded_dataset["train"]
    eval_dataset = preloaded_dataset["test"]

    model = TransformerMT()
    model.train(train_dataset, eval_dataset, suffix="opus")
    
    test = []
    version = []
    for i in range(len(test_dataset)):
        test.append(test_dataset[i]["translation"]["en"])
        version.append(test_dataset[i]["translation"]["fr"])
    predictions = model.predict(test)
    print(sacrebleu_metric(version, predictions))