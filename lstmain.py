from datasets import load_dataset, ReadInstruction
from models.lstm import LSTMMT
from bleu_metric import sacrebleu_metric
from models.lstm_utils import normalizeString

if __name__ == "__main__":
    dataset = load_dataset("tatoeba", lang1 = "en", lang2 = "fr", split=ReadInstruction("train",from_=0, to=100, unit="%", rounding="pct1_dropremainder"))
    lstm_model = LSTMMT(hidden_size=1024, max_length=500)
    lstm_model.build_language(dataset)
    lstm_model.init_models()
    dataset = dataset.train_test_split(test_size=0.01)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    lstm_model.load_model()
    #lstm_model.train(train_dataset, 100000)

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