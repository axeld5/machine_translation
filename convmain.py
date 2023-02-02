from datasets import load_dataset, ReadInstruction
from models.cnn import ConvS2SMT
from bleu_metric import sacrebleu_metric

if __name__ == "__main__":
    dataset = load_dataset("tatoeba", lang1 = "en", lang2 = "fr", split=ReadInstruction("train",from_=0, to=100, unit="%", rounding="pct1_dropremainder"))
    cnn_model = ConvS2SMT(dataset)
    dataset = dataset.train_test_split(test_size=0.01)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    cnn_model.train(train_dataset, 10, 8)

    sentence = test_dataset[0]["translation"]["en"]
    conv_predictions = cnn_model.predict([sentence])
    print(conv_predictions)
    test = []
    for i in range(len(test_dataset)):
        test.append(test_dataset[i]["translation"]["en"])
    conv_predictions = cnn_model.predict(test)
    
    conv_score = sacrebleu_metric(test, conv_predictions)
    print(conv_score)