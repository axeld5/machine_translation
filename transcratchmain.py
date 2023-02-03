from datasets import load_dataset, ReadInstruction
from models.scratch_transformer import ScratchTransfoNMT
from bleu_metric import sacrebleu_metric

if __name__ == "__main__":
    dataset = load_dataset("tatoeba", lang1 = "en", lang2 = "fr", split=ReadInstruction("train",from_=0, to=10, unit="%", rounding="pct1_dropremainder"))
    scratch_transfo_model = ScratchTransfoNMT(dataset)
    dataset = dataset.train_test_split(test_size=0.01)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    scratch_transfo_model.train(train_dataset, 10, 8)

    sentence = test_dataset[0]["translation"]["en"]
    scratch_transfo_predictions = scratch_transfo_model.predict([sentence])
    print(scratch_transfo_predictions)
    test = []
    version = []
    for i in range(len(test_dataset)):
        test.append(test_dataset[i]["translation"]["en"])
        version.append(test_dataset[i]["translation"]["fr"])
    scratch_transfo_predictions = scratch_transfo_model.predict(test)
    
    scratch_transfo_score = sacrebleu_metric(version, scratch_transfo_predictions)
    print(scratch_transfo_score)