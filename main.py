from datasets import load_dataset, ReadInstruction

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq

import evaluate
import numpy as np

from models.transformer import TransformerMT


if __name__ == "__main__":
    #dataset = load_dataset("wmt14", "fr-en")
    #remove everything after en-fr when using it for real
    dataset = load_dataset("opus_books", "en-fr", split=ReadInstruction("train",from_=11, to=12, unit="%", rounding="pct1_dropremainder"))
    dataset = dataset.train_test_split(test_size=0.2)

    model = TransformerMT()
    model.train(dataset)

    test_dataset = ["Legumes share resources with nitrogen-fixing bacteria."]
    print(model.predict(test_dataset))