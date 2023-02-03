import evaluate
import numpy as np

from joblib import dump, load
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, pipeline
from typing import List


from .transformer_utils import postprocess_text

class TransformerMT:

    def __init__(self, source_lang:str="en", target_lang:str="fr", 
            prefix:str="translate English to French: ") -> None:   
        self.tokenizer = AutoTokenizer.from_pretrained("t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")  
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.prefix = prefix

    def preprocess_function(self, examples):
        inputs = [self.prefix + example[self.source_lang] for example in examples["translation"]]
        targets = [example[self.target_lang] for example in examples["translation"]]
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    def train(self, train_dataset, eval_dataset, n_iters:int=2, suffix="") -> None:
        train_tokenized = train_dataset.map(self.preprocess_function, batched=True)
        eval_tokenized = eval_dataset.map(self.preprocess_function, batched=True)
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        training_args = Seq2SeqTrainingArguments(
            output_dir="models/saved_models/transf",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=n_iters,
            predict_with_generate=True,
            #fp16=True,
        )
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        dump(self.model, "models/saved_models/transformer"+suffix+".joblib")

    def predict(self, test_dataset:List[str]) -> None:
        translator = pipeline("translation_en_to_fr", model=self.model.to("cpu"), tokenizer=self.tokenizer)
        predictions = [0]*len(test_dataset)
        print(len(test_dataset))
        for i in range(len(test_dataset)):
            predictions[i] = translator(self.prefix + test_dataset[i])[0]["translation_text"]
            if i%1000 == 0:
                print(i)
        return predictions

    def load_model(self, suffix=""):
        self.model = load("models/saved_models/transformer"+suffix+".joblib")

    def compute_metrics(self, eval_preds):
        metric = evaluate.load("sacrebleu")
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result