import evaluate 
import numpy as np

from typing import List

def sacrebleu_metric(sentences:List[str], translated_sentences:List[str]):
    metric = evaluate.load("sacrebleu")
    references = []
    hypothesis = []
    assert len(sentences) == len(translated_sentences)
    n = len(sentences)
    for i in range(n):
        references.append([sentences[i]])
        hypothesis.append(translated_sentences[i])
    result = metric.compute(predictions=translated_sentences, references=references)
    result = {"bleu": result["score"]}
    result = {k: round(v, 4) for k, v in result.items()}
    return result