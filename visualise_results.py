import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

from typing import Dict 

def visualize(model_perf:Dict[str, float]) -> None:
    perf_dict = {"scores": [], "model_names": []}
    for model_name, score in model_perf.items():
        perf_dict["model_names"].append(model_name)
        perf_dict["scores"].append(score)
    perf_df = pd.DataFrame(perf_dict)
    ax = sns.barplot(data=perf_df, x="model_names", y="scores")
    ax.set_title("Bleu score for each model")
    plt.show() 

def show_boxplot(model_dict:Dict[str, float], x:str, y:str) -> None:
    df = pd.DataFrame.from_dict(model_dict)
    sns.boxplot(x=x, y=y, data=df)
    plt.show()


if __name__ == "__main__":
    model_dict = {'model_name': ['s2s_lstm', 'hugface_transformer', 's2s_lstm', 'hugface_transformer', 's2s_lstm', 'hugface_transformer', 's2s_lstm', 'hugface_transformer', 's2s_lstm', 'hugface_transformer'], 'sacrebleu': [26.3549, 41.9926, 26.0677, 42.6792, 26.7663, 43.8609, 25.2895, 43.3692, 25.9, 42.1943]}
    show_boxplot(model_dict, x="model_name", y="sacrebleu")