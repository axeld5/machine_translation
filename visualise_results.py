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

visualize({"LSTM": 0.9, "ConvS2S": 0.8})