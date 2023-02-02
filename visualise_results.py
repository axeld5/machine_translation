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

if __name__ == "__main__":
    model_perf = {"LSTM":35, "ConvS2S":37, "Transformer":40}
    visualize(model_perf)