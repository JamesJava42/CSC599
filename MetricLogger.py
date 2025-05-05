# MetricLogger.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def plot_metrics(metrics_file, model_name):
    df = pd.read_csv(metrics_file)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Metric", y="Score", hue="Model")
    plt.title(f'Model Performance Comparison - {model_name}')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f"{model_name}_metrics_plot.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="Model name to display")
    parser.add_argument("--metrics_file", required=True, help="CSV file with metrics")

    args = parser.parse_args()
    plot_metrics(args.metrics_file, args.model_name)
