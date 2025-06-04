import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def bar_plot(data, metrics):

    x = np.arange(len(methods)) 
    width = 0.2 
    num_metrics = len(metrics)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_ylim(0.6,1.0)
    for i, metric_data in enumerate(data):
        ax.bar(x + i * width, metric_data, width, label=metrics[i])

    ax.set_xlabel('Machine Learning Methods')
    ax.set_ylabel('Scores')
    ax.set_title('Performance Metrics for ML Methods')
    ax.set_xticks(x + width * (num_metrics - 1) / 2)
    ax.set_xticklabels(methods)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return

def plot_two_condition_violin(condition1, condition2, condition1_label='FIT-WM-MS dataset', condition2_label='PSI-MS dataset', title='accuracy with feature pre-selection'):

    data = pd.DataFrame({
        'accuracy': condition1 + condition2,
        'dataset': [condition1_label] * len(condition1) + [condition2_label] * len(condition2)
    })

    plt.figure(figsize=(8, 6))
    sns.violinplot(x='dataset', y='accuracy', data=data, inner='box', palette='Set2')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return

    
    
