import matplotlib.pyplot as plt
import numpy as np


def show_rag_evaluation_metrics_by_context_type():
    # Data from the table
    contexts = ['TOP_5', 'TOP_5_RERANKING', 'PERFECT']
    faithfulness_means = [0.69, 0.78, 0.84]
    faithfulness_std = [0.33, 0.32, 0.25]
    correctness_means = [0.70, 0.71, 0.77]
    correctness_std = [0.20, 0.20, 0.17]

    # Bar width and position setup
    x = np.arange(len(contexts))
    width = 0.35  # width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, faithfulness_means, width, yerr=faithfulness_std,
                label='Faithfulness', capsize=5, color='skyblue')
    bars2 = ax.bar(x + width/2, correctness_means, width, yerr=correctness_std,
                label='Answer Correctness', capsize=5, color='lightgreen')

    # Labels and title
    ax.set_ylabel('Scores')
    ax.set_title('RAG Evaluation Metrics by Context Type (LLaMA3)')
    ax.set_xticks(x)
    ax.set_xticklabels(contexts)
    ax.legend()

    # Grid and layout
    ax.yaxis.grid(True)
    fig.tight_layout()

    # Show the plot
    plt.show()
    plt.savefig('rag_evaluation_metrics_by_context_type.png')


if __name__ == "__main__":
    show_rag_evaluation_metrics_by_context_type()