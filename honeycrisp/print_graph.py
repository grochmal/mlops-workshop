from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from sklearn.manifold import TSNE


def build_figure(Y: np.ndarray, labels_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.scatter(Y[:, 0], Y[:, 1], color="crimson", alpha=0.5)
    for cl_apple, xy in zip(labels_df["apples"], Y):
        ax.annotate(
            cl_apple, xy=xy, textcoords="offset points", xytext=(0, 7), ha="center"
        )
    ax.set_title("Cider Label Embeddings")
    ax.axis("off")
    return fig


def print_graph(
    bottle_labels_embedded: Annotated[str, "Input"],
    graph_localtion: Annotated[str, "Output"],
) -> None:
    labels_df = pd.read_parquet(bottle_labels_embedded)

    tsne = TSNE(n_components=2, perplexity=10, max_iter=1024, verbose=1)

    X = np.vstack(labels_df["embeddings"])
    Y = tsne.fit_transform(X)
    fig = build_figure(Y, labels_df)
    fig.savefig(graph_localtion)


if __name__ == "__main__":
    typer.run(print_graph)
