from pathlib import Path
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import typer
from sklearn.manifold import TSNE

from honeycrisp.core import start_run


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
    experiment_id: str,
    bottle_labels_embedded: Annotated[str, "Input"],
    graph_localtion: Annotated[str, "Output"],
    run_name: Optional[str] = None,
    tracking_uri: str = "https://dagshub.com/grochmal/mlops-workshop.mlflow",
) -> None:
    with start_run(
        tracking_uri=tracking_uri, experiment=experiment_id, run_name=run_name
    ):
        labels_df = pd.read_parquet(bottle_labels_embedded)
        mlflow.log_param("number_of_labels", len(labels_df))
        # Validate outputs before doing any work
        Path(graph_localtion).touch()

        tsne = TSNE(n_components=2, perplexity=10, max_iter=1024, verbose=1)

        X = np.vstack(labels_df["embeddings"])
        Y = tsne.fit_transform(X)
        mlflow.log_param("reduced_size", Y.shape)
        mlflow.log_metric("KL_dicergence", tsne.kl_divergence_)
        fig = build_figure(Y, labels_df)
        fig.savefig(graph_localtion)


if __name__ == "__main__":
    typer.run(print_graph)
