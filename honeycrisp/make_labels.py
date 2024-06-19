from dataclasses import dataclass
from typing import Annotated, Optional

import mlflow
import numpy as np
import pandas as pd
import typer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer

from honeycrisp.core import start_run

LABEL_PROMPT = "The label on a bottle of cider made out of {apple} apples reads:"
EMBEDDING_MODEL = "sentence-transformers/sentence-t5-large"


@dataclass
class CiderLabel:
    apple: str
    label: str
    embedding: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"CiderLabel {self.apple}: {self.label}"

    def __str__(self) -> str:
        return self.__repr__()


def cider_labels(
    gpt2_model: AutoModelForCausalLM,
    gpt2_tokenizer: AutoTokenizer,
    apples: list[str],
) -> list[CiderLabel]:
    labels = []
    for apple in tqdm(apples, desc="Generating bottle labels"):
        prompt = LABEL_PROMPT.format(apple=apple)
        inputs = gpt2_tokenizer(prompt, return_tensors="pt")
        outputs = gpt2_model.generate(
            **inputs,
            max_new_tokens=36,
            do_sample=True,
            temperature=0.6,
            top_p=0.7,
            top_k=10,
            return_dict_in_generate=True,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
        input_length = inputs.input_ids.shape[1]
        tokens = outputs.sequences[0, input_length:]
        output_str = gpt2_tokenizer.decode(tokens)
        labels.append(CiderLabel(apple=apple, label=output_str))
    return labels


def label_to_embedding(
    embedding_model: SentenceTransformer, labels: list[CiderLabel]
) -> None:
    for cl in tqdm(labels, desc="Embedding labels"):
        cl.embedding = embedding_model.encode(cl.label, convert_to_numpy=True)
    return labels


def make_labels(
    experiment: str,
    apples_in_basemet: Annotated[str, "Input"],
    bottle_labels_embedded: Annotated[str, "Output"],
    gen_model: Optional[str] = "gpt2-large",
    run_name: Optional[str] = None,
    tracking_uri: str = "https://dagshub.com/grochmal/mlops-workshop.mlflow",
) -> None:
    gpt2_tokenizer = AutoTokenizer.from_pretrained(gen_model)
    gpt2_model = AutoModelForCausalLM.from_pretrained(gen_model)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    with start_run(tracking_uri=tracking_uri, experiment=experiment, run_name=run_name):
        mlflow.log_param("gen_model", gen_model)
        apples_df = pd.read_parquet(apples_in_basemet)
        apples = apples_df["apples_in_basement"].tolist()
        mlflow.log_param("number_of_apples", len(apples))
        pd.DataFrame().to_parquet(bottle_labels_embedded)

        bottle_labels = cider_labels(gpt2_model, gpt2_tokenizer, apples)
        embedded_labels = label_to_embedding(embedding_model, bottle_labels)

        embedded_labels_df = pd.DataFrame(
            {
                "apples": [cl.apple for cl in embedded_labels],
                "labels": [cl.label for cl in embedded_labels],
                "embeddings": [cl.embedding for cl in embedded_labels],
            }
        )
        embedded_labels_df.to_parquet(bottle_labels_embedded)
        mlflow.log_metric("number_of_labels", len(embedded_labels))


if __name__ == "__main__":
    typer.run(make_labels)
