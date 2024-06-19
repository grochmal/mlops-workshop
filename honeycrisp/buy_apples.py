from typing import Annotated, Optional

import mlflow
import pandas as pd
import typer
from tqdm import tqdm
from transformers.pipelines import Pipeline, pipeline

from honeycrisp.core import start_run

ZERO_SHOT_TASK = "zero-shot-classification"
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"
ZERO_SHOT_THRESHOLD = 0.5


def emails_to_purchases(
    emails: dict[str, str],
    apple_types: list[str],
    zero_shot_model: Pipeline,
) -> tuple[list[str], list[str]]:
    rejected_emails = []
    basement = []
    for email in tqdm(emails, desc="Processing emails"):
        labels = zero_shot_model(email, apple_types)
        if labels["scores"][0] < ZERO_SHOT_THRESHOLD:
            # the model is not confident, a human should have a look
            rejected_emails.append(email)
        else:
            basement.append(labels["labels"][0])
    return rejected_emails, basement


def buy_apples(
    experiment: str,
    emails_from_farmers: Annotated[str, "Input"],
    apple_types: Annotated[str, "Input"],
    rejected_mail: Annotated[str, "Output"],
    apples_in_basement: Annotated[str, "Output"],
    run_name: Optional[str] = None,
    tracking_uri: str = "https://dagshub.com/grochmal/mlops-workshop.mlflow",
) -> None:
    zero_shot = pipeline(ZERO_SHOT_TASK, model=ZERO_SHOT_MODEL)
    with start_run(tracking_uri=tracking_uri, experiment=experiment, run_name=run_name):
        emails_df = pd.read_parquet(emails_from_farmers)
        emails = emails_df["emails"]
        mlflow.log_param("number_of_emails", len(emails))
        apple_types_df = pd.read_parquet(apple_types)
        apple_types = list(apple_types_df["apple_types"])
        mlflow.log_param("apple_types", len(apple_types))
        # Validate outputs before doing any work
        pd.DataFrame().to_parquet(rejected_mail)
        pd.DataFrame().to_parquet(apples_in_basement)

        rejected_emails, basement = emails_to_purchases(emails, apple_types, zero_shot)

        mlflow.log_metric("rejected_emails", len(rejected_emails))
        pd.DataFrame({"emails": rejected_emails}).to_parquet(rejected_mail)
        mlflow.log_metric("apples_in_basement", len(basement))
        pd.DataFrame({"apples_in_basement": basement}).to_parquet(apples_in_basement)


if __name__ == "__main__":
    typer.run(buy_apples)
