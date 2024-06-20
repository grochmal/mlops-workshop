import pandas as pd
import typer
from transformers.pipelines import Pipeline, pipeline

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
    for email in emails:
        labels = zero_shot_model(email, apple_types)
        if labels["scores"][0] < ZERO_SHOT_THRESHOLD:
            # the model is not confident, a human should have a look
            rejected_emails.append(email)
        else:
            basement.append(labels["labels"][0])
    return rejected_emails, basement


def buy_apples(
    emails_from_farmers: str,
    apple_types: str,
    rejected_mail: str,
    apples_in_basement: str,
) -> None:
    zero_shot = pipeline(ZERO_SHOT_TASK, model=ZERO_SHOT_MODEL)
    emails_df = pd.read_parquet(emails_from_farmers)
    emails = emails_df["emails"]
    apple_types_df = pd.read_parquet(apple_types)
    apple_types = list(apple_types_df["apple_types"])

    rejected_emails, basement = emails_to_purchases(emails, apple_types, zero_shot)

    pd.DataFrame({"emails": rejected_emails}).to_parquet(rejected_mail)
    pd.DataFrame({"apples_in_basement": basement}).to_parquet(apples_in_basement)


if __name__ == "__main__":
    typer.run(buy_apples)
