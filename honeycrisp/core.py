import mlflow


def get_or_create_experiment_id(experiment: str) -> str:
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        exp_id = mlflow.create_experiment(experiment)
        return exp_id
    return exp.experiment_id


def get_run_id(experiment: str, run_name: str) -> str:
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"run_name='{run_name}'",
        max_results=1,
        output_format="list",
    )
    if not runs:
        return None
    return runs[0].info.run_id


def start_run(
    tracking_uri: str, experiment: str, run_name: str | None = None
) -> mlflow.ActiveRun:
    if not mlflow.is_tracking_uri_set():
        mlflow.set_tracking_uri(tracking_uri)
    current_run = mlflow.active_run()
    if current_run is not None:
        return current_run
    run_id = get_run_id(experiment, run_name)
    return mlflow.start_run(
        run_id=run_id,
        experiment_id=get_or_create_experiment_id(experiment),
        run_name=run_name,
        log_system_metrics=True,
    )
