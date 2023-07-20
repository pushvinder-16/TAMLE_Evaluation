import mlflow

from housepred.api import data_prep_main, score_main, train_main


def main():
    # Set the remote server URI
    remote_server_uri = "http://localhost:5001/"  # set to your server URI
    mlflow.set_tracking_uri(remote_server_uri)  # or set the MLFLOW_TRACKING_URI in the env

    exp_name = "house_price_prediction"
    mlflow.set_experiment(exp_name)

    # Create a parent mlflow run
    with mlflow.start_run(run_name="Main Run") as parent_run:
        # Get the ID of the parent run
        parent_id = parent_run.info.run_id

        # Data Preparation
        with mlflow.start_run(run_name="Data Preparation", nested=True):
            data, model_path, prep_info = data_prep_main(
                "/home/pushvinder/mle_training/data/raw/",
                "/home/pushvinder/mle_training/data/processed",
            )

            mlflow.log_param("number_of_samples", prep_info["num_samples"])
            mlflow.log_param("number_of_features", prep_info["num_features"])
            mlflow.log_param("missing_values", prep_info["missing_values"])
            mlflow.log_param("parent_id", parent_id)

        # Model Training
        with mlflow.start_run(run_name="Model Training", nested=True):
            input_dir, output_dir, train_info = train_main(data, model_path)

            mlflow.log_params(train_info["best_params_grid_search"])
            mlflow.log_metric("training_loss", train_info["train_loss"])
            mlflow.log_metric("validation_loss", train_info["validation_loss"])

        # Model Evaluation
        with mlflow.start_run(run_name="Model Evaluation", nested=True):
            report, eval_info = score_main(input_dir, output_dir)
            mlflow.log_metric("final_rmse", report)


if __name__ == "__main__":
    main()
