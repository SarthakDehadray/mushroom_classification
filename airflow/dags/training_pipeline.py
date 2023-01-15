from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os
from airflow import DAG 
from airflow.operators.python import PythonOperator

with DAG(
    "mushroom_prediction",
    default_args = {"retries":2},
    #[END default args]
    description = "mushroom classification",
    schedule_interval = "@weekly",
    start_date = pendulum.datetime(2023,1,7,tz = "UTC"),
    catchup = False,
    tags = ["example"],

) as dag :


    def training(**kwargs):
        from mushroom.pipeline.training_pipeline import start_training_pipeline
        start_training_pipeline()

    def sync_artifact_to_s3_bucket(**kwargs):
        bucket_name = os.getenv("BUCKET_NAME")
        os.system(f"aws s3 sync /app/artifacts3://{bucket_name}/artifacts")
        os.system(f"aws s3 sync /app/saved_modelss3://{bucket_name}/saved_models")
        
        training_pipeline = PythonOperator(
            task_id = "train_pipeline",
            python_callable = training
        )

        sync_data_to_s3 = PythonOperator(
            task_id = "training_pipeline",
            pyhthon_callable = training
        )

        training_pipeline  >> sync_data_to_s3

