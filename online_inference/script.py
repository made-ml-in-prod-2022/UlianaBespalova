import os

import boto3
import click
import requests
import pandas as pd

from pathlib import Path

from ml_project.enities import get_downloading_params

def download_dataset(s3_bucket: str, s3_access: str, output: str):
    session = boto3.session.Session()
    s3 = session.client(
        service_name='s3',
        endpoint_url=s3_access['endpoint_url'],
        aws_access_key_id=s3_access['pk'],
        aws_secret_access_key=s3_access['sk']
    )
    s3.download_file(s3_bucket, s3_access['path'], output)


@click.command()
@click.option("--config", default="config/config_get_data.yaml", type=str)
@click.option("--model", default="logreg", type=str)
@click.option("--ip", default="0.0.0.0")
@click.option("--port", default=8000)
def main(config, model, ip, port):

    downloading_params = get_downloading_params(config)
    if downloading_params is None:
        raise ValueError(f"Unable to load data {config}")

    path = os.path.join("online_inference/data",
                        Path(downloading_params.path).name)

    os.makedirs(downloading_params.output_folder, exist_ok=True)
    download_dataset(
        downloading_params.s3_bucket,
        {'endpoint_url': downloading_params.endpoint_url,
         'path': downloading_params.path,
         'pk': downloading_params.pk,
         'sk': downloading_params.sk},
        path,
        )

    if model != "logreg" and model != "forest":
        raise ValueError(f"Invalid model type {model}")
    model_name = model

    dataset = pd.read_csv(path)
    dataset.drop(["condition"], axis=1, inplace=True)
    data_json = {
        "data": dataset.values.tolist(),
        "features_names": dataset.columns.to_list(),
        "model": model_name,
    }
    # print("Data fot predicting:", data_json)
    response = requests.post(
        f"http://{ip}:{port}/predict",
        json=data_json,
    )
    print(f"Status code:\n{response.status_code}")
    print(f"Result:\n{response.json()}")


if __name__ == "__main__":
    main()
