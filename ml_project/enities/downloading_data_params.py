import yaml
from marshmallow import EXCLUDE
from marshmallow_dataclass import class_schema

from dataclasses import dataclass, field


@dataclass()
class DownloadParams:
    endpoint_url: str = field(default="https://ib.bizmrg.com")
    path: str = field(default="heart_cleveland_upload.csv")
    pk: str = field(default="r3NQGAzctTMwnxyN7SCLmV")
    sk: str = field(default="hzV7YbiKYVwDzSyTZe2eMjWZuuA9GaGysX6kG3ntAJmS")
    output_folder: str = field(default="data/")
    s3_bucket: str = field(default="fir_backet")


def get_downloading_params(path: str) -> DownloadParams:
    schema = class_schema(DownloadParams)
    with open(path, "r") as input:
        config_params = yaml.safe_load(input)
        return schema().load(config_params, unknown=EXCLUDE)
