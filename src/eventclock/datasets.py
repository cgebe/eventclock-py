from importlib import resources
from importlib.resources.abc import Traversable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _data_dir() -> Traversable:
    return resources.files("eventclock").joinpath("data")


def dataset_names() -> list[str]:
    return sorted(
        p.name.removesuffix(".parquet")
        for p in _data_dir().iterdir()
        if p.name.endswith(".parquet")
    )


def load_dataset(name: str) -> pd.DataFrame:
    names = dataset_names()
    if name not in names:
        raise ValueError(f"Dataset {name!r} not found. Available: {', '.join(names)}.")
    with _data_dir().joinpath(f"{name}.parquet").open("rb") as f:
        table = pq.read_table(f)
    df = table.to_pandas(date_as_object=True)
    for field in table.schema:
        if pa.types.is_timestamp(field.type):
            s = df[field.name]
            if s.dt.tz is None:
                s = s.dt.tz_localize("UTC")
            df[field.name] = s.dt.as_unit("ns")
    return df