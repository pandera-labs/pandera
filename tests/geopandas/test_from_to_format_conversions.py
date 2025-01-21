# pylint: disable=missing-class-docstring,too-few-public-methods
"""Test conversion of GeoDataFrame from and to different formats."""

import io
import json
import tempfile
from typing import Any

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

import pandera as pa
from pandera.engines import pandas_engine
from pandera.typing.geopandas import GeoDataFrame, GeoSeries


class InSchema(pa.DataFrameModel):
    geometry: GeoSeries
    int_col: int


class InSchemaNone(InSchema):
    class Config:
        from_format = None


class InSchemaCsv(InSchema):
    class Config:
        from_format = "csv"


class InSchemaCsvCallable(InSchema):
    class Config:
        from_format = pd.read_csv


class InSchemaDict(InSchema):
    class Config:
        from_format = "dict"


class InSchemaJson(InSchema):
    class Config:
        from_format = "json"


class InSchemaFeather(InSchema):
    class Config:
        from_format = "feather"


class InSchemaParquet(InSchema):
    class Config:
        from_format = "parquet"


class InSchemaPickle(InSchema):
    class Config:
        from_format = "pickle"


class InSchemaPickleCallable(InSchema):
    class Config:
        from_format = pd.read_pickle


class OutSchema(InSchema):
    float_col: float


class OutSchemaNone(OutSchema):
    class Config:
        to_format = None


class OutSchemaCsv(OutSchema):
    class Config:
        to_format = "csv"
        to_format_kwargs = {"index": None}


def custom_to_csv(data: Any, *args, **kwargs):
    """
    Function to save data to csv, used in to_format function.

    Args:
        data: The data object to be saved.

    Returns:
        Returns none if file path provided, else to_format buffer
    """
    return data.to_csv(*args, **kwargs)


class OutSchemaCsvCallable(OutSchema):
    class Config:
        to_format = custom_to_csv
        to_format_kwargs = {"index": None}


class OutSchemaDict(OutSchema):
    class Config:
        to_format = "dict"
        to_format_kwargs = {"orient": "records"}


class OutSchemaJson(OutSchema):
    class Config:
        to_format = "json"


class OutSchemaFeather(OutSchema):
    class Config:
        to_format = "feather"


class OutSchemaParquet(OutSchema):
    class Config:
        to_format = "parquet"


class OutSchemaPickle(OutSchema):
    class Config:
        to_format = "pickle"


def custom_to_pickle(data: Any, *args, **kwargs):
    """
    Function to save data to pickle, used in to_format function.

    Args:
        data: The data object to be saved.

    Returns:
        Returns none if file path provided, else to_format buffer
    """
    return data.to_pickle(*args, **kwargs)


class OutSchemaPickleCallable(OutSchema):
    class Config:
        to_format = custom_to_pickle
        to_format_buffer = io.BytesIO


def custom_to_buffer():
    """Creates a file handle to a temporary file."""
    return tempfile.NamedTemporaryFile()


class OutSchemaPickleCallableWithFile(OutSchema):
    class Config:
        to_format = custom_to_pickle
        to_format_buffer = custom_to_buffer


def mock_dataframe() -> gpd.GeoDataFrame:
    """Create a valid mock dataframe."""
    return gpd.GeoDataFrame({"geometry": [Point([0, 1])], "int_col": [1]})


def invalid_input_dataframe() -> gpd.GeoDataFrame:
    """Invalid data under the InSchema* models defined above."""
    return gpd.GeoDataFrame({"geometry": [Point([0, 1])], "str_col": ["a"]})


def _needs_pyarrow(schema) -> bool:
    return (
        schema
        in {
            InSchemaParquet,
            InSchemaFeather,
            OutSchemaParquet,
            OutSchemaFeather,
        }
        and not pandas_engine.PYARROW_INSTALLED
    )


@pytest.mark.parametrize(
    "schema,to_fn,buf_cls",
    [
        [InSchemaNone, lambda gdf: gdf, None],
        [InSchemaCsv, lambda gdf, x: gdf.to_csv(x, index=None), io.StringIO],
        [
            InSchemaCsvCallable,
            lambda gdf, x: gdf.to_csv(x, index=None),
            io.StringIO,
        ],
        [InSchemaDict, lambda gdf: gdf.to_dict(orient="records"), None],
        [InSchemaJson, lambda gdf: gdf.to_json(), None],
        [InSchemaFeather, lambda gdf, x: gdf.to_feather(x), io.BytesIO],
        [InSchemaParquet, lambda gdf, x: gdf.to_parquet(x), io.BytesIO],
        [InSchemaPickle, lambda gdf, x: gdf.to_pickle(x), io.BytesIO],
        [InSchemaPickleCallable, lambda gdf, x: gdf.to_pickle(x), io.BytesIO],
    ],
)
def test_from_format(schema, to_fn, buf_cls):
    """
    Test that check_types-guarded function reads data from source serialization
    format.
    """

    @pa.check_types
    def fn(gdf: GeoDataFrame[schema]):
        return gdf

    for gdf, invalid in [
        (mock_dataframe(), False),
        (invalid_input_dataframe(), True),
    ]:
        buf = None if buf_cls is None else buf_cls()

        if _needs_pyarrow(schema):
            with pytest.raises(ImportError):
                to_fn(gdf, *([buf] if buf else []))
        else:
            arg = to_fn(gdf, *([buf] if buf else []))
            if buf:
                if buf.closed:
                    pytest.skip(
                        "skip test for older pandas versions where to_pickle "
                        "closes user-provided buffers: "
                        "https://github.com/pandas-dev/pandas/issues/35679"
                    )
                buf.seek(0)
                arg = buf
            if invalid:
                with pytest.raises(pa.errors.SchemaError):
                    fn(arg)
                return

            out = fn(arg)
            assert gdf.equals(out)


def custom_pickle_file_reader(fp):
    """Custom file reader that closes the file handle."""
    out = pd.read_pickle(fp)
    fp.close()
    return out


def df2gdf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert pd.DataFrame to gpd.GeoDataFrame for testing purposes"""
    # For some unknown reason, GeoPandas implements its own
    # to_* methods but not {read|from}_* methods, so we're
    # left to manually convert DataFrame inputs to GeoDataFrame.
    # This wrapper is handled internally in the GeoDataFrame
    # model, but for testing we need to explicitly wrap so
    # proper pre/post comparisons can be made.
    return GeoDataFrame._coerce_geometry(df)


@pytest.mark.parametrize(
    "schema,from_fn,buf_cls",
    [
        [OutSchemaNone, gpd.GeoDataFrame, None],
        [OutSchemaCsv, lambda x: df2gdf(pd.read_csv(x)), io.StringIO],
        [OutSchemaCsvCallable, lambda x: df2gdf(pd.read_csv(x)), io.StringIO],
        [OutSchemaDict, gpd.GeoDataFrame, None],
        [
            OutSchemaJson,
            lambda x: gpd.GeoDataFrame.from_features(json.loads(x)),
            None,
        ],
        [OutSchemaFeather, lambda x: df2gdf(pd.read_feather(x)), io.BytesIO],
        [OutSchemaParquet, lambda x: df2gdf(pd.read_parquet(x)), io.BytesIO],
        [OutSchemaPickle, pd.read_pickle, io.BytesIO],
        [OutSchemaPickleCallable, pd.read_pickle, io.BytesIO],
        [
            OutSchemaPickleCallableWithFile,
            custom_pickle_file_reader,
            tempfile._TemporaryFileWrapper,
        ],  # noqa
    ],
)
def test_to_format(schema, from_fn, buf_cls):
    """
    Test that check_types-guarded function writes data to source serialization
    format.
    """

    @pa.check_types
    def fn(gdf: GeoDataFrame[InSchema]) -> GeoDataFrame[schema]:
        return gdf.assign(float_col=1.1)  # type: ignore

    @pa.check_types
    def invalid_fn(
        gdf: GeoDataFrame[InSchema],
    ) -> GeoDataFrame[schema]:
        return gdf

    gdf = mock_dataframe()

    if _needs_pyarrow(schema):
        with pytest.raises(ImportError):
            fn(gdf)
        return

    try:
        out = fn(gdf)
    except OSError:
        pytest.skip(
            f"pandas=={pd.__version__} automatically closes the buffer, for "
            "more details see: "
            "https://github.com/pandas-dev/pandas/issues/35679"
        )
    if buf_cls and not isinstance(out, buf_cls):
        out = buf_cls(out)
    out_gdf = from_fn(out)

    assert gdf.assign(float_col=1.1).equals(out_gdf)

    with pytest.raises(pa.errors.SchemaError):
        invalid_fn(gdf)
