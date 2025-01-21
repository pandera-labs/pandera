# pylint: disable=missing-class-docstring,too-few-public-methods
"""Test conversion of data from and to different formats."""

import io
import tempfile
from typing import Any

import pandas as pd
import pytest

import pandera as pa
from pandera.engines import pandas_engine


class InSchema(pa.DataFrameModel):
    str_col: pa.typing.Series[str] = pa.Field(unique=True, isin=[*"abcd"])
    int_col: pa.typing.Series[int]


class InSchemaCsv(InSchema):
    class Config:
        from_format = "csv"


class InSchemaCsvCallable(InSchema):
    class Config:
        from_format = pd.read_csv


class InSchemaDict(InSchema):
    class Config:
        from_format = "dict"


class InSchemaDictKwargs(InSchema):
    class Config:
        from_format = "dict"
        from_format_kwargs = {"orient": "index"}


class InSchemaJson(InSchema):
    class Config:
        from_format = "json"
        from_format_kwargs = {"orient": "records"}


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


class InSchemaJsonNormalize(InSchema):
    class Config:
        from_format = "json_normalize"


class OutSchema(InSchema):
    float_col: pa.typing.Series[float]


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
        to_format_kwargs = {"orient": "records"}


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


def mock_dataframe() -> pd.DataFrame:
    """Create a valid mock dataframe."""
    return pd.DataFrame({"str_col": ["a"], "int_col": [1]})


def invalid_input_dataframe() -> pd.DataFrame:
    """Invalid data under the InSchema* models defined above."""
    return pd.DataFrame({"str_col": ["a"]})


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
        [InSchemaCsv, lambda df, x: df.to_csv(x, index=None), io.StringIO],
        [
            InSchemaCsvCallable,
            lambda df, x: df.to_csv(x, index=None),
            io.StringIO,
        ],
        [InSchemaDict, lambda df: df.to_dict(orient="records"), None],
        [InSchemaDictKwargs, lambda df: df.to_dict(orient="index"), None],
        [
            InSchemaJson,
            lambda df, x: df.to_json(x, orient="records"),
            io.StringIO,
        ],
        [InSchemaJson, lambda df: df.to_json(orient="records"), None],
        [InSchemaFeather, lambda df, x: df.to_feather(x), io.BytesIO],
        [InSchemaParquet, lambda df, x: df.to_parquet(x), io.BytesIO],
        [InSchemaPickle, lambda df, x: df.to_pickle(x), io.BytesIO],
        [InSchemaPickleCallable, lambda df, x: df.to_pickle(x), io.BytesIO],
        [InSchemaJsonNormalize, lambda df: df.to_dict(orient="records"), None],
    ],
)
def test_from_format(schema, to_fn, buf_cls):
    """
    Test that check_types-guarded function reads data from source serialization
    format.
    """

    @pa.check_types
    def fn(df: pa.typing.DataFrame[schema]):
        return df

    for df, invalid in [
        (mock_dataframe(), False),
        (invalid_input_dataframe(), True),
    ]:
        buf = None if buf_cls is None else buf_cls()

        if _needs_pyarrow(schema):
            with pytest.raises(ImportError):
                to_fn(df, *([buf] if buf else []))
        else:
            arg = to_fn(df, *([buf] if buf else []))
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
            assert df.equals(out)


def custom_pickle_file_reader(fp):
    """Custom file reader that closes the file handle."""
    out = pd.read_pickle(fp)
    fp.close()
    return out


@pytest.mark.parametrize(
    "schema,from_fn,buf_cls",
    [
        [OutSchemaCsv, pd.read_csv, io.StringIO],
        [OutSchemaCsvCallable, pd.read_csv, io.StringIO],
        [OutSchemaDict, pd.DataFrame, None],
        [OutSchemaJson, lambda x: pd.read_json(x, orient="records"), None],
        [OutSchemaFeather, pd.read_feather, io.BytesIO],
        [OutSchemaParquet, pd.read_parquet, io.BytesIO],
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
    def fn(df: pa.typing.DataFrame[InSchema]) -> pa.typing.DataFrame[schema]:
        return df.assign(float_col=1.1)  # type: ignore

    @pa.check_types
    def invalid_fn(
        df: pa.typing.DataFrame[InSchema],
    ) -> pa.typing.DataFrame[schema]:
        return df

    df = mock_dataframe()

    if _needs_pyarrow(schema):
        with pytest.raises(ImportError):
            fn(df)
        return

    try:
        out = fn(df)
    except OSError:
        pytest.skip(
            f"pandas=={pd.__version__} automatically closes the buffer, for "
            "more details see: "
            "https://github.com/pandas-dev/pandas/issues/35679"
        )
    if buf_cls and not isinstance(out, buf_cls):
        out = buf_cls(out)
    out_df = from_fn(out)
    assert df.assign(float_col=1.1).equals(out_df)

    with pytest.raises(pa.errors.SchemaError):
        invalid_fn(df)
