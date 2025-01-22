"""Test typing annotations for the model api."""

# pylint:disable=missing-class-docstring,too-few-public-methods
import re
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np
import pandas as pd
import pytest

import pandera as pa
from pandera.dtypes import DataType
from pandera.typing import DataFrame, Index, Series

try:  # python 3.9+
    from typing import Annotated  # type: ignore
except ImportError:
    from typing_extensions import Annotated  # type: ignore


class SchemaBool(pa.DataFrameModel):
    col: Series[pa.typing.Bool]


class SchemaDateTime(pa.DataFrameModel):
    col: Series[pa.typing.DateTime]


class SchemaCategory(pa.DataFrameModel):
    col: Series[pa.typing.Category]


class SchemaFloat(pa.DataFrameModel):
    col: Series[pa.typing.Float]


class SchemaFloat16(pa.DataFrameModel):
    col: Series[pa.typing.Float16]


class SchemaFloat32(pa.DataFrameModel):
    col: Series[pa.typing.Float32]


class SchemaFloat64(pa.DataFrameModel):
    col: Series[pa.typing.Float64]


class SchemaInt(pa.DataFrameModel):
    col: Series[pa.typing.Int]


class SchemaInt8(pa.DataFrameModel):
    col: Series[pa.typing.Int8]


class SchemaUInt8(pa.DataFrameModel):
    col: Series[pa.typing.UInt8]


class SchemaInt16(pa.DataFrameModel):
    col: Series[pa.typing.Int16]


class SchemaUInt16(pa.DataFrameModel):
    col: Series[pa.typing.UInt16]


class SchemaInt32(pa.DataFrameModel):
    col: Series[pa.typing.Int32]


class SchemaUInt32(pa.DataFrameModel):
    col: Series[pa.typing.UInt32]


class SchemaInt64(pa.DataFrameModel):
    col: Series[pa.typing.Int64]


class SchemaUInt64(pa.DataFrameModel):
    col: Series[pa.typing.UInt64]


class SchemaObject(pa.DataFrameModel):
    col: Series[pa.typing.Object]


class SchemaString(pa.DataFrameModel):
    col: Series[pa.typing.String]


class SchemaTimedelta(pa.DataFrameModel):
    col: Series[pa.typing.Timedelta]


class SchemaINT8(pa.DataFrameModel):
    col: Series[pa.typing.INT8]


class SchemaUINT8(pa.DataFrameModel):
    col: Series[pa.typing.UINT8]


class SchemaINT16(pa.DataFrameModel):
    col: Series[pa.typing.INT16]


class SchemaUINT16(pa.DataFrameModel):
    col: Series[pa.typing.UINT16]


class SchemaINT32(pa.DataFrameModel):
    col: Series[pa.typing.INT32]


class SchemaUINT32(pa.DataFrameModel):
    col: Series[pa.typing.UINT32]


class SchemaINT64(pa.DataFrameModel):
    col: Series[pa.typing.INT64]


class SchemaUINT64(pa.DataFrameModel):
    col: Series[pa.typing.UINT64]


def _test_literal_pandas_dtype(
    model: Type[pa.DataFrameModel], pandas_dtype: DataType
):
    schema = model.to_schema()
    expected = pa.Column(pandas_dtype, name="col").dtype
    assert schema.columns["col"].dtype == expected


@pytest.mark.parametrize(
    "model, pandas_dtype",
    [
        (SchemaBool, pa.Bool),
        (SchemaDateTime, pa.DateTime),
        (SchemaCategory, pa.Category),
        (SchemaFloat, pa.Float),
        (SchemaFloat16, pa.Float16),
        (SchemaFloat32, pa.Float32),
        (SchemaFloat64, pa.Float64),
        (SchemaInt, pa.Int),
        (SchemaInt8, pa.Int8),
        (SchemaInt16, pa.Int16),
        (SchemaInt32, pa.Int32),
        (SchemaInt64, pa.Int64),
        (SchemaUInt8, pa.UInt8),
        (SchemaUInt16, pa.UInt16),
        (SchemaUInt32, pa.UInt32),
        (SchemaUInt64, pa.UInt64),
        (SchemaObject, pa.Object),
        (SchemaString, pa.String),
        (SchemaTimedelta, pa.Timedelta),
    ],
)
def test_literal_legacy_pandas_dtype(
    model: Type[pa.DataFrameModel], pandas_dtype: DataType
):
    """Test literal annotations with the legacy pandas dtypes."""
    _test_literal_pandas_dtype(model, pandas_dtype)


@pytest.mark.parametrize(
    "model, pandas_dtype",
    [
        (SchemaUINT8, pa.UINT8),
        (SchemaUINT16, pa.UINT16),
        (SchemaUINT32, pa.UINT32),
        (SchemaUINT64, pa.UINT64),
        (SchemaUINT8, pa.UINT8),
        (SchemaUINT16, pa.UINT16),
        (SchemaUINT32, pa.UINT32),
        (SchemaUINT64, pa.UINT64),
    ],
)
def test_literal_new_pandas_dtype(
    model: Type[pa.DataFrameModel], pandas_dtype: DataType
):
    """Test literal annotations with the new nullable pandas dtypes."""
    _test_literal_pandas_dtype(model, pandas_dtype)


class SchemaFieldCategoricalDtype(pa.DataFrameModel):
    col: Series[pd.CategoricalDtype] = pa.Field(
        dtype_kwargs={"categories": ["b", "a"], "ordered": True}
    )


def _test_annotated_dtype(
    model: Type[pa.DataFrameModel],
    dtype: Type,
    dtype_kwargs: Optional[Dict[str, Any]] = None,
):
    dtype_kwargs = dtype_kwargs or {}
    schema = model.to_schema()

    actual = schema.columns["col"].dtype
    expected = pa.Column(dtype(**dtype_kwargs), name="col").dtype
    assert actual == expected


def _test_default_annotated_dtype(
    model: Type[pa.DataFrameModel], dtype: Any, has_mandatory_args: bool
):
    if has_mandatory_args:
        err_msg = "cannot be instantiated"
        with pytest.raises(TypeError, match=err_msg):
            model.to_schema()
    else:
        _test_annotated_dtype(model, dtype)


class SchemaFieldDatetimeTZDtype(pa.DataFrameModel):
    col: Series[pd.DatetimeTZDtype] = pa.Field(
        dtype_kwargs={"unit": "ns", "tz": "EST"}
    )


class SchemaFieldIntervalDtype(pa.DataFrameModel):
    col: Series[pd.IntervalDtype] = pa.Field(dtype_kwargs={"subtype": "int32"})


class SchemaFieldPeriodDtype(pa.DataFrameModel):
    col: Series[pd.PeriodDtype] = pa.Field(dtype_kwargs={"freq": "D"})


class SchemaFieldSparseDtype(pa.DataFrameModel):
    col: Series[pd.SparseDtype] = pa.Field(
        dtype_kwargs={"dtype": np.int32, "fill_value": 0}
    )


@pytest.mark.parametrize(
    "model, dtype, dtype_kwargs",
    [
        (
            SchemaFieldCategoricalDtype,
            pd.CategoricalDtype,
            {"categories": ["b", "a"], "ordered": True},
        ),
        (
            SchemaFieldDatetimeTZDtype,
            pd.DatetimeTZDtype,
            {"unit": "ns", "tz": "EST"},
        ),
        (SchemaFieldIntervalDtype, pd.IntervalDtype, {"subtype": "int32"}),
        (SchemaFieldPeriodDtype, pd.PeriodDtype, {"freq": "D"}),
        (
            SchemaFieldSparseDtype,
            pd.SparseDtype,
            {"dtype": np.int32, "fill_value": 0},
        ),
    ],
)
def test_parametrized_pandas_extension_dtype_field(
    model: Type[pa.DataFrameModel], dtype: Type, dtype_kwargs: Dict[str, Any]
):
    """Test type annotations for parametrized pandas extension dtypes."""
    _test_annotated_dtype(model, dtype, dtype_kwargs)


class SchemaDefaultCategoricalDtype(pa.DataFrameModel):
    col: Series[pd.CategoricalDtype]


class SchemaDefaultDatetimeTZDtype(pa.DataFrameModel):
    col: Series[pd.DatetimeTZDtype]


class SchemaDefaultIntervalDtype(pa.DataFrameModel):
    col: Series[pd.IntervalDtype]


class SchemaDefaultPeriodDtype(pa.DataFrameModel):
    col: Series[pd.PeriodDtype]


class SchemaDefaultSparseDtype(pa.DataFrameModel):
    col: Series[pd.SparseDtype]


@pytest.mark.parametrize(
    "model, dtype, has_mandatory_args",
    [
        (SchemaDefaultCategoricalDtype, pd.CategoricalDtype, False),
        # DatetimeTZDtype: tz is implictly required
        (SchemaDefaultDatetimeTZDtype, pd.DatetimeTZDtype, True),
        (SchemaDefaultIntervalDtype, pd.IntervalDtype, False),
        # PeriodDtype: freq is implicitely required -> str(pd.PeriodDtype())
        # raises AttributeError
        (SchemaDefaultPeriodDtype, pd.PeriodDtype, True),
        (SchemaDefaultSparseDtype, pd.SparseDtype, False),
    ],
)
def test_legacy_default_pandas_extension_dtype(
    model, dtype: pd.core.dtypes.base.ExtensionDtype, has_mandatory_args: bool
):
    """Test type annotations for default pandas extension dtypes."""
    _test_default_annotated_dtype(model, dtype, has_mandatory_args)


class SchemaAnnotatedCategoricalDtype(pa.DataFrameModel):
    col: Series[Annotated[pd.CategoricalDtype, ["b", "a"], True]]


class SchemaAnnotatedDatetimeTZDtype(pa.DataFrameModel):
    col: Series[Annotated[pd.DatetimeTZDtype, "ns", "est"]]


if pa.PANDAS_1_3_0_PLUS:

    class SchemaAnnotatedIntervalDtype(pa.DataFrameModel):
        col: Series[Annotated[pd.IntervalDtype, "int32", "both"]]

else:

    class SchemaAnnotatedIntervalDtype(pa.DataFrameModel):  # type: ignore
        col: Series[Annotated[pd.IntervalDtype, "int32"]]


class SchemaAnnotatedPeriodDtype(pa.DataFrameModel):
    col: Series[Annotated[pd.PeriodDtype, "D"]]


class SchemaAnnotatedSparseDtype(pa.DataFrameModel):
    col: Series[Annotated[pd.SparseDtype, np.int32, 0]]


@pytest.mark.parametrize(
    "model, dtype, dtype_kwargs",
    [
        (
            SchemaAnnotatedCategoricalDtype,
            pd.CategoricalDtype,
            {"categories": ["b", "a"], "ordered": True},
        ),
        (
            SchemaAnnotatedDatetimeTZDtype,
            pd.DatetimeTZDtype,
            {"unit": "ns", "tz": "EST"},
        ),
        (
            SchemaAnnotatedIntervalDtype,
            pd.IntervalDtype,
            (
                {"subtype": "int32", "closed": "both"}
                if pa.PANDAS_1_3_0_PLUS
                else {"subtype": "int32"}
            ),
        ),
        (SchemaAnnotatedPeriodDtype, pd.PeriodDtype, {"freq": "D"}),
        (
            SchemaAnnotatedSparseDtype,
            pd.SparseDtype,
            {"dtype": np.int32, "fill_value": 0},
        ),
    ],
)
def test_annotated_dtype(
    model: Type[pa.DataFrameModel],
    dtype: Type,
    dtype_kwargs: Dict[str, Any],
):
    """Test type annotations for parametrized pandas extension dtypes."""
    _test_annotated_dtype(model, dtype, dtype_kwargs)


class SchemaInvalidAnnotatedDtype(pa.DataFrameModel):
    col: Series[Annotated[pd.DatetimeTZDtype, "utc"]]


def test_invalid_annotated_dtype():
    """
    Test incorrect number of parameters for parametrized pandas extension
    dtypes.
    """
    err_msg = re.escape(
        "Annotation 'DatetimeTZDtype' requires all "
        r"positional arguments ['unit', 'tz']."
    )
    with pytest.raises(TypeError, match=err_msg):
        SchemaInvalidAnnotatedDtype.to_schema()


class SchemaRedundantField(pa.DataFrameModel):
    col: Series[Annotated[pd.DatetimeTZDtype, "utc"]] = pa.Field(
        dtype_kwargs={"tz": "utc"}
    )


def test_pandas_extension_dtype_redundant_field():
    """
    Test incorrect number of parameters for parametrized pandas extension
    dtypes.
    """
    err_msg = r"Cannot specify redundant 'dtype_kwargs' for"
    with pytest.raises(TypeError, match=err_msg):
        SchemaRedundantField.to_schema()


class SchemaInt8Dtype(pa.DataFrameModel):
    col: Series[pd.Int8Dtype]


class SchemaInt16Dtype(pa.DataFrameModel):
    col: Series[pd.Int16Dtype]


class SchemaInt32Dtype(pa.DataFrameModel):
    col: Series[pd.Int32Dtype]


class SchemaInt64Dtype(pa.DataFrameModel):
    col: Series[pd.Int64Dtype]


class SchemaUInt8Dtype(pa.DataFrameModel):
    col: Series[pd.UInt8Dtype]


class SchemaUInt16Dtype(pa.DataFrameModel):
    col: Series[pd.UInt16Dtype]


class SchemaUInt32Dtype(pa.DataFrameModel):
    col: Series[pd.UInt32Dtype]


class SchemaUInt64Dtype(pa.DataFrameModel):
    col: Series[pd.UInt64Dtype]


class SchemaStringDtype(pa.DataFrameModel):
    col: Series[pd.StringDtype]


class SchemaBooleanDtype(pa.DataFrameModel):
    col: Series[pd.BooleanDtype]


@pytest.mark.parametrize(
    "model, dtype, has_mandatory_args",
    [
        (SchemaInt8Dtype, pd.Int8Dtype, False),
        (SchemaInt16Dtype, pd.Int16Dtype, False),
        (SchemaInt32Dtype, pd.Int32Dtype, False),
        (SchemaInt64Dtype, pd.Int64Dtype, False),
        (SchemaUInt8Dtype, pd.UInt8Dtype, False),
        (SchemaUInt16Dtype, pd.UInt16Dtype, False),
        (SchemaUInt32Dtype, pd.UInt32Dtype, False),
        (SchemaUInt64Dtype, pd.UInt64Dtype, False),
        (SchemaStringDtype, pd.StringDtype, False),
        (SchemaBooleanDtype, pd.BooleanDtype, False),
    ],
)
def test_new_pandas_extension_dtype_class(
    model,
    dtype: pd.core.dtypes.base.ExtensionDtype,
    has_mandatory_args: bool,
):
    """Test type annotations with the new nullable pandas dtypes."""
    _test_default_annotated_dtype(model, dtype, has_mandatory_args)


class InitSchema(pa.DataFrameModel):
    col1: Series[int]
    col2: Series[float]
    col3: Series[str]
    index: Index[int]


def test_init_pandas_dataframe():
    """Test initialization of pandas.typing.DataFrame with Schema."""
    assert isinstance(
        DataFrame[InitSchema]({"col1": [1], "col2": [1.0], "col3": ["1"]}),
        DataFrame,
    )


@pytest.mark.parametrize(
    "invalid_data",
    [
        {"col1": [1.0], "col2": [1.0], "col3": ["1"]},
        {"col1": [1], "col2": [1], "col3": ["1"]},
        {"col1": [1], "col2": [1.0], "col3": [1]},
        {"col1": [1]},
    ],
)
def test_init_pandas_dataframe_errors(invalid_data):
    """Test errors from initializing a pandas.typing.DataFrame with Schema."""
    with pytest.raises(pa.errors.SchemaError):
        DataFrame[InitSchema](invalid_data)


class ExampleNamedTuple(NamedTuple):
    a: int
    b: float


class SchemaComplexPythonCollectionTypes(pa.DataFrameModel):
    list: List[pa.typing.Int32]
    dict: Dict[str, pa.typing.Int32]
    tuple2: Tuple[pa.typing.Int32, pa.typing.Int32]
    named_tuple: ExampleNamedTuple


def test_complex_python_collection_types():
    """Test complex python collection types."""
    assert isinstance(
        DataFrame[SchemaComplexPythonCollectionTypes](
            {
                "list": [[1, 2], [3, 4, 5]],
                "dict": [{"a": 1, "b": 2}, {"c": 1, "d": 2}],
                "tuple2": [[6, 7], [8, 9]],
                "named_tuple": [
                    ExampleNamedTuple(1, 2),
                    ExampleNamedTuple(3, 4),
                ],
            }
        ),
        DataFrame,
    )
