"""Testing creation and manipulation of DataFrameSchema objects."""

# pylint: disable=too-many-lines,redefined-outer-name

import copy
from datetime import datetime, timedelta
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
import pandas as pd
import pytest

from pandera import (
    Category,
    Check,
    Column,
    DataFrameModel,
    DataFrameSchema,
    Field,
    Index,
    Int,
    MultiIndex,
    Parser,
    SeriesSchema,
    String,
    errors,
)
from pandera.api.pandas.array import ArraySchema
from pandera.dtypes import UniqueSettings
from pandera.engines.pandas_engine import Engine


def test_dataframe_schema() -> None:
    """Tests the Checking of a DataFrame that has a wide variety of types and
    conditions. Tests include: when the Schema works, when a column is dropped,
    and when a columns values change its type.
    """
    schema = DataFrameSchema(
        {
            "a": Column(
                int,
                Check(lambda x: x > 0, element_wise=True),
            ),
            "b": Column(
                float,
                Check(lambda x: 0 <= x <= 10, element_wise=True),
            ),
            "c": Column(
                str,
                Check(lambda x: set(x) == {"x", "y", "z"}),
            ),
            "d": Column(
                bool,
                Check(lambda x: x.mean() > 0.5),
            ),
            "e": Column(
                Category,
                Check(lambda x: set(x) == {"c1", "c2", "c3"}),
            ),
            "f": Column(
                object,
                Check(lambda x: x.isin([(1,), (2,), (3,)])),
            ),
            "g": Column(
                datetime,
                Check(
                    lambda x: x >= pd.Timestamp("2015-01-01"),
                    element_wise=True,
                ),
            ),
            "i": Column(
                timedelta,
                Check(
                    lambda x: x < pd.Timedelta(10, unit="D"), element_wise=True
                ),
            ),
        }
    )
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [1.1, 2.5, 9.9],
            "c": ["z", "y", "x"],
            "d": [True, True, False],
            "e": pd.Series(["c2", "c1", "c3"], dtype="category"),
            "f": [(3,), (2,), (1,)],
            "g": [
                pd.Timestamp("2015-02-01"),
                pd.Timestamp("2015-02-02"),
                pd.Timestamp("2015-02-03"),
            ],
            "i": [
                pd.Timedelta(1, unit="D"),
                pd.Timedelta(5, unit="D"),
                pd.Timedelta(9, unit="D"),
            ],
        }
    )
    assert isinstance(schema.validate(df), pd.DataFrame)

    # error case
    with pytest.raises(errors.SchemaError):
        schema.validate(df.drop(columns="a"))

    with pytest.raises(errors.SchemaError):
        schema.validate(df.assign(a=[-1, -2, -1]))

    # checks if 'a' is converted to float, while schema says int, will a schema
    # error be thrown
    with pytest.raises(errors.SchemaError):
        schema.validate(df.assign(a=[1.7, 2.3, 3.1]))


def test_dataframe_single_element_coerce() -> None:
    """Test that coercing a single element dataframe works correctly."""
    schema = DataFrameSchema({"x": Column(int, coerce=True)})
    assert isinstance(schema(pd.DataFrame({"x": [1]})), pd.DataFrame)
    with pytest.raises(
        errors.SchemaError,
        match="Error while coercing 'x' to type int64",
    ):
        schema(pd.DataFrame({"x": [None]}))


def test_dataframe_empty_coerce() -> None:
    """Test that coercing an empty element dataframe works correctly."""
    schema = DataFrameSchema({"x": Column(int, coerce=True)})
    assert isinstance(schema(pd.DataFrame({"x": []})), pd.DataFrame)


def test_dataframe_schema_equality() -> None:
    """Test DataframeSchema equality."""
    schema = DataFrameSchema({"a": Column(int)})
    assert schema == copy.copy(schema)
    assert schema != "schema"
    assert DataFrameSchema(coerce=True) != DataFrameSchema(coerce=False)
    assert schema != schema.update_column("a", dtype=float)
    assert schema != schema.update_column("a", checks=Check.eq(1))


def test_dataframe_schema_strict() -> None:
    """
    Checks if strict=True whether a schema error is raised because 'a' is
    not present in the dataframe.
    """
    schema = DataFrameSchema(
        {
            "a": Column(int, nullable=True),
            "b": Column(int, nullable=True),
        },
        strict=True,
    )
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})

    assert isinstance(schema.validate(df.loc[:, ["a", "b"]]), pd.DataFrame)
    with pytest.raises(errors.SchemaError):
        schema.validate(df)

    schema.strict = "filter"
    assert isinstance(schema.validate(df), pd.DataFrame)
    assert list(schema.validate(df).columns) == ["a", "b"]

    with pytest.raises(errors.SchemaInitError):
        DataFrameSchema(
            {
                "a": Column(int, nullable=True),
                "b": Column(int, nullable=True),
            },
            strict="foobar",  # type: ignore[arg-type]
        )

    with pytest.raises(errors.SchemaError):
        schema.validate(df.loc[:, ["a"]])
    with pytest.raises(errors.SchemaError):
        schema.validate(df.loc[:, ["a", "c"]])


def test_dataframe_schema_strict_regex() -> None:
    """Test that strict dataframe schema checks for regex matches."""
    schema = DataFrameSchema(
        {"foo_*": Column(int, regex=True)},
        strict=True,
    )
    df = pd.DataFrame({f"foo_{i}": range(10) for i in range(5)})

    assert isinstance(schema.validate(df), pd.DataFrame)

    # Raise a SchemaError if schema is strict and a regex pattern yields
    # no matches
    with pytest.raises(errors.SchemaError):
        schema.validate(
            pd.DataFrame({f"bar_{i}": range(10) for i in range(5)})
        )


def test_dataframe_schema_regex_error() -> None:
    """Test that regex schemas raise an error when no column is found."""
    schema = DataFrameSchema({"old_name.*": Column(regex=True)})
    df = pd.DataFrame(
        [
            {"error_name": 1},
        ]
    )

    try:
        schema.validate(df, lazy=True)
    except errors.SchemaErrors as exc:
        assert exc.failure_cases["check"].tolist() == [
            "no_regex_column_match('old_name.*')"
        ]


@pytest.mark.parametrize("with_columns", [True, False])
def test_dataframe_dtype_coerce(with_columns):
    """
    Test that pandas dtype specified at the dataframe level overrides
    column data types.
    """
    schema = DataFrameSchema(
        columns=(
            {f"column_{i}": Column(float) for i in range(5)}
            if with_columns
            else None
        ),
        dtype=int,
        coerce=True,
    )

    df = pd.DataFrame(
        {f"column_{i}": range(10) for i in range(5)}, dtype=float
    )
    int_alias = str(Engine.dtype(int))
    assert (schema(df).dtypes == int_alias).all()

    # test that dtype in schema.columns are preserved
    for col in schema.columns.values():
        assert col.dtype == Engine.dtype(float)

    # raises SchemeError if dataframe can't be coerced
    with pytest.raises(errors.SchemaErrors):
        schema.coerce_dtype(pd.DataFrame({"foo": list("abcdef")}))

    # raises SchemaErrors on lazy validation
    with pytest.raises(errors.SchemaErrors):
        schema(pd.DataFrame({"foo": list("abcdef")}), lazy=True)

    # test that original dataframe dtypes are preserved
    float_alias = str(Engine.dtype(float))
    assert (df.dtypes == float_alias).all()

    # test setting coerce as false causes SchemaError
    schema.coerce = False
    with pytest.raises(errors.SchemaError):
        schema(df)


def test_dataframe_coerce_regex() -> None:
    """Test dataframe pandas dtype coercion for regex columns"""
    schema = DataFrameSchema(
        columns={
            "column_": Column(
                float,
                checks=Check.gt(0),
                regex=True,
                required=False,
            )
        },
        dtype=int,
        coerce=True,
    )

    no_match_df = pd.DataFrame({"foo": [1, 2, 3]})
    match_valid_df = pd.DataFrame(
        {
            "column_1": [1, 2, 3],
            "column_2": ["1", "2", "3"],
        }
    )
    match_invalid_df = pd.DataFrame(
        {
            "column_1": [-1, 2, 3],
            "column_2": ["1", "-2", "3"],
        }
    )

    schema(no_match_df)
    schema(match_valid_df)

    with pytest.raises(errors.SchemaError):
        schema(match_invalid_df)

    # if the regex column is required, no matches should raise an error
    schema_required = schema.update_column("column_", required=True)
    with pytest.raises(
        errors.SchemaError, match="Column regex name='column_' did not match"
    ):
        schema_required(no_match_df)


def test_dataframe_reuse_column() -> None:
    """Test reusing columns in a dataframe schema."""
    unnamed = Column()
    named = Column(name="old_name")
    schema = DataFrameSchema(columns={"a": named, "b": unnamed})
    assert set(schema.columns) == {"a", "b"}
    assert named.name == "old_name"
    assert unnamed.name is None


@pytest.mark.parametrize(
    "columns,index",
    [
        (
            {
                "a": Column(int, required=False),
                "b": Column(int, required=False),
            },
            None,
        ),
        (
            None,
            MultiIndex(
                indexes=[Index(int, name="a"), Index(int, name="b")],
            ),
        ),
    ],
)
def test_ordered_dataframe(
    columns: Dict[str, Column], index: MultiIndex
) -> None:
    """Test that columns are ordered."""
    schema = DataFrameSchema(columns=columns, index=index, ordered=True)

    df = pd.DataFrame(
        data=[[1, 2, 3]],
        columns=["a", "a", "b"],
        index=pd.MultiIndex.from_arrays(
            [[1], [2], [3]], names=["a", "a", "b"]
        ),
    )
    assert isinstance(schema.validate(df), pd.DataFrame)

    # test optional column
    df = pd.DataFrame(
        data=[[1]],
        columns=["b"],
        index=pd.MultiIndex.from_arrays([[1], [2]], names=["a", "b"]),
    )
    assert isinstance(schema.validate(df), pd.DataFrame)

    # test out-of-order columns
    df = pd.DataFrame(
        data=[[1, 2]],
        columns=["b", "a"],
        index=pd.MultiIndex.from_arrays([[1], [2]], names=["b", "a"]),
    )
    with pytest.raises(errors.SchemaErrors, match=r"out-of-order"):
        schema.validate(df, lazy=True)

    # test out-of-order duplicates
    df = pd.DataFrame(
        data=[[1, 2, 3, 4]],
        columns=["a", "b", "c", "a"],
        index=pd.MultiIndex.from_arrays(
            [[1], [2], [3], [4]], names=["a", "b", "c", "a"]
        ),
    )
    with pytest.raises(errors.SchemaErrors, match=r"out-of-order"):
        schema.validate(df, lazy=True)


def test_duplicate_columns_dataframe():
    """Test that duplicate columns are detected."""
    col_labels = ["a", "a", "b"]
    frame = pd.DataFrame(data=[[1, 2, 3]], columns=col_labels)

    schema = DataFrameSchema(
        columns={i: Column(int) for i in col_labels},
        unique_column_names=True,
    )

    assert schema.unique_column_names

    with pytest.raises(
        errors.SchemaError,
        match="dataframe contains multiple columns with label",
    ):
        schema.validate(frame)

    schema.unique_column_names = False
    assert not schema.unique_column_names


def test_add_missing_columns_order():
    """Test that missing columns are added in the correct order."""
    col_labels = ["a", "b", "c"]

    # Missing column is first in schema
    frame_missing_first = pd.DataFrame(data=[[2, 3]], columns=["b", "c"])

    schema = DataFrameSchema(
        columns={i: Column(int, default=9) for i in col_labels},
        strict=True,
        add_missing_columns=False,
    )

    assert not schema.add_missing_columns

    with pytest.raises(
        errors.SchemaError,
        match="column 'a' not in dataframe",
    ):
        schema.validate(frame_missing_first)

    schema.add_missing_columns = True
    assert schema.add_missing_columns

    validated_frame_first = schema.validate(frame_missing_first)
    assert validated_frame_first.columns.tolist() == col_labels
    assert validated_frame_first["a"].eq(9).all()

    # Missing column is in middle of schema
    frame_missing_middle = pd.DataFrame(data=[[1, 3]], columns=["a", "c"])
    validated_frame_middle = schema.validate(frame_missing_middle)
    assert validated_frame_middle.columns.tolist() == col_labels
    assert validated_frame_middle["b"].eq(9).all()

    # Missing column is last in schema
    frame_missing_last = pd.DataFrame(data=[[1, 2]], columns=["a", "b"])
    validated_frame_last = schema.validate(frame_missing_last)
    assert validated_frame_last.columns.tolist() == col_labels
    assert validated_frame_last["c"].eq(9).all()

    # Front and last schema columns are missing
    frame_missing_multiple = pd.DataFrame(data=[[2]], columns=["b"])
    validated_frame_multiple = schema.validate(frame_missing_multiple)
    assert validated_frame_multiple.columns.tolist() == col_labels
    assert validated_frame_multiple[["a", "c"]].eq(9).all(axis=None)

    # Add missing column according to schema order but
    # ensure unknown column position is left intact
    frame_unknown_col = pd.DataFrame(
        data=[[1, 2, 3]], columns=["a", "missing", "c"]
    )
    with pytest.raises(
        errors.SchemaError,
        match="column 'missing' not in DataFrameSchema",
    ):
        schema.validate(frame_unknown_col)

    schema.strict = False
    assert not schema.strict

    validated_frame_unknown = schema.validate(frame_unknown_col)
    assert validated_frame_unknown.columns.tolist() == [
        "a",
        "b",
        "missing",
        "c",
    ]
    assert validated_frame_unknown["b"].eq(9).all()

    # Validate schema containing non-nullable column without a default value
    schema_no_default_not_nullable = DataFrameSchema(
        columns={i: Column(int, nullable=False) for i in ["a", "b", "c"]},
        strict=True,
        add_missing_columns=True,
    )
    with pytest.raises(
        errors.SchemaError,
        match="column 'a' in .* requires a default value when non-nullable add_missing_columns is enabled",
    ):
        schema_no_default_not_nullable.validate(frame_missing_first)

    # Validate missing column isn't added multiple times when multiple
    # trailing columns not in the schema exists in the dataframe
    # https://github.com/unionai-oss/pandera/issues/1370
    schema = DataFrameSchema(
        columns={
            "col_a": Column(str),
            "col_missing": Column(str, nullable=True),
        },
        add_missing_columns=True,
    )
    df = pd.DataFrame(
        {
            "col_a": ["a", "b", "c"],
            "col_b": ["d", "e", "f"],
            "col_c": ["g", "h", "i"],
        }
    )
    validated_frame_trailing = schema.validate(df)
    assert validated_frame_trailing.columns.equals(
        pd.Index(["col_a", "col_missing", "col_b", "col_c"])
    )


def test_add_missing_columns_dtype():
    """Test that missing columns are added with the correct dtype."""
    ref_df = pd.DataFrame(
        {
            "a": pd.Series([2, 5], dtype=np.int64),
            "b": pd.Series([9, 9], dtype=np.int64),
            "c": pd.Series([9, 9], dtype=np.int8),
            "d": pd.Series([np.nan, np.nan], dtype=np.float64),
            "e": pd.Series(
                [7, 7], dtype=pd.SparseDtype(dtype=np.int8, fill_value=5)
            ),
            "f": pd.Series([pd.NA, pd.NA], dtype=pd.Int32Dtype()),
        }
    )

    schema = DataFrameSchema(
        columns={
            "a": Column(np.int64),
            "b": Column(np.int64, default=9),
            "c": Column(np.int8, default=9),
            "d": Column(np.float64, default=np.nan, nullable=True),
            "e": Column(
                pd.SparseDtype(dtype=np.int8, fill_value=5), default=7
            ),
            "f": Column(pd.Int32Dtype(), default=pd.NA, nullable=True),
        },
        add_missing_columns=True,
        coerce=False,
    )
    test_df = schema.validate(pd.DataFrame(data={"a": [2, 5]}))
    pd.testing.assert_frame_equal(ref_df, test_df)


def test_series_schema() -> None:
    """Tests that a SeriesSchema Check behaves as expected for integers and
    strings. Tests error cases for types, duplicates, name errors, and issues
    around float and integer handling of nulls"""

    SeriesSchema("int").validate(pd.Series([1, 2, 3]))

    int_schema = SeriesSchema(
        int, Check(lambda x: 0 <= x <= 100, element_wise=True)
    )
    assert isinstance(
        int_schema.validate(pd.Series([0, 30, 50, 100])), pd.Series
    )

    def f(series):
        return series.isin(["foo", "bar", "baz"])

    str_schema = SeriesSchema(
        str,
        Check(f),
        nullable=True,
        coerce=True,
    )
    assert isinstance(
        str_schema.validate(pd.Series(["foo", "bar", "baz", None])), pd.Series
    )
    assert isinstance(
        str_schema.validate(pd.Series(["foo", "bar", "baz", np.nan])),
        pd.Series,
    )

    # error cases
    for data in [-1, 101, 50.1, "foo"]:
        with pytest.raises(errors.SchemaError):
            int_schema.validate(pd.Series([data]))

    for data in [-1, {"a": 1}, -1.0]:
        with pytest.raises(TypeError):
            int_schema.validate(TypeError)  # type: ignore

    non_duplicate_schema = SeriesSchema(Int, unique=True)
    with pytest.raises(errors.SchemaError):
        non_duplicate_schema.validate(pd.Series([0, 1, 2, 3, 4, 1]))

    # when series name doesn't match schema
    named_schema = SeriesSchema(int, name="my_series")
    with pytest.raises(errors.SchemaError, match=r"^Expected .+ to have name"):
        named_schema.validate(pd.Series(range(5), name="your_series"))

    # when series floats are declared to be integer
    with pytest.raises(errors.SchemaError):
        SeriesSchema(int, nullable=True).validate(
            pd.Series([1.1, 2.3, 5.5, np.nan])
        )

    # when series contains null values when schema is not nullable
    with pytest.raises(
        errors.SchemaError,
        match=r"^non-nullable series .+ contains null values",
    ):
        SeriesSchema(float, nullable=False).validate(
            pd.Series([1.1, 2.3, 5.5, np.nan])
        )

    # when series can't be coerced
    with pytest.raises(
        errors.SchemaError,
        match="Error while coercing",
    ):
        SeriesSchema(float, coerce=True).validate(pd.Series(list("abcdefg")))


def test_series_schema_checks() -> None:
    """Test SeriesSchema check property."""
    series_schema_no_checks = SeriesSchema()
    series_schema_one_check = SeriesSchema(checks=Check.eq(0))
    series_schema_multiple_checks = SeriesSchema(
        checks=[Check.gt(0), Check.lt(100)]
    )

    for schema in [
        series_schema_no_checks,
        series_schema_one_check,
        series_schema_multiple_checks,
    ]:
        assert isinstance(schema.checks, list)

    assert len(series_schema_no_checks.checks) == 0
    assert len(series_schema_one_check.checks) == 1
    assert len(series_schema_multiple_checks.checks) == 2


def test_series_schema_multiple_validators() -> None:
    """Tests how multiple Checks on a Series Schema are handled both
    successfully and when errors are expected."""
    schema = SeriesSchema(
        int,
        [
            Check(lambda x: 0 <= x <= 50, element_wise=True),
            Check(lambda s: (s == 21).any()),
        ],
    )
    validated_series = schema.validate(pd.Series([1, 5, 21, 50]))
    assert isinstance(validated_series, pd.Series)

    # raise error if any of the validators fails
    with pytest.raises(errors.SchemaError):
        schema.validate(pd.Series([1, 5, 20, 50]))


@pytest.mark.parametrize("coerce", [True, False])
def test_series_schema_with_index(coerce: bool) -> None:
    """Test SeriesSchema with Index and MultiIndex components."""
    schema_with_index = SeriesSchema(
        dtype=int,
        index=Index(int, coerce=coerce),
    )
    validated_series = schema_with_index(pd.Series([1, 2, 3], index=[1, 2, 3]))
    assert isinstance(validated_series, pd.Series)

    schema_with_multiindex = SeriesSchema(
        dtype=int,
        index=MultiIndex(
            [
                Index(int, coerce=coerce),
                Index(str, coerce=coerce),
            ]
        ),
    )
    multi_index = pd.MultiIndex.from_arrays(
        [[0, 1, 2], ["foo", "bar", "foo"]],
    )
    validated_series_multiindex = schema_with_multiindex(
        pd.Series([1, 2, 3], index=multi_index)
    )
    assert isinstance(validated_series_multiindex, pd.Series)
    assert (validated_series_multiindex.index == multi_index).all()


def test_series_schema_with_index_errors() -> None:
    """Test that SeriesSchema raises errors for invalid index."""
    schema_with_index = SeriesSchema(dtype=int, index=Index(int))
    data = pd.Series([1, 2, 3], index=[1.0, 2.0, 3.0])
    with pytest.raises(errors.SchemaError):
        schema_with_index(data)

    schema_with_index_check = SeriesSchema(
        dtype=int, index=Index(float, Check(lambda x: x == 1.0))
    )
    with pytest.raises(errors.SchemaError):
        schema_with_index_check(data)

    schema_with_index_coerce = SeriesSchema(
        dtype=int, index=Index(int, coerce=True)
    )
    expected = pd.Series([1, 2, 3], index=[1, 2, 3])
    schema_with_index_coerce(data)
    assert schema_with_index_coerce(data).equals(expected)


class SeriesGreaterCheck:
    # pylint: disable=too-few-public-methods
    """Class creating callable objects to check if series elements exceed a
    lower bound.
    """

    def __init__(self, lower_bound):
        self.lower_bound = lower_bound

    def __call__(self, series: pd.Series) -> pd.Series:
        """Check if the elements of s are > lower_bound.

        :returns Series with bool elements
        """
        return series > self.lower_bound


def series_greater_than_zero(series: pd.Series) -> pd.Series:
    """Return a bool series indicating whether the elements of s are > 0"""
    return series > 0


def series_greater_than_ten(series: pd.Series) -> pd.Series:
    """Return a bool series indicating whether the elements of s are > 10"""
    return series > 10


@pytest.mark.parametrize(
    "check_function, should_fail",
    [
        (lambda s: s > 0, False),
        (lambda s: s > 10, True),
        (series_greater_than_zero, False),
        (series_greater_than_ten, True),
        (SeriesGreaterCheck(lower_bound=0), False),
        (SeriesGreaterCheck(lower_bound=10), True),
    ],
)
def test_dataframe_schema_check_function_types(
    check_function: Callable[[pd.Series], pd.Series], should_fail: bool
) -> None:
    """Tests a DataFrameSchema against a variety of Check conditions."""
    schema = DataFrameSchema(
        {
            "a": Column(int, Check(check_function, element_wise=False)),
            "b": Column(float, Check(check_function, element_wise=False)),
        }
    )
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1.1, 2.5, 9.9]})
    if should_fail:
        with pytest.raises(errors.SchemaError):
            schema.validate(df)
    else:
        schema.validate(df)


def test_coerce_dtype_in_dataframe():
    """Tests coercions of datatypes, especially regarding nullable integers."""
    df = pd.DataFrame(
        {
            "column1": [10.0, 20.0, 30.0],
            "column2": ["2018-01-01", "2018-02-01", "2018-03-01"],
            "column3": [1, 2, None],
            "column4": [1.0, 1.0, np.nan],
        }
    )
    # specify `coerce` at the Column level
    schema1 = DataFrameSchema(
        {
            "column1": Column(int, Check(lambda x: x > 0), coerce=True),
            "column2": Column(datetime, coerce=True),
        }
    )
    # specify `coerce` at the DataFrameSchema level
    schema2 = DataFrameSchema(
        {
            "column1": Column(int, Check(lambda x: x > 0)),
            "column2": Column(datetime),
        },
        coerce=True,
    )

    for schema in [schema1, schema2]:
        validated = schema.validate(df)
        coerced = schema.coerce_dtype(df)
        assert Engine.dtype(validated.column1.dtype) == Engine.dtype(int)
        assert Engine.dtype(coerced.column1.dtype) == Engine.dtype(int)
        assert Engine.dtype(validated.column2.dtype) == Engine.dtype(datetime)
        assert Engine.dtype(coerced.column2.dtype) == Engine.dtype(datetime)

    # make sure that correct error is raised when null values are present
    # in a float column that's coerced to an int
    schema = DataFrameSchema({"column4": Column(int, coerce=True)})
    with pytest.raises(
        errors.SchemaError,
        match=r"^Error while coercing .+ to type u{0,1}int[0-9]{1,2}: "
        r"Could not coerce .+ data_container into type",
    ):
        schema.validate(df)


def test_no_dtype_dataframe():
    """Test how nullability is handled in DataFrameSchemas where no type is
    specified."""
    schema = DataFrameSchema({"col": Column(nullable=False)})
    validated_df = schema.validate(pd.DataFrame({"col": [-123.1, -76.3, 1.0]}))
    assert isinstance(validated_df, pd.DataFrame)

    schema = DataFrameSchema({"col": Column(nullable=True)})
    validated_df = schema.validate(pd.DataFrame({"col": [-123.1, None, 1.0]}))
    assert isinstance(validated_df, pd.DataFrame)

    with pytest.raises(errors.SchemaError):
        schema = DataFrameSchema({"col": Column(nullable=False)})
        schema.validate(pd.DataFrame({"col": [-123.1, None, 1.0]}))


def test_no_dtype_series() -> None:
    """Test how nullability is handled in SeriesSchemas where no type is
    specified."""
    schema = SeriesSchema(nullable=False)
    validated_series = schema.validate(pd.Series([0, 1, 2, 3, 4, 1]))
    assert isinstance(validated_series, pd.Series)

    schema = SeriesSchema(nullable=True)
    validated_series = schema.validate(pd.Series([0, 1, 2, None, 4, 1]))
    assert isinstance(validated_series, pd.Series)

    with pytest.raises(errors.SchemaError):
        schema = SeriesSchema(nullable=False)
        schema.validate(pd.Series([0, 1, 2, None, 4, 1]))


def test_coerce_without_dtype() -> None:
    """Test that an error is thrown when a dtype isn't specified and coerce
    is True."""
    df = pd.DataFrame({"col": [1, 2, 3]})
    for schema in [
        DataFrameSchema({"col": Column(coerce=True)}),
        DataFrameSchema({"col": Column()}, coerce=True),
    ]:
        assert isinstance(schema(df), pd.DataFrame)


def test_required() -> None:
    """
    Tests how a required Column is handled when it's not included, included
    and then not specified and a second column which is implicitly required
    isn't available.
    """
    schema = DataFrameSchema(
        {"col1": Column(int, required=False), "col2": Column(str)}
    )

    df_ok_1 = pd.DataFrame({"col2": ["hello", "world"]})

    df = schema.validate(df_ok_1)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1
    assert set(df.columns) == {"col2"}

    df_ok_2 = pd.DataFrame({"col1": [1, 2], "col2": ["hello", "world"]})

    df = schema.validate(df_ok_2)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert set(df.columns) == {"col1", "col2"}

    df_not_ok = pd.DataFrame({"col1": [1, 2]})

    with pytest.raises(errors.SchemaError):
        schema.validate(df_not_ok)


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame({"col": [1, 2, 3]}),
        pd.DataFrame({"col": ["1", "2", "3"]}),
        pd.DataFrame(),
    ],
)
@pytest.mark.parametrize("required", [True, False])
def test_coerce_not_required(data: pd.DataFrame, required: bool) -> None:
    """Test that not required columns are not coerced."""
    schema = DataFrameSchema(
        {"col": Column(int, required=required)}, coerce=True
    )
    if required and data.empty:
        with pytest.raises(errors.SchemaError):
            schema(data)
        return
    schema(data)


def test_head_dataframe_schema() -> None:
    """Test that schema can validate head of dataframe, returns entire
    dataframe."""

    df = pd.DataFrame(
        {"col1": list(range(0, 100)) + list(range(-1, -1001, -1))}
    )

    schema = DataFrameSchema(
        columns={"col1": Column(int, Check(lambda s: s >= 0))}
    )

    # Validating with head of 100 should pass
    assert schema.validate(df, head=100).equals(df)
    with pytest.raises(errors.SchemaError):
        schema.validate(df)


def test_tail_dataframe_schema() -> None:
    """Checks that validating the tail of a dataframe validates correctly."""
    df = pd.DataFrame(
        {"col1": list(range(0, 100)) + list(range(-1, -1001, -1))}
    )

    schema = DataFrameSchema(
        columns={"col1": Column(int, Check(lambda s: s < 0))}
    )

    # Validating with tail of 1000 should pass
    assert schema.validate(df, tail=1000).equals(df)
    with pytest.raises(errors.SchemaError):
        schema.validate(df)


def test_sample_dataframe_schema() -> None:
    """Test the sample argument of schema.validate."""
    df = pd.DataFrame({"col1": range(1, 1001)})

    # assert all values -1
    schema = DataFrameSchema(
        columns={"col1": Column(int, Check(lambda s: s == -1))}
    )

    for seed in [11, 123456, 9000, 654]:
        sample_index = df.sample(100, random_state=seed).index
        df.loc[sample_index] = -1  # type: ignore
        assert schema.validate(df, sample=100, random_state=seed).equals(df)


def test_dataframe_schema_str_repr() -> None:
    """Test the __str__ and __repr__ methods which are used for cleanly
    printing/logging of a DataFrameSchema."""
    schema = DataFrameSchema(
        columns={
            "col1": Column(int),
            "col2": Column(str),
            "col3": Column(datetime),
        },
        index=Index(int, name="my_index"),
    )

    for x in [schema.__str__(), schema.__repr__()]:
        assert isinstance(x, str)
        assert schema.__class__.__name__ in x
        for name in ["col1", "col2", "col3", "my_index"]:
            assert name in x


def test_dataframe_schema_dtype_property() -> None:
    """Test that schema.dtype returns the matching Column types."""
    schema = DataFrameSchema(
        columns={
            "col1": Column(int),
            "col2": Column(str),
            "col3": Column(datetime),
            "col4": Column("uint16"),
        }
    )
    assert schema.dtypes == {
        "col1": Engine.dtype("int64"),
        "col2": Engine.dtype("str"),
        "col3": Engine.dtype("datetime64[ns]"),
        "col4": Engine.dtype("uint16"),
    }


def test_schema_equality_operators():
    """Test the usage of == for DataFrameSchema, SeriesSchema and
    SeriesSchemaBase."""
    df_schema = DataFrameSchema(
        {
            "col1": Column(int, Check(lambda s: s >= 0)),
            "col2": Column(str, Check(lambda s: s >= 2)),
        },
        strict=True,
    )
    df_schema_columns_in_different_order = DataFrameSchema(
        {
            "col2": Column(str, Check(lambda s: s >= 2)),
            "col1": Column(int, Check(lambda s: s >= 0)),
        },
        strict=True,
    )
    series_schema = SeriesSchema(
        str,
        checks=[Check(lambda s: s.str.startswith("foo"))],
        parsers=Parser(lambda s: s.str.upper()),
        nullable=False,
        unique=False,
        name="my_series",
    )
    series_schema_base = ArraySchema(
        str,
        checks=[Check(lambda s: s.str.startswith("foo"))],
        parsers=[Parser(lambda s: s.str.upper())],
        nullable=False,
        unique=False,
        name="my_series",
    )
    not_equal_schema = DataFrameSchema({"col1": Column(str)}, strict=False)

    assert df_schema == copy.deepcopy(df_schema)
    assert df_schema != not_equal_schema
    assert df_schema == df_schema_columns_in_different_order
    assert series_schema == copy.deepcopy(series_schema)
    assert series_schema != not_equal_schema
    assert series_schema_base == copy.deepcopy(series_schema_base)
    assert series_schema_base != not_equal_schema


def test_add_and_remove_columns() -> None:
    """Check that adding and removing columns works as expected and doesn't
    modify the original underlying DataFrameSchema."""
    schema1 = DataFrameSchema(
        {
            "col1": Column(int, Check(lambda s: s >= 0)),
        },
        strict=True,
    )

    schema1_exact_copy = copy.deepcopy(schema1)

    # test that add_columns doesn't modify schema1 after add_columns:
    schema2 = schema1.add_columns(
        {
            "col2": Column(str, Check(lambda x: x <= 0)),
            "col3": Column(object, Check(lambda x: x == 0)),
        }
    )

    schema2_exact_copy = copy.deepcopy(schema2)

    assert schema1 == schema1_exact_copy

    # test that add_columns changed schema1 into schema2:
    expected_schema_2 = DataFrameSchema(
        {
            "col1": Column(int, Check(lambda s: s >= 0)),
            "col2": Column(str, Check(lambda x: x <= 0)),
            "col3": Column(object, Check(lambda x: x == 0)),
        },
        strict=True,
    )

    assert schema2 == expected_schema_2

    # test that remove_columns doesn't modify schema2:
    schema3 = schema2.remove_columns(["col2"])

    assert schema2 == schema2_exact_copy

    # test that remove_columns has removed the changes as expected:
    expected_schema_3 = DataFrameSchema(
        {
            "col1": Column(int, Check(lambda s: s >= 0)),
            "col3": Column(object, Check(lambda x: x == 0)),
        },
        strict=True,
    )

    assert schema3 == expected_schema_3

    # test that remove_columns can remove two columns:
    schema4 = schema2.remove_columns(["col2", "col3"])

    expected_schema_4 = DataFrameSchema(
        {"col1": Column(int, Check(lambda s: s >= 0))}, strict=True
    )

    assert schema4 == expected_schema_4 == schema1

    # test raising error if column name is not in the schema
    with pytest.raises(errors.SchemaInitError):
        schema2.remove_columns(["foo", "bar"])


def test_schema_get_dtypes():
    """Test that schema dtype and get_dtypes methods handle regex columns."""
    schema = DataFrameSchema(
        {
            "col1": Column(int),
            "var*": Column(float, regex=True),
        }
    )

    data = pd.DataFrame(
        {
            "col1": [1, 2, 3],
            "var1": [1.0, 1.1, 1.2],
            "var2": [1.0, 1.1, 1.2],
            "var3": [1.0, 1.1, 1.2],
        }
    )

    with pytest.warns(UserWarning) as record:
        assert schema.dtypes == {"col1": Engine.dtype(int)}
    assert len(record) == 1
    assert (
        record[0]  # type: ignore[union-attr]
        .message.args[0]
        .startswith("Schema has columns specified as regex column names:")
    )

    assert schema.get_dtypes(data) == {
        "col1": Engine.dtype(int),
        "var1": Engine.dtype(float),
        "var2": Engine.dtype(float),
        "var3": Engine.dtype(float),
    }


def _boolean_update_column_case(
    bool_kwarg: str,
) -> List[Any]:
    def _assert_bool_case(old_schema, new_schema):
        assert not getattr(old_schema.columns["col"], bool_kwarg)
        assert getattr(new_schema.columns["col"], bool_kwarg)

    return [
        Column(int, **{bool_kwarg: False}),  # type: ignore
        "col",
        {bool_kwarg: True},
        _assert_bool_case,
    ]


@pytest.mark.parametrize(
    "column, column_to_update, update, assertion_fn",
    [
        [
            Column(int),
            "col",
            {"dtype": str},
            lambda old, new: [
                old.columns["col"].dtype is int,
                new.columns["col"].dtype is str,
            ],
        ],
        *[
            _boolean_update_column_case(bool_kwarg)
            for bool_kwarg in [
                "nullable",
                "coerce",
                "required",
                "regex",
                "unique",
            ]
        ],
        [
            Column(int, checks=Check.greater_than(0)),
            "col",
            {"checks": Check.less_than(10)},
            lambda old, new: [
                old.columns["col"].checks == [Check.greater_than(0)],
                new.columns["col"].checks == [Check.less_than(10)],
            ],
        ],
        # error cases
        [Column(int), "col", {"name": "renamed_col"}, ValueError],
        [Column(int), "foobar", {}, ValueError],
    ],
)
def test_dataframe_schema_update_column(
    column: Column,
    column_to_update: str,
    update: Dict[str, Any],
    assertion_fn: Callable[[DataFrameSchema, DataFrameSchema], None],
) -> None:
    """Test that DataFrameSchema columns create updated copies."""
    schema = DataFrameSchema({"col": column})
    if assertion_fn is ValueError:
        with pytest.raises(ValueError):
            schema.update_column(column_to_update, **update)
        return

    new_schema = schema.update_column(column_to_update, **update)
    assertion_fn(schema, new_schema)


def test_rename_columns() -> None:
    """Check that DataFrameSchema.rename_columns() method does its job"""

    rename_dict = {"col1": "col1_new_name", "col2": "col2_new_name"}
    schema_original = DataFrameSchema(
        columns={"col1": Column(int), "col2": Column(float)}
    )

    schema_renamed = schema_original.rename_columns(rename_dict)

    # Check if new column names are indeed present in the new schema
    assert all(
        col_name in rename_dict.values() for col_name in schema_renamed.columns
    )
    # Check if original schema didn't change in the process
    assert all(col_name in schema_original.columns for col_name in rename_dict)

    with pytest.raises(errors.SchemaInitError):
        schema_original.rename_columns({"foo": "bar"})

    # Test raising error if new column name is already in schema
    for rename_dict in [{"col1": "col2"}, {"col2": "col1"}]:
        with pytest.raises(errors.SchemaInitError):
            schema_original.rename_columns(rename_dict)

    # Test doesn't raise error if column maps to itself
    rename_dict = {"col1": "col1", "col2": "col2_new_name"}
    schema_original.rename_columns(rename_dict)


@pytest.mark.parametrize(
    "select_columns, schema, expected_order",
    [
        (
            ["col1", "col2"],
            DataFrameSchema(
                columns={
                    "col1": Column(int),
                    "col2": Column(int),
                    "col3": Column(int),
                }
            ),
            ["col1", "col2"],
        ),
        (
            ["col3", "col1"],
            DataFrameSchema(
                columns={
                    "col1": Column(int),
                    "col2": Column(int),
                    "col3": Column(int),
                }
            ),
            ["col3", "col1"],
        ),
        (
            [("col1", "col1b"), ("col2", "col2b")],
            DataFrameSchema(
                columns={
                    ("col1", "col1a"): Column(int),
                    ("col1", "col1b"): Column(int),
                    ("col2", "col2a"): Column(int),
                    ("col2", "col2b"): Column(int),
                }
            ),
            [("col1", "col1b"), ("col2", "col2b")],
        ),
        (
            [("col2", "col2b"), ("col1", "col1b")],
            DataFrameSchema(
                columns={
                    ("col1", "col1a"): Column(int),
                    ("col1", "col1b"): Column(int),
                    ("col2", "col2a"): Column(int),
                    ("col2", "col2b"): Column(int),
                }
            ),
            [("col2", "col2b"), ("col1", "col1b")],
        ),
    ],
)
def test_select_columns(
    select_columns: List[Union[str, Tuple[str, str]]],
    schema: DataFrameSchema,
    expected_order: List[Union[str, Tuple[str, str]]],
) -> None:
    """Check that select_columns method correctly creates new subset schema."""
    original_columns = list(schema.columns)
    schema_selected = schema.select_columns(select_columns)

    assert all(x in select_columns for x in schema_selected.columns)
    assert all(x in original_columns for x in schema.columns)
    assert list(schema_selected.columns) == expected_order

    with pytest.raises(errors.SchemaInitError):
        schema.select_columns(["foo", "bar"])


def test_lazy_dataframe_validation_error() -> None:
    """Test exceptions on lazy dataframe validation."""
    schema = DataFrameSchema(
        columns={
            "int_col": Column(int, Check.greater_than(5)),
            "int_col2": Column(int),
            "float_col": Column(float, Check.less_than(0)),
            "str_col": Column(str, Check.isin(["foo", "bar"])),
            "not_in_dataframe": Column(int),
        },
        checks=Check(
            lambda df: df != 1, error="dataframe_not_equal_1", ignore_na=False
        ),
        index=Index(str, name="str_index"),
        strict=True,
    )

    dataframe = pd.DataFrame(
        data={
            "int_col": [1, 2, 6],
            "int_col2": ["a", "b", "c"],
            "float_col": [1.0, -2.0, 3.0],
            "str_col": ["foo", "b", "c"],
            "unknown_col": [None, None, None],
        },
        index=pd.Index(["index0", "index1", "index2"], name="str_index"),
    )

    expectation = {
        # schema object context -> check failure cases
        "DataFrameSchema": {
            # check name -> failure cases
            "column_in_schema": ["unknown_col"],
            "dataframe_not_equal_1": [{"int_col": 1.0, "float_col": 1.0}],
            "column_in_dataframe": ["not_in_dataframe"],
        },
        "Column": {
            "greater_than(5)": [1, 2],
            "dtype('int64')": ["object"],
            "less_than(0)": [1, 3],
        },
    }

    with pytest.raises(errors.SchemaErrors) as e:
        schema.validate(dataframe, lazy=True)

    assert len(e.value.message["SCHEMA"]) == 3

    try:
        schema.validate(dataframe, lazy=True)
    except errors.SchemaErrors as err:
        # data in the caught exception should be equal to the dataframe
        # passed into validate
        assert err.data.equals(dataframe)  # type: ignore

        # make sure all expected check errors are in schema errors
        for schema_context, check_failure_cases in expectation.items():
            err_df = err.failure_cases.loc[
                err.failure_cases.schema_context == schema_context
            ]
            for check, failure_cases in check_failure_cases.items():
                assert check in err_df.check.values
                assert (
                    err_df.loc[err_df.check == check]
                    .failure_case.isin(failure_cases)
                    .all()
                )


def test_lazy_validation_multiple_checks() -> None:
    """Lazy validation with multiple checks should report all failures."""
    schema = DataFrameSchema(
        {
            "col1": Column(
                Int,
                checks=[
                    Check.in_range(1, 4, include_min=True, include_max=True),
                    Check(lambda s: s % 2 == 0, name="is_even"),
                ],
                coerce=True,
                nullable=False,
            ),
            "col2": Column(Int, Check.gt(3), coerce=True, nullable=False),
        }
    )

    data = pd.DataFrame(
        {"col1": [0, 1, 2, 3, 4], "col2": [np.nan, 53, 23, np.nan, 2]}
    )

    expectation = {
        "col1": {
            "in_range(1, 4)": [0],
            "is_even": [1, 3],
        },
        "col2": {
            "coerce_dtype('int64')": [np.nan, np.nan],
        },
    }

    try:
        schema.validate(data, lazy=True)
    except errors.SchemaErrors as err:
        for column_name, check_failure_cases in expectation.items():
            err_df = err.failure_cases.loc[
                err.failure_cases.column == column_name
            ]
            for check, failure_cases in check_failure_cases.items():  # type: ignore
                assert check in err_df.check.values
                failed = list(err_df.loc[err_df.check == check].failure_case)
                if pd.isna(failure_cases).all():
                    assert pd.isna(failed).all()
                else:
                    assert failed == failure_cases


def test_lazy_dataframe_validation_nullable() -> None:
    """
    Test that non-nullable column failure cases are correctly processed during
    lazy validation.
    """
    schema = DataFrameSchema(
        columns={
            "int_column": Column(int, nullable=False),
            "float_column": Column(float, nullable=False),
            "str_column": Column(str, nullable=False),
        },
        strict=True,
    )

    df = pd.DataFrame(
        {
            "int_column": [1, None, 3],
            "float_column": [0.1, 1.2, None],
            "str_column": [None, "foo", "bar"],
        }
    )

    try:
        schema.validate(df, lazy=True)
    except errors.SchemaErrors as err:
        # report not_nullable checks
        assert (
            err.failure_cases.query("check == 'not_nullable'")
            .failure_case.isna()
            .all()
        )
        # report invalid type in int_column
        assert (
            err.failure_cases.query(
                "check == \"pandas_dtype('int64')\""
            ).failure_case
            == "float64"
        ).all()

        for col, index in [
            ("int_column", 1),
            ("float_column", 2),
            ("str_column", 0),
        ]:
            # pylint: disable=cell-var-from-loop
            assert (
                err.failure_cases.loc[
                    lambda df: df.column == col, "index"
                ].iloc[0]
                == index
            )


def test_lazy_dataframe_validation_with_checks() -> None:
    """Test that all failure cases are reported for schemas with checks."""
    schema = DataFrameSchema(
        columns={
            "analysis_path": Column(String),
            "run_id": Column(String),
            "sample_type": Column(String, Check.isin(["DNA", "RNA"])),
            "sample_valid": Column(String, Check.isin(["Yes", "No"])),
        },
        strict=False,
        coerce=True,
    )

    df = pd.DataFrame.from_dict(
        {
            "analysis_path": ["/", "/", "/", "/", "/"],
            "run_id": ["1", "2", "3", "4", "5"],
            "sample_type": ["DNA", "RNA", "DNA", "RNA", "RNA"],
            "sample_valid": ["Yes", "YES", "YES", "NO", "NO"],
        }
    )

    try:
        schema(df, lazy=True)
    except errors.SchemaErrors as err:
        failure_case = err.failure_cases.failure_case.tolist()
        assert failure_case == ["YES", "YES", "NO", "NO"]


def test_lazy_dataframe_validation_nullable_with_checks() -> None:
    """
    Test that checks in non-nullable column failure cases are correctly
    processed during lazy validation.
    """
    schema = DataFrameSchema(
        {
            "id": Column(
                String,
                checks=Check.str_matches(r"^ID[\d]{3}$"),
                name="id",
                required=True,
                unique=True,
            )
        }
    )
    df = pd.DataFrame({"id": ["ID001", None, "XXX"]})
    try:
        schema(df, lazy=True)
    except errors.SchemaErrors as err:
        expected_failure_cases = pd.DataFrame.from_dict(
            {
                0: {
                    "schema_context": "Column",
                    "column": "id",
                    "check": "not_nullable",
                    "check_number": None,
                    "failure_case": None,
                    "index": 1,
                },
                1: {
                    "schema_context": "Column",
                    "column": "id",
                    "check": r"str_matches('^ID[\d]{3}$')",
                    "check_number": 0,
                    "failure_case": "XXX",
                    "index": 2,
                },
            },
            orient="index",
        ).astype({"check_number": object})
        pd.testing.assert_frame_equal(
            err.failure_cases, expected_failure_cases
        )


@pytest.mark.parametrize(
    "schema_cls, data",
    [
        [DataFrameSchema, pd.DataFrame({"column": [1]})],
        [SeriesSchema, pd.Series([1, 2, 3])],
        [partial(Column, name="column"), pd.DataFrame({"column": [1]})],
        [
            partial(Index, name="index"),
            pd.DataFrame(index=pd.Index([1, 2, 3], name="index")),
        ],
    ],
)
def test_lazy_dataframe_scalar_false_check(
    schema_cls: Type[Union[DataFrameSchema, SeriesSchema, Column, Index]],
    data: Union[pd.DataFrame, pd.Series, pd.Index],
) -> None:
    """Lazy validation handles checks returning scalar False values."""
    # define a check that always returns a scalare False value
    check = Check(
        check_fn=lambda _: False, element_wise=False, error="failing check"
    )
    schema = schema_cls(checks=check)
    with pytest.raises(errors.SchemaErrors):
        schema(data, lazy=True)  # type: ignore


def test_lazy_dataframe_unique() -> None:
    """Tests the lazy evaluation of the unique keyword"""
    data = pd.DataFrame.from_dict(
        {"A": [1, 2, 3, 4], "B": [1, 2, 3, 1], "C": [1, 2, 3, 1]}
    )
    schema = DataFrameSchema(
        columns={"A": Column(Int), "B": Column(Int), "C": Column(Int)},
        strict=False,
        coerce=True,
        unique=None,
    )
    assert isinstance(schema.validate(data, lazy=False), pd.DataFrame)
    schema.unique = ["A", "B"]
    assert isinstance(schema.validate(data, lazy=False), pd.DataFrame)
    schema.unique = ["B", "C"]
    try:
        schema.validate(data, lazy=True)
    except errors.SchemaErrors as err:
        errors_df = pd.DataFrame(err.failure_cases)
        assert list(errors_df["column"].values) == ["B", "B", "C", "C"]
        assert list(errors_df["index"].values) == [0, 3, 0, 3]


@pytest.mark.parametrize(
    "schema, data, expectation",
    [
        # case: series name doesn't match schema name
        [
            SeriesSchema(name="foobar"),
            pd.Series(range(3)),
            {
                "data": pd.Series(range(3)),
                "schema_errors": {
                    "SeriesSchema": {"field_name('foobar')": [None]},
                },
            },
        ],
        # case: series type doesn't match schema type
        [
            SeriesSchema(int),
            pd.Series([0.1]),
            {
                "data": pd.Series([0.1]),
                "schema_errors": {
                    "SeriesSchema": {"dtype('int64')": ["float64"]},
                },
            },
        ],
        # case: series index doesn't satisfy schema index
        [
            SeriesSchema(index=Index(int)),
            pd.Series([1, 2, 3], index=list("abc")),
            {
                "data": pd.Series([1, 2, 3], index=list("abc")),
                "schema_errors": {
                    "Index": {"dtype('int64')": ["object"]},
                },
            },
        ],
        # case: SeriesSchema data-type coercion error
        [
            SeriesSchema(float, coerce=True),
            pd.Series(["1", "foo", "bar"]),
            {
                "data": pd.Series(["1", "foo", "bar"]),
                "schema_errors": {
                    "SeriesSchema": {
                        "dtype('float64')": ["object"],
                        "coerce_dtype('float64')": ["foo", "bar"],
                    },
                },
            },
        ],
        # case: series index coercion error
        [
            SeriesSchema(index=Index(int, coerce=True)),
            pd.Series([1, 2, 3], index=list("abc")),
            {
                "data": pd.Series([1, 2, 3], index=list("abc")),
                "schema_errors": {
                    "Index": {"coerce_dtype('int64')": ["a", "b", "c"]},
                },
            },
        ],
        # case: series type and check doesn't satisfy schema
        [
            SeriesSchema(int, checks=Check.greater_than(0)),
            pd.Series(["a", "b", "c"]),
            {
                "data": pd.Series(["a", "b", "c"]),
                "schema_errors": {
                    # schema object context -> check failure cases
                    "SeriesSchema": {
                        # check name -> failure cases
                        "greater_than(0)": [
                            "TypeError(\"'>' not supported between instances "
                            "of 'str' and 'int'\")",
                            # TypeError raised in python=3.5
                            'TypeError("unorderable types: str() > int()")',
                        ],
                        "dtype('int64')": ["object"],
                    },
                },
            },
        ],
        # case: multiple series checks don't satisfy schema
        [
            Column(
                int,
                checks=[Check.greater_than(1), Check.less_than(3)],
                name="column",
            ),
            pd.DataFrame({"column": [1, 2, 3]}),
            {
                "data": pd.DataFrame({"column": [1, 2, 3]}),
                "schema_errors": {
                    "Column": {"greater_than(1)": [1], "less_than(3)": [3]},
                },
            },
        ],
        [
            Index(str, checks=Check.isin(["a", "b", "c"])),
            pd.DataFrame({"col": [1, 2, 3]}, index=["a", "b", "d"]),
            {
                "data": pd.DataFrame(
                    {"col": [1, 2, 3]}, index=["a", "b", "d"]
                ),
                "schema_errors": {
                    "Index": {"isin(['a', 'b', 'c'])": ["d"]},
                },
            },
        ],
        [
            MultiIndex(
                indexes=[
                    Index(int, checks=Check.greater_than(0), name="index0"),
                    Index(int, checks=Check.less_than(0), name="index1"),
                ]
            ),
            pd.DataFrame(
                {"column": [1, 2, 3]},
                index=pd.MultiIndex.from_arrays(
                    [[0, 1, 2], [-2, -1, 0]],
                    names=["index0", "index1"],
                ),
            ),
            {
                "data": pd.DataFrame(
                    {"column": [1, 2, 3]},
                    index=pd.MultiIndex.from_arrays(
                        [[0, 1, 2], [-2, -1, 0]],
                        names=["index0", "index1"],
                    ),
                ),
                "schema_errors": {
                    "MultiIndex": {
                        "greater_than(0)": [0],
                        "less_than(0)": [0],
                    },
                },
            },
        ],
    ],
)
def test_lazy_series_validation_error(schema, data, expectation) -> None:
    """Test exceptions on lazy series validation."""
    try:
        schema.validate(data, lazy=True)
    except errors.SchemaErrors as err:
        # data in the caught exception should be equal to the data
        # passed into validate
        assert err.data.equals(expectation["data"])

        # make sure all expected check errors are in schema errors
        for schema_context, check_failure_cases in expectation[
            "schema_errors"
        ].items():
            assert schema_context in err.failure_cases.schema_context.values
            err_df = err.failure_cases.loc[
                err.failure_cases.schema_context == schema_context
            ]
            for check, failure_cases in check_failure_cases.items():
                assert check in err_df.check.values
                assert (
                    err_df.loc[err_df.check == check]
                    .failure_case.isin(failure_cases)
                    .all()
                )


def test_capture_check_errors() -> None:
    """Test that exceptions raised within checks can be captured."""

    def fail_with_msg(data):
        raise KeyError("fail")

    def fail_without_msg(data):
        raise ValueError()

    schema = SeriesSchema(
        checks=[Check(fail_with_msg), Check(fail_without_msg)]
    )
    with pytest.raises(errors.SchemaError):
        schema.validate(pd.Series([1, 2, 3]))

    try:
        schema.validate(pd.Series([1, 2, 3]), lazy=True)
    except errors.SchemaErrors as err:
        cases = err.failure_cases
        failure_with_msg = cases.loc[
            cases.check == "fail_with_msg", "failure_case"
        ].iloc[0]
        assert failure_with_msg == 'KeyError("fail")'

        failure_without_msg = cases.loc[
            cases.check == "fail_without_msg", "failure_case"
        ].iloc[0]
        assert failure_without_msg == "ValueError()"


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "from_dtype,to_dtype",
    [
        [float, int],
        [int, float],
        [object, int],
        [object, float],
        [int, object],
        [float, object],
    ],
)
def test_schema_coerce_inplace_validation(
    inplace: bool, from_dtype: Type, to_dtype: Type
) -> None:
    """Test coercion logic for validation when inplace is True and False"""
    from_dtype = (
        from_dtype if from_dtype is not int else str(Engine.dtype(from_dtype))
    )
    to_dtype = to_dtype if to_dtype is not int else str(Engine.dtype(to_dtype))
    df = pd.DataFrame({"column": pd.Series([1, 2, 6], dtype=from_dtype)})  # type: ignore[call-overload]
    schema = DataFrameSchema({"column": Column(to_dtype, coerce=True)})  # type: ignore
    validated_df = schema.validate(df, inplace=inplace)

    assert validated_df["column"].dtype == to_dtype
    if inplace:
        # inplace mutates original dataframe
        assert df["column"].dtype == to_dtype
    else:
        # not inplace preserves original dataframe type
        assert df["column"].dtype == from_dtype


@pytest.mark.parametrize(
    "unique,answers",
    [
        # unique is True -- default is to report all unique violations except the first
        ("exclude_first", [4, 5, 6, 7]),
        ("all", [0, 1, 2, 4, 5, 6, 7]),
        ("exclude_first", [4, 5, 6, 7]),
        ("exclude_last", [0, 1, 2, 4]),
    ],
)
def test_different_unique_settings(unique: UniqueSettings, answers: List[int]):
    """Test that different unique settings work as expected"""
    df: pd.DataFrame = pd.DataFrame({"a": [1, 2, 3, 4, 1, 1, 2, 3]})
    schemas = [
        DataFrameSchema(
            {"a": Column(int)}, unique="a", report_duplicates=unique
        ),
        DataFrameSchema(
            {"a": Column(int, unique=True, report_duplicates=unique)}
        ),
    ]

    for schema in schemas:
        with pytest.raises(errors.SchemaError) as err:
            schema.validate(df)

        assert err.value.failure_cases["index"].to_list() == answers

    series_schema = SeriesSchema(int, unique=True, report_duplicates=unique)

    with pytest.raises(errors.SchemaError) as err:
        srs: pd.Series = df["a"]  # type: ignore[assignment]
        series_schema.validate(srs)

    assert err.value.failure_cases["index"].to_list() == answers


@pytest.mark.parametrize(
    "report_duplicates", ["all", "exclude_first", "exclude_last", "invalid"]
)
def test_valid_unique_settings(report_duplicates):
    """Test that valid unique settings work and invalid ones will raise a ValueError"""
    schema = DataFrameSchema(
        {"a": Column(String)}, unique="a", report_duplicates=report_duplicates
    )
    df = pd.DataFrame({"a": ["A", "BC", "C", "C", "BC"]})

    # If we're given an invalid value for report_duplicates, then it should raise a ValueError
    if report_duplicates == "invalid":
        with pytest.raises(ValueError):
            schema.validate(df)
    else:
        with pytest.raises(errors.SchemaError) as err:
            schema.validate(df)

        # There are unique errors--assert that pandera reports them properly
        # Actual content of the unique errors is tested in test_different_unique_settings
        assert not err.value.failure_cases.empty


@pytest.fixture
def schema_simple() -> DataFrameSchema:
    """Simple schema fixture."""
    schema = DataFrameSchema(
        columns={
            "col1": Column(dtype=int),
            "col2": Column(dtype=float),
        },
        index=Index(dtype=str, name="ind0"),
    )
    return schema


@pytest.fixture
def schema_multiindex() -> DataFrameSchema:
    """Fixture for schema with MultiIndex."""
    schema = DataFrameSchema(
        columns={
            "col1": Column(dtype=int),
            "col2": Column(dtype=float),
        },
        index=MultiIndex(
            [
                Index(dtype=str, name="ind0"),
                Index(dtype=str, name="ind1"),
            ]
        ),
    )
    return schema


@pytest.mark.parametrize("drop", [True, False])
def test_set_index_drop(drop: bool, schema_simple: DataFrameSchema) -> None:
    """Test that setting index correctly handles column dropping."""
    test_schema = schema_simple.set_index(keys=["col1"], drop=drop)
    if drop is True:
        assert len(test_schema.columns) == 1
        assert list(test_schema.columns.keys()) == ["col2"]
    else:
        assert len(test_schema.columns) == 2
        assert list(test_schema.columns.keys()) == ["col1", "col2"]
        assert test_schema.index.name == "col1"


@pytest.mark.parametrize("append", [True, False])
def test_set_index_append(
    append: bool, schema_simple: DataFrameSchema
) -> None:
    """
    Test that setting index correctly handles appending to existing index.
    """

    expected_index_names = ["ind0"]
    test_schema = schema_simple

    for key in ["col1", "col2"]:
        expected_index_names.append(key)
        test_schema = test_schema.set_index(keys=[key], append=append)
        if append is True:
            assert isinstance(test_schema.index, MultiIndex)
            assert [
                x.name for x in test_schema.index.indexes
            ] == expected_index_names
            assert (
                test_schema.index.columns[key].dtype
                == schema_simple.columns[key].dtype
            )
        else:
            assert isinstance(test_schema.index, Index)
            assert test_schema.index.name == key


@pytest.mark.parametrize("drop", [True, False])
def test_reset_index_drop(drop: bool, schema_simple: DataFrameSchema) -> None:
    """Test that resetting index correctly handles dropping index levels."""
    test_schema = schema_simple.reset_index(drop=drop)
    if drop:
        assert len(test_schema.columns) == 2
        assert list(test_schema.columns.keys()) == ["col1", "col2"]
    else:
        assert len(test_schema.columns) == 3
        assert list(test_schema.columns.keys()) == ["col1", "col2", "ind0"]
        assert test_schema.index is None


@pytest.mark.parametrize(
    "level, columns, index",
    [
        [None, {"col1", "col2", "ind0", "ind1"}, None],
        [[], {"col1", "col2"}, ["ind0", "ind1"]],
        [["ind0"], {"col1", "col2", "ind0"}, ["ind1"]],
        [["ind0", "ind1"], {"col1", "col2", "ind0", "ind1"}, None],
    ],
)
def test_reset_index_level(
    schema_multiindex: DataFrameSchema, level, columns, index
):
    """Test that resetting index correctly handles specifying level."""
    test_schema = schema_multiindex.reset_index(level=level)
    if index:
        assert isinstance(test_schema.index, (Index, MultiIndex))
        assert test_schema.index.names == index
    else:
        assert test_schema.index is None

    assert set(test_schema.columns.keys()) == columns


def test_invalid_keys(schema_simple: DataFrameSchema) -> None:
    """Test that re/set_index raises expected exceptions."""
    with pytest.raises(errors.SchemaInitError):
        schema_simple.set_index(["foo", "bar"])
    with pytest.raises(TypeError):
        # mypy correctly identifies the bug
        schema_simple.set_index()  # type: ignore[call-arg]
    with pytest.raises(errors.SchemaInitError):
        schema_simple.reset_index(["foo", "bar"])

    schema_simple.index = None
    with pytest.raises(errors.SchemaInitError):
        schema_simple.reset_index()


def test_update_columns(schema_simple: DataFrameSchema) -> None:
    """Catch-all test for update columns functionality"""

    # Basic function
    test_schema = schema_simple.update_columns({"col2": {"dtype": int}})
    assert (
        schema_simple.columns["col1"].properties
        == test_schema.columns["col1"].properties
    )
    assert test_schema.columns["col2"].dtype == Engine.dtype(int)

    # Multiple columns, multiple properties
    test_schema = schema_simple.update_columns(
        {
            "col1": {"dtype": Category, "coerce": True},
            "col2": {"dtype": Int, "unique": True},
        }
    )
    assert test_schema.columns["col1"].dtype == Engine.dtype(Category)
    assert test_schema.columns["col1"].coerce is True
    assert test_schema.columns["col2"].dtype == Engine.dtype(int)
    assert test_schema.columns["col2"].unique

    # Errors
    with pytest.raises(errors.SchemaInitError):
        schema_simple.update_columns({"col3": {"dtype": int}})
    with pytest.raises(errors.SchemaInitError):
        schema_simple.update_columns({"col1": {"name": "foo"}})
    with pytest.raises(errors.SchemaInitError):
        schema_simple.update_columns({"ind0": {"dtype": int}})


@pytest.mark.parametrize("dtype", [int, None])  # type: ignore
def test_series_schema_dtype(dtype):
    """Series schema dtype property should return
    a Engine-compatible dtype."""
    if dtype is None:
        series_schema = SeriesSchema(dtype)
        assert series_schema.dtype is None
    else:
        assert SeriesSchema(dtype).dtype == Engine.dtype(dtype)


@pytest.mark.parametrize(
    "data, error",
    [
        [
            pd.DataFrame(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=["a", "a", "b"]
            ),
            None,
        ],
        [
            pd.DataFrame(
                [[1, 2, 3], list("xyz"), [7, 8, 9]], columns=["a", "a", "b"]
            ),
            errors.SchemaError,
        ],
    ],
)
@pytest.mark.parametrize(
    "schema",
    [
        DataFrameSchema({"a": Column(int), "b": Column(int)}),
        DataFrameSchema({"a": Column(int, coerce=True), "b": Column(int)}),
        DataFrameSchema({"a": Column(int, regex=True), "b": Column(int)}),
    ],
)
def test_dataframe_duplicated_columns(data, error, schema) -> None:
    """Test that schema can handle dataframes with duplicated columns."""
    if error is None:
        assert isinstance(schema(data), pd.DataFrame)
    else:
        with pytest.raises(error):
            schema(data)


@pytest.mark.parametrize(
    "schema,fields",
    [
        [
            DataFrameSchema(
                columns={"col": Column(int)},
                checks=Check.gt(0),
                parsers=Parser(lambda x: x),
                index=Index(int),
                dtype=int,
                coerce=True,
                strict=True,
                name="schema",
                ordered=False,
            ),
            [
                "columns",
                "checks",
                "parsers",
                "index",
                "dtype",
                "coerce",
                "strict",
                "name",
                "ordered",
            ],
        ],
        [
            MultiIndex(
                indexes=[
                    Index(int),
                    Index(int),
                    Index(int),
                ],
                coerce=True,
                strict=True,
                name="multiindex_schema",
                ordered=True,
            ),
            [
                "indexes",
                "coerce",
                "strict",
                "name",
                "ordered",
            ],
        ],
        [SeriesSchema(int, name="series_schema"), ["type", "name"]],
    ],
)
def test_schema_str_repr(schema, fields: List[str]) -> None:
    """Test the __str__ and __repr__ methods for schemas."""
    for x in [
        schema.__str__(),
        schema.__repr__(),
    ]:
        assert x.startswith(f"<Schema {schema.__class__.__name__}(")
        assert x.endswith(")>")
        for field in fields:
            assert field in x


@pytest.mark.parametrize(
    "unique_kw,expected",
    [
        (["a", "c"], "SchemaError"),
        (["a", "b"], True),
        ([["a", "b"], ["b", "c"]], True),
        ([["a", "b"], ["a", "c"]], "SchemaError"),
        (False, True),
        ((("a", "b"), ["b", "c"]), True),
    ],
)
def test_schema_level_unique_keyword(unique_kw, expected):
    """
    Test that dataframe schema-level unique keyword correctly validates
    uniqueness of multiple columns.
    """
    test_schema = DataFrameSchema(
        columns={"a": Column(int), "b": Column(int), "c": Column(int)},
        unique=unique_kw,
    )
    df = pd.DataFrame({"a": [1, 2, 1], "b": [1, 5, 6], "c": [1, 5, 1]})
    if expected == "SchemaError":
        with pytest.raises(errors.SchemaError):
            test_schema.validate(df)
    else:
        assert isinstance(test_schema.validate(df), pd.DataFrame)


def test_schema_level_unique_missing_columns():
    """Test that the uniqueness level check is skipped if a column is missing"""
    test_schema = DataFrameSchema(
        columns={"a": Column(int), "b": Column(int)},
        unique=["a", "b"],
    )

    df = pd.DataFrame({"a": [1, 2, 1], "c": [1, 5, 1]})

    try:
        test_schema.validate(df, lazy=True)
    except errors.SchemaErrors as err:
        assert len(err.failure_cases) == 3
        assert (
            err.schema_errors[0].reason_code
            == errors.SchemaErrorReason.COLUMN_NOT_IN_DATAFRAME
        )
        assert (
            err.schema_errors[1].reason_code
            == errors.SchemaErrorReason.DUPLICATES
        )


def test_column_set_unique():
    """
    Test that unique Column attribute can be set via property setter and
    update_column method.
    """

    test_schema = DataFrameSchema(
        columns={
            "a": Column(int, unique=True),
            "b": Column(int),
            "c": Column(int),
        }
    )
    assert test_schema.columns["a"].unique
    test_schema.columns["a"].unique = False
    assert not test_schema.columns["a"].unique
    test_schema = test_schema.update_column("a", unique=True)
    assert test_schema.columns["a"].unique


def test_missing_columns():
    """Test that multiple missing columns is correctly reported."""
    schema = DataFrameSchema(
        {
            "column3": Column(int),
            "column2": Column(float),
        }
    )

    df = pd.DataFrame({"column1": [1]})

    try:
        schema.validate(df, lazy=True)
    except errors.SchemaErrors as exc:
        assert (exc.failure_cases["check"] == "column_in_dataframe").all()
        assert exc.failure_cases["failure_case"].tolist() == [
            "column3",
            "column2",
        ]


@pytest.mark.parametrize(
    "series_schema,series,expected_values",
    [
        (
            SeriesSchema(str, default="the second"),
            pd.Series(["the first", None], dtype=str),
            ["the first", "the second"],
        ),
        (
            SeriesSchema(float, default=0.0),
            pd.Series([1.0, None], dtype=float),
            [1.0, 0.0],
        ),
        (
            SeriesSchema(bool, default=False),
            pd.Series([True, None], dtype=bool),
            [True, False],
        ),
        (
            SeriesSchema("Int64", default=0),
            pd.Series([1, None], dtype="Int64"),
            [1, 0],
        ),
    ],
)
def test_series_default_with_correct_dtype(
    series_schema: SeriesSchema, series: pd.Series, expected_values: list
):
    """Test that missing rows are backfilled with the default if missing"""
    validated_series = series_schema.validate(series)
    assert set(validated_series.values) == set(expected_values)


def test_series_default_with_incorrect_dtype_raises_error():
    """Test that if a default with the incorrect dtype is passed, a SchemaError is raised"""
    series_schema = SeriesSchema(str, default=1)

    series = pd.Series(["the first", None])
    with pytest.raises(errors.SchemaError):
        series_schema.validate(series)


@pytest.mark.parametrize(
    "dataframe_schema,dataframe,expected_dataframe",
    [
        (
            DataFrameSchema(
                columns={
                    "a": Column(pd.Int64Dtype(), default=9),
                    "b": Column(pd.Int64Dtype(), nullable=True),
                },
            ),
            pd.DataFrame({"a": [0, None], "b": [None, 5]}, dtype="Int64"),
            pd.DataFrame({"a": [0, 9], "b": [None, 5]}, dtype="Int64"),
        ),
    ],
)
def test_dataframe_default_with_correct_dtype(
    dataframe_schema: DataFrameSchema,
    dataframe: pd.DataFrame,
    expected_dataframe: pd.DataFrame,
):
    """Test that missing rows are backfilled with the default if missing"""
    validated_dataframe = dataframe_schema.validate(dataframe)
    pd.testing.assert_frame_equal(validated_dataframe, expected_dataframe)


def test_default_works_correctly_on_schemas_with_multiple_colummns():
    """Test that each column defaults to the correct value"""

    df = pd.DataFrame(
        {"x": [1, 2, 3], "y": [None, None, None], "z": [None, None, None]}
    ).astype("Int64")

    schema = DataFrameSchema(
        columns={
            "x": Column("Int64", nullable=True, default=-999),
            "y": Column("Int64", nullable=True, default=123),
            "z": Column("Int64", nullable=True, default=1000),
        }
    )

    schema.validate(df, inplace=True)

    expected_df = pd.DataFrame(
        {"x": [1, 2, 3], "y": [123, 123, 123], "z": [1000, 1000, 1000]}
    ).astype("Int64")

    pd.testing.assert_frame_equal(df, expected_df)


def test_pandas_dataframe_subclass_validation():
    """Test that DataFrame subclasses can be validated by pandera."""

    class MyDataFrame(pd.DataFrame):
        """A custom dataframe subclass"""

        @property
        def _constructor(self):
            return MyDataFrame

    dummy_check = Check(lambda _: True)
    schema = DataFrameSchema({"x": Column(int, checks=dummy_check)})

    dataframe = MyDataFrame([1, 2, 3], columns=["x"])
    print(schema.validate(dataframe))


@pytest.mark.parametrize(
    "schema, obj, expected_obj",
    [
        (
            DataFrameSchema(
                {"numbers": Column(int, checks=[Check(lambda x: x >= 3)])},
                drop_invalid_rows=True,
            ),
            pd.DataFrame({"numbers": [1, 2, 3, 4, 5]}),
            pd.DataFrame({"numbers": [3, 4, 5]}),
        ),
        (
            DataFrameSchema({"numbers": Column(str)}, drop_invalid_rows=True),
            pd.DataFrame({"numbers": [1, 2, 3, 4, 5]}),
            pd.DataFrame({"numbers": []}),
        ),
        (
            DataFrameSchema(
                {
                    "letters": Column(str),
                    "numbers": Column(int, checks=[Check(lambda x: x >= 3)]),
                },
                drop_invalid_rows=True,
            ),
            pd.DataFrame(
                {
                    "letters": ["a", "b", "c", "d", "e"],
                    "numbers": [1, 2, 3, 4, 5],
                }
            ),
            pd.DataFrame({"letters": ["c", "d", "e"], "numbers": [3, 4, 5]}),
        ),
    ],
)
def test_drop_invalid_for_dataframe_schema(schema, obj, expected_obj):
    """Test drop_invalid_rows works as expected on DataFrameSchemaBackend.validate"""
    actual_obj = schema.validate(obj, lazy=True)
    actual_obj.index = expected_obj.index
    actual_obj.numbers = actual_obj.numbers.astype(expected_obj.numbers.dtype)

    pd.testing.assert_frame_equal(actual_obj, expected_obj)

    with pytest.raises(errors.SchemaDefinitionError):
        schema.validate(obj, lazy=False)


@pytest.mark.parametrize(
    "schema, obj, expected_obj",
    [
        (
            SeriesSchema(
                int,
                checks=[Check(lambda x: x > 3)],
                drop_invalid_rows=True,
            ),
            pd.Series([9, 6, 3]),
            pd.Series([9, 6]),
        ),
        (
            SeriesSchema(str, drop_invalid_rows=True),
            pd.Series(["nine", 6, "three"]),
            pd.Series(["nine", "three"]),
        ),
    ],
)
def test_drop_invalid_for_series_schema(schema, obj, expected_obj):
    """Test drop_invalid_rows works as expected on SeriesSchemaBackend.validate"""
    actual_obj = schema.validate(obj, lazy=True).reset_index(drop=True)
    expected_obj = expected_obj.reset_index(drop=True)

    pd.testing.assert_series_equal(actual_obj, expected_obj)

    with pytest.raises(errors.SchemaDefinitionError):
        schema.validate(obj, lazy=False)


@pytest.mark.parametrize(
    "col, obj, expected_obj",
    [
        (
            Column(str, name="letters", drop_invalid_rows=True),
            pd.DataFrame({"letters": [None, 1, "c"]}),
            pd.DataFrame({"letters": ["c"]}),
        )
    ],
)
def test_drop_invalid_for_column(col, obj, expected_obj):
    """Test drop_invalid_rows works as expected on ColumnBackend.validate"""
    actual_obj = col.validate(obj, lazy=True)

    pd.testing.assert_frame_equal(
        expected_obj.reset_index(drop=True), actual_obj.reset_index(drop=True)
    )

    with pytest.raises(errors.SchemaDefinitionError):
        col.validate(obj, lazy=False)


def test_drop_invalid_for_model_schema():
    """Test drop_invalid_rows works as expected on DataFrameModel.validate"""

    class MySchema(DataFrameModel):
        """Schema for the test"""

        counter: int = Field(in_range={"min_value": 3, "max_value": 5})

        class Config:
            """Config for the schema model for the test"""

            drop_invalid_rows = True

    expected_obj = pd.DataFrame({"counter": [3, 4, 5]})

    actual_obj = MySchema.validate(
        pd.DataFrame({"counter": [1, 2, 3, 4, 5, 6]}), lazy=True
    )

    actual_obj.index = expected_obj.index
    pd.testing.assert_frame_equal(expected_obj, actual_obj)

    with pytest.raises(errors.SchemaDefinitionError):
        MySchema.validate(actual_obj, lazy=False)


def test_schema_coerce() -> None:
    """Test that setting coerce=True for a DataFrameSchema is sufficient to coerce a column."""

    schema = DataFrameSchema(
        columns={"col": Column(dtype=bool)},
        coerce=True,
    )

    df = pd.DataFrame({"col": [1, 0]})

    assert isinstance(schema.validate(df), pd.DataFrame)


def test_schema_coerce_with_regex() -> None:
    """Test that setting coerce=True for a DataFrameSchema is sufficient to coerce a column in the case
    where the column has regex=True."""

    schema_with_regex = DataFrameSchema(
        columns={"col": Column(dtype=bool, regex=True)},
        coerce=True,
    )

    df = pd.DataFrame({"col": [1, 0]})

    assert isinstance(schema_with_regex.validate(df), pd.DataFrame)


@pytest.mark.parametrize(
    "schema, obj, expected_obj",
    [
        (
            DataFrameSchema(
                columns={
                    "name": Column(str),
                    "occupation": Column(str, nullable=False),
                },
                index=MultiIndex(
                    [
                        Index(str, name="state"),
                        Index(str, name="city"),
                    ]
                ),
                drop_invalid_rows=True,
            ),
            pd.DataFrame(
                {
                    "name": ["Frodo", "Boromir"],
                    "occupation": ["Ring bearer", None],
                },
                index=pd.MultiIndex.from_tuples(
                    (("MiddleEarth", "TheShire"), ("MiddleEarth", "Gondor")),
                    names=["state", "city"],
                ),
            ),
            pd.DataFrame(
                {"name": ["Frodo"], "occupation": ["Ring bearer"]},
                index=pd.MultiIndex.from_tuples(
                    (("MiddleEarth", "TheShire"),), names=["state", "city"]
                ),
            ),
        ),
        (
            DataFrameSchema(
                columns={
                    "path_description": Column(str, nullable=False),
                    "days_to_travel": Column(float, nullable=False),
                },
                index=MultiIndex(
                    [
                        Index(str, name="character_name"),
                        Index(int, name="path_id"),
                    ]
                ),
                drop_invalid_rows=True,
            ),
            pd.DataFrame(
                {
                    "path_description": [
                        "To Rivendell",
                        "To Mordor",
                        "To Gondor",
                        None,
                    ],
                    "days_to_travel": [30.0, 60.5, None, 15.9],
                },
                index=pd.MultiIndex.from_tuples(
                    (("Frodo", 1), ("Sam", 2), ("Boromir", 3), ("Legolas", 4)),
                    names=["character_name", "path_id"],
                ),
            ),
            pd.DataFrame(
                {
                    "path_description": [
                        "To Rivendell",
                        "To Mordor",
                    ],
                    "days_to_travel": [30.0, 60.5],
                },
                index=pd.MultiIndex.from_tuples(
                    (("Frodo", 1), ("Sam", 2)),
                    names=["character_name", "path_id"],
                ),
            ),
        ),
        (
            DataFrameSchema(
                columns={
                    "battle_name": Column(str, nullable=False),
                    "victor": Column(str, nullable=False),
                },
                index=MultiIndex(
                    [
                        Index(int, name="year"),
                        Index(float, name="coordinates"),
                    ]
                ),
                drop_invalid_rows=True,
            ),
            pd.DataFrame(
                {
                    "battle_name": [
                        "Battle of Helm's Deep",
                        "Battle of the Black Gate",
                        "Siege of Gondor",
                        "Skirmish at Weathertop",
                    ],
                    "victor": [
                        "Rohan & Allies",
                        "Free Peoples",
                        None,
                        "Nazgûl",
                    ],
                },
                index=pd.MultiIndex.from_tuples(
                    ((3019, 42.5), (3019, 42.6), (3019, 42.7), (3018, 42.8)),
                    names=["year", "coordinates"],
                ),
            ),
            pd.DataFrame(
                {
                    "battle_name": [
                        "Battle of Helm's Deep",
                        "Battle of the Black Gate",
                        "Skirmish at Weathertop",
                    ],
                    "victor": ["Rohan & Allies", "Free Peoples", "Nazgûl"],
                },
                index=pd.MultiIndex.from_tuples(
                    ((3019, 42.5), (3019, 42.6), (3018, 42.8)),
                    names=["year", "coordinates"],
                ),
            ),
        ),
    ],
)
def test_drop_invalid_for_multi_index(schema, obj, expected_obj):
    """Test drop_invalid_rows works as expected on multi-index dataframes"""
    actual_obj = schema.validate(obj, lazy=True)

    pd.testing.assert_frame_equal(actual_obj, expected_obj)


def test_get_schema_metadata():
    """Test fetching schema metadata."""

    schema = DataFrameSchema(
        columns={
            "int_column": Column(int, metadata={"key": "value"}),
            "float_column": Column(float, metadata={"key": [1, 2, 3]}),
            "str_column": Column(str, metadata={"key": {"nested": "value"}}),
        },
        metadata={"df_level": "schema-level metadata"},
        name="dataframe_name",
    )
    metadata = schema.get_metadata()
    expected = {
        "dataframe_name": {
            "columns": {
                "int_column": {"key": "value"},
                "float_column": {
                    "key": [
                        1,
                        2,
                        3,
                    ]
                },
                "str_column": {"key": {"nested": "value"}},
            },
            "dataframe": {"df_level": "schema-level metadata"},
        }
    }
    assert expected == metadata


@pytest.fixture()
def xfail_int_with_nans(request):
    dtype = request.getfixturevalue("dtype")
    input_value = request.getfixturevalue("input_value")
    coerce = request.getfixturevalue("coerce")

    if dtype == "Int64" and input_value is not None and not coerce:
        pytest.xfail("NaN is considered a Float64")


@pytest.mark.parametrize(
    "dtype,default",
    [
        (str, "a default"),
        (bool, True),
        (bool, False),
        (float, 42.0),
        ("Int64", 0),
    ],
)
@pytest.mark.parametrize("input_value", [None, np.nan])
@pytest.mark.parametrize("coerce", [True, False])
@pytest.mark.usefixtures("xfail_int_with_nans")
def test_schema_column_default_handle_nans(
    input_value: Any, coerce: bool, dtype: Any, default: Any
):
    """Test ``default`` fills ``nan`` values as expected."""
    schema = DataFrameSchema(
        columns={"column1": Column(dtype, default=default, coerce=coerce)}
    )
    df = pd.DataFrame({"column1": [input_value]})
    schema.validate(df, inplace=True)
    assert df.iloc[0]["column1"] == default


@pytest.mark.parametrize(
    "schema, obj, expected_obj, check_dtype",
    [
        (
            DataFrameSchema(
                columns={
                    "temperature": Column(float, nullable=False),
                },
                index=MultiIndex(
                    [
                        Index(pd.Timestamp, name="timestamp"),
                        Index(str, name="city"),
                    ]
                ),
                drop_invalid_rows=True,
            ),
            pd.DataFrame(
                {
                    "temperature": [
                        3.0,
                        4.0,
                        5.0,
                        5.0,
                        np.nan,
                        2.0,
                    ],
                },
                index=pd.MultiIndex.from_tuples(
                    (
                        (pd.Timestamp("2022-01-01"), "Paris"),
                        (pd.Timestamp("2023-01-01"), "Paris"),
                        (pd.Timestamp("2024-01-01"), "Paris"),
                        (pd.Timestamp("2022-01-01"), "Oslo"),
                        (pd.Timestamp("2023-01-01"), "Oslo"),
                        (pd.Timestamp("2024-01-01"), "Oslo"),
                    ),
                    names=["timestamp", "city"],
                ),
            ),
            pd.DataFrame(
                {
                    "temperature": [3.0, 4.0, 5.0, 5.0, 2.0],
                },
                index=pd.MultiIndex.from_tuples(
                    (
                        (pd.Timestamp("2022-01-01"), "Paris"),
                        (pd.Timestamp("2023-01-01"), "Paris"),
                        (pd.Timestamp("2024-01-01"), "Paris"),
                        (pd.Timestamp("2022-01-01"), "Oslo"),
                        (pd.Timestamp("2024-01-01"), "Oslo"),
                    ),
                    names=["timestamp", "city"],
                ),
            ),
            True,
        ),
        (
            DataFrameSchema(
                columns={
                    "temperature": Column(float, nullable=False),
                },
                index=MultiIndex(
                    [
                        Index(pd.Timestamp, name="timestamp"),
                        Index(str, name="city"),
                    ]
                ),
                drop_invalid_rows=True,
            ),
            pd.DataFrame(
                {
                    "temperature": [
                        3.0,
                        4.0,
                        5.0,
                        -1.0,
                        np.nan,
                        -2.0,
                        4.0,
                        5.0,
                        2.0,
                    ],
                },
                index=pd.MultiIndex.from_tuples(
                    (
                        (pd.Timestamp("2022-01-01"), "Paris"),
                        (pd.Timestamp("2023-01-01"), "Paris"),
                        (pd.Timestamp("2024-01-01"), "Paris"),
                        (pd.Timestamp("2022-01-01"), "Oslo"),
                        (pd.Timestamp("2023-01-01"), "Oslo"),
                        (pd.Timestamp("2024-01-01"), "Oslo"),
                        (
                            pd.Timestamp("2024-01-01", tz="Europe/London"),
                            "London",
                        ),
                        (pd.Timestamp(pd.NaT), "Frankfurt"),
                        (pd.Timestamp("2024-01-01"), 6),
                    ),
                    names=["timestamp", "city"],
                ),
            ),
            pd.DataFrame(
                {
                    "temperature": [3.0, 4.0, 5.0, -1.0, -2.0, 4],
                },
                index=pd.MultiIndex.from_tuples(
                    (
                        (pd.Timestamp("2022-01-01"), "Paris"),
                        (pd.Timestamp("2023-01-01"), "Paris"),
                        (pd.Timestamp("2024-01-01"), "Paris"),
                        (pd.Timestamp("2022-01-01"), "Oslo"),
                        (pd.Timestamp("2024-01-01"), "Oslo"),
                        (
                            pd.Timestamp("2024-01-01", tz="Europe/London"),
                            "London",
                        ),
                    ),
                    names=["timestamp", "city"],
                ),
            ),
            False,
        ),
    ],
)
def test_drop_invalid_for_multi_index_with_datetime(
    schema, obj, expected_obj, check_dtype
):
    """Test drop_invalid_rows works as expected on multi-index dataframes"""
    actual_obj = schema.validate(obj, lazy=True)

    # the datatype of the index is not casted, In this cases its an object
    pd.testing.assert_frame_equal(
        actual_obj,
        expected_obj,
        check_dtype=check_dtype,
        check_index_type=check_dtype,
    )
