# pylint: disable=undefined-variable,redefined-outer-name,invalid-name,undefined-loop-variable,too-many-lines  # noqa
"""Unit tests for pandera data generating strategies."""
import datetime
import operator
import re
from typing import Any, Callable, Optional, Set
from unittest.mock import MagicMock
from warnings import catch_warnings

import numpy as np
import pandas as pd
import pytest

import pandera as pa
from pandera import strategies
from pandera.api.checks import Check
from pandera.api.extensions import register_check_statistics
from pandera.dtypes import is_category, is_complex, is_float
from pandera.engines import pandas_engine, geopandas_engine

try:
    import hypothesis
    import hypothesis.extra.numpy as npst
    import hypothesis.strategies as st
except ImportError:
    HAS_HYPOTHESIS = False
    hypothesis = MagicMock()
    st = MagicMock()
else:
    HAS_HYPOTHESIS = True


UNSUPPORTED_DTYPE_CLS: Set[Any] = {
    pandas_engine.Interval,
    pandas_engine.Period,
    pandas_engine.Sparse,
    pandas_engine.PydanticModel,
    pandas_engine.Decimal,
    pandas_engine.Date,
    pandas_engine.PythonDict,
    pandas_engine.PythonList,
    pandas_engine.PythonTuple,
    pandas_engine.PythonTypedDict,
    pandas_engine.PythonNamedTuple,
}

if pandas_engine.PYARROW_INSTALLED and pandas_engine.PANDAS_2_0_0_PLUS:
    UNSUPPORTED_DTYPE_CLS.update(
        [
            pandas_engine.ArrowBool,
            pandas_engine.ArrowDecimal128,
            pandas_engine.ArrowDictionary,
            pandas_engine.ArrowFloat16,
            pandas_engine.ArrowFloat32,
            pandas_engine.ArrowFloat64,
            pandas_engine.ArrowInt8,
            pandas_engine.ArrowInt16,
            pandas_engine.ArrowInt32,
            pandas_engine.ArrowInt64,
            pandas_engine.ArrowString,
            pandas_engine.ArrowTimestamp,
            pandas_engine.ArrowUInt8,
            pandas_engine.ArrowUInt16,
            pandas_engine.ArrowUInt32,
            pandas_engine.ArrowUInt64,
            pandas_engine.ArrowList,
            pandas_engine.ArrowStruct,
            pandas_engine.ArrowNull,
            pandas_engine.ArrowDate32,
            pandas_engine.ArrowDate64,
            pandas_engine.ArrowDuration,
            pandas_engine.ArrowTime32,
            pandas_engine.ArrowTime64,
            pandas_engine.ArrowMap,
            pandas_engine.ArrowBinary,
            pandas_engine.ArrowLargeBinary,
            pandas_engine.ArrowLargeString,
        ]
    )

SUPPORTED_DTYPES = set()
for data_type in pandas_engine.Engine.get_registered_dtypes():
    if (
        # valid hypothesis.strategies.floats <=64
        getattr(data_type, "bit_width", -1) > 64
        or is_category(data_type)
        or data_type in UNSUPPORTED_DTYPE_CLS
        or (
            geopandas_engine.GEOPANDAS_INSTALLED
            and data_type == geopandas_engine.Geometry
        )
    ):
        continue

    SUPPORTED_DTYPES.add(pandas_engine.Engine.dtype(data_type))

SUPPORTED_DTYPES.add(pandas_engine.Engine.dtype("datetime64[ns, UTC]"))

NUMERIC_DTYPES = [
    data_type for data_type in SUPPORTED_DTYPES if data_type.continuous
]

NULLABLE_DTYPES = [
    data_type
    for data_type in SUPPORTED_DTYPES
    if not is_complex(data_type)
    and not is_category(data_type)
    and not data_type == pandas_engine.Engine.dtype("object")
]


@pytest.mark.parametrize(
    "data_type",
    [
        pa.Category,
        pandas_engine.Interval(  # type: ignore # pylint:disable=unexpected-keyword-arg,no-value-for-parameter
            subtype=np.int64
        ),
    ],
)
def test_unsupported_pandas_dtype_strategy(data_type):
    """Test unsupported pandas dtype strategy raises error."""
    with pytest.raises(TypeError, match=r"is currently unsupported"):
        strategies.pandas_dtype_strategy(data_type)


@pytest.mark.parametrize("data_type", SUPPORTED_DTYPES)
@hypothesis.given(st.data())
def test_pandas_dtype_strategy(data_type, data):
    """Test that series can be constructed from pandas dtype."""

    strategy = strategies.pandas_dtype_strategy(data_type)
    example = data.draw(strategy)

    expected_type = strategies.to_numpy_dtype(data_type).type
    if isinstance(example, pd.Timestamp):
        example = example.to_numpy()
    assert example.dtype.type == expected_type

    chained_strategy = strategies.pandas_dtype_strategy(data_type, strategy)
    chained_example = data.draw(chained_strategy)
    if isinstance(chained_example, pd.Timestamp):
        chained_example = chained_example.to_numpy()
    assert chained_example.dtype.type == expected_type


@pytest.mark.parametrize("data_type", NUMERIC_DTYPES)
@hypothesis.given(st.data())
def test_check_strategy_continuous(data_type, data):
    """Test built-in check strategies can generate continuous data."""
    np_dtype = strategies.to_numpy_dtype(data_type)
    value = data.draw(
        npst.from_dtype(
            strategies.to_numpy_dtype(data_type),
            allow_nan=False,
            allow_infinity=False,
        )
    )
    # don't overstep bounds of representation
    hypothesis.assume(np.finfo(np_dtype).min < value < np.finfo(np_dtype).max)

    assert data.draw(strategies.ne_strategy(data_type, value=value)) != value
    assert data.draw(strategies.eq_strategy(data_type, value=value)) == value
    assert (
        data.draw(strategies.gt_strategy(data_type, min_value=value)) > value
    )
    assert (
        data.draw(strategies.ge_strategy(data_type, min_value=value)) >= value
    )
    assert (
        data.draw(strategies.lt_strategy(data_type, max_value=value)) < value
    )
    assert (
        data.draw(strategies.le_strategy(data_type, max_value=value)) <= value
    )


def value_ranges(data_type: pa.DataType):
    """Strategy to generate value range based on the pandas datatype."""
    kwargs = dict(
        allow_nan=False,
        allow_infinity=False,
        exclude_min=False,
        exclude_max=False,
    )
    return (
        st.tuples(
            strategies.pandas_dtype_strategy(
                data_type, strategy=None, **kwargs
            ),
            strategies.pandas_dtype_strategy(
                data_type, strategy=None, **kwargs
            ),
        )
        .map(sorted)
        .filter(lambda x: x[0] < x[1])  # type: ignore
    )


@pytest.mark.parametrize("data_type", NUMERIC_DTYPES)
@pytest.mark.parametrize(
    "strat_fn, arg_name, base_st_type, compare_op",
    [
        [strategies.ne_strategy, "value", "type", operator.ne],
        [strategies.eq_strategy, "value", "just", operator.eq],
        [strategies.gt_strategy, "min_value", "limit", operator.gt],
        [strategies.ge_strategy, "min_value", "limit", operator.ge],
        [strategies.lt_strategy, "max_value", "limit", operator.lt],
        [strategies.le_strategy, "max_value", "limit", operator.le],
    ],
)
@hypothesis.given(st.data())
def test_check_strategy_chained_continuous(
    data_type, strat_fn, arg_name, base_st_type, compare_op, data
):
    """
    Test built-in check strategies can generate continuous data building off
    of a parent strategy.
    """
    min_value, max_value = data.draw(value_ranges(data_type))
    hypothesis.assume(min_value < max_value)
    value = min_value
    base_st = strategies.pandas_dtype_strategy(
        data_type,
        min_value=min_value,
        max_value=max_value,
        allow_nan=False,
        allow_infinity=False,
    )
    if base_st_type == "type":
        assert_base_st = base_st
    elif base_st_type == "just":
        assert_base_st = st.just(value)
    elif base_st_type == "limit":
        assert_base_st = strategies.pandas_dtype_strategy(
            data_type,
            min_value=min_value,
            max_value=max_value,
            allow_nan=False,
            allow_infinity=False,
        )
    else:
        raise RuntimeError(f"base_st_type {base_st_type} not recognized")

    local_vars = locals()
    assert_value = local_vars[arg_name]
    example = data.draw(
        strat_fn(data_type, assert_base_st, **{arg_name: assert_value})
    )
    assert compare_op(example, assert_value)


@pytest.mark.parametrize("data_type", NUMERIC_DTYPES)
@pytest.mark.parametrize("chained", [True, False])
@hypothesis.given(st.data())
def test_in_range_strategy(data_type, chained, data):
    """Test the built-in in-range strategy can correctly generate data."""
    min_value, max_value = data.draw(value_ranges(data_type))
    hypothesis.assume(min_value < max_value)

    base_st_in_range = None
    if chained:
        if is_float(data_type):
            base_st_kwargs = {
                "exclude_min": False,
                "exclude_max": False,
            }
        else:
            base_st_kwargs = {}

        # constraining the strategy this way makes testing more efficient
        base_st_in_range = strategies.pandas_dtype_strategy(
            data_type,
            min_value=min_value,
            max_value=max_value,
            **base_st_kwargs,  # type: ignore[arg-type]
        )
    strat = strategies.in_range_strategy(
        data_type,
        base_st_in_range,
        min_value=min_value,
        max_value=max_value,
    )

    assert min_value <= data.draw(strat) <= max_value


@pytest.mark.parametrize(
    "data_type",
    [data_type for data_type in SUPPORTED_DTYPES if data_type.continuous],
)
@pytest.mark.parametrize("chained", [True, False])
@hypothesis.given(st.data())
def test_isin_notin_strategies(data_type, chained, data):
    """Test built-in check strategies that rely on discrete values."""
    value_st = strategies.pandas_dtype_strategy(
        data_type,
        allow_nan=False,
        allow_infinity=False,
        exclude_min=False,
        exclude_max=False,
    )
    values = [data.draw(value_st) for _ in range(10)]

    isin_base_st = None
    notin_base_st = None
    if chained:
        base_values = values + [data.draw(value_st) for _ in range(10)]
        isin_base_st = strategies.isin_strategy(
            data_type, allowed_values=base_values
        )
        notin_base_st = strategies.notin_strategy(
            data_type, forbidden_values=base_values
        )

    isin_st = strategies.isin_strategy(
        data_type, isin_base_st, allowed_values=values
    )
    notin_st = strategies.notin_strategy(
        data_type, notin_base_st, forbidden_values=values
    )
    assert data.draw(isin_st) in values
    assert data.draw(notin_st) not in values


@pytest.mark.parametrize(
    "str_strat, pattern_fn",
    [
        [
            strategies.str_matches_strategy,
            lambda patt: f"^{patt}$",
        ],
        [strategies.str_contains_strategy, None],
        [strategies.str_startswith_strategy, None],
        [strategies.str_endswith_strategy, None],
    ],
)
@pytest.mark.parametrize("chained", [True, False])
@hypothesis.given(st.data(), st.text())
def test_str_pattern_checks(
    str_strat: Callable,
    pattern_fn: Optional[Callable[..., str]],
    chained: bool,
    data,
    pattern,
) -> None:
    """Test built-in check strategies for string pattern checks."""
    try:
        re.compile(pattern)
        re_compiles = True
    except re.error:
        re_compiles = False
    hypothesis.assume(re_compiles)

    pattern = pattern if pattern_fn is None else pattern_fn(pattern)

    base_st = None
    if chained:
        try:
            base_st = str_strat(pa.String, pattern=pattern)
        except TypeError:
            base_st = str_strat(pa.String, string=pattern)

    try:
        st = str_strat(pa.String, base_st, pattern=pattern)
    except TypeError:
        st = str_strat(pa.String, base_st, string=pattern)
    example = data.draw(st)

    assert re.search(pattern, example)


@pytest.mark.parametrize("chained", [True, False])
@hypothesis.given(
    st.data(),
    (
        st.tuples(
            st.integers(min_value=0, max_value=100),
            st.integers(min_value=0, max_value=100),
        )
        .map(sorted)  # type: ignore[arg-type]
        .filter(lambda x: x[0] < x[1])  # type: ignore
    ),
)
def test_str_length_checks(chained, data, value_range):
    """Test built-in check strategies for string length."""
    min_value, max_value = value_range
    base_st = None
    if chained:
        base_st = strategies.str_length_strategy(
            pa.String,
            min_value=max(0, min_value - 5),
            max_value=max_value + 5,
        )
    str_length_st = strategies.str_length_strategy(
        pa.String, base_st, min_value=min_value, max_value=max_value
    )
    example = data.draw(str_length_st)
    assert min_value <= len(example) <= max_value


@hypothesis.given(st.data())
def test_register_check_strategy(data) -> None:
    """Test registering check strategy on a custom check."""

    # pylint: disable=unused-argument
    def custom_eq_strategy(
        pandas_dtype: pa.DataType,
        strategy: st.SearchStrategy = None,
        *,
        value: Any,
    ):
        return st.just(value).map(strategies.to_numpy_dtype(pandas_dtype).type)

    # pylint: disable=no-member
    class CustomCheck(Check):
        """Custom check class."""

        @classmethod
        @strategies.register_check_strategy(custom_eq_strategy)
        @register_check_statistics(["value"])
        def custom_equals(cls, value, **kwargs) -> "CustomCheck":
            """Define a built-in check."""

            def _custom_equals(series: pd.Series) -> pd.Series:
                """Comparison function for check"""
                return series == value

            return cls(
                _custom_equals,
                name=cls.custom_equals.__name__,
                error=f"equal_to({value})",
                **kwargs,
            )

    check = CustomCheck.custom_equals(100)
    result = data.draw(check.strategy(pa.Int(), **check.statistics))
    assert result == 100


def test_register_check_strategy_exception() -> None:
    """Check method needs statistics attr to register a strategy."""

    def custom_strat() -> None:
        pass

    class CustomCheck(Check):
        """Custom check class."""

        @classmethod
        @strategies.register_check_strategy(custom_strat)  # type: ignore[arg-type]
        # mypy correctly identifies the error
        def custom_check(cls, **kwargs) -> "CustomCheck":
            """Built-in check with no statistics."""

            def _custom_check(series: pd.Series) -> pd.Series:
                """Some check function."""
                return series

            return cls(
                _custom_check,
                name=cls.custom_check.__name__,
                **kwargs,
            )

    assert not CustomCheck.custom_check().statistics


@hypothesis.given(st.data())
def test_series_strategy(data) -> None:
    """Test SeriesSchema strategy."""
    series_schema = pa.SeriesSchema(pa.Int(), pa.Check.gt(0))
    series_schema(data.draw(series_schema.strategy()))


def test_series_example() -> None:
    """Test SeriesSchema example method generate examples that pass."""
    series_schema = pa.SeriesSchema(pa.Int(), pa.Check.gt(0))
    for _ in range(10):
        series_schema(series_schema.example())


@hypothesis.given(st.data())
def test_column_strategy(data) -> None:
    """Test Column schema strategy."""
    column_schema = pa.Column(pa.Int(), pa.Check.gt(0), name="column")
    column_schema(data.draw(column_schema.strategy()))


def test_column_example():
    """Test Column schema example method generate examples that pass."""
    column_schema = pa.Column(pa.Int(), pa.Check.gt(0), name="column")
    for _ in range(10):
        column_schema(column_schema.example())


@pytest.mark.parametrize("data_type", SUPPORTED_DTYPES)
@pytest.mark.parametrize("size", [None, 0, 1, 3, 5])
@hypothesis.given(st.data())
def test_dataframe_strategy(data_type, size, data):
    """Test DataFrameSchema strategy."""
    dataframe_schema = pa.DataFrameSchema(
        {f"{data_type}_col": pa.Column(data_type)}
    )
    df_sample = data.draw(dataframe_schema.strategy(size=size))
    if size == 0:
        assert df_sample.empty
    elif size is None:
        assert df_sample.empty or isinstance(
            dataframe_schema(df_sample), pd.DataFrame
        )
    else:
        assert isinstance(dataframe_schema(df_sample), pd.DataFrame)

    with pytest.raises(pa.errors.BaseStrategyOnlyError):
        strategies.dataframe_strategy(
            data_type, strategies.pandas_dtype_strategy(data_type)
        )


def test_dataframe_strategy_empty_localized():
    """Test DataFrameSchema strategy localizes timezones when empty."""
    schema = pa.DataFrameSchema(
        {"localized": pa.Column(pd.DatetimeTZDtype(tz="UTC", unit="ns"))}
    )
    example = schema.example(0)
    schema(example)


@pytest.mark.parametrize("size", [None, 0, 1, 3, 5])
@hypothesis.given(st.data())
def test_dataframe_strategy_with_check(size, data):
    """Test DataFrameSchema strategy with dataframe-level check."""
    dataframe_schema = pa.DataFrameSchema(
        {"col": pa.Column(float)},
        checks=pa.Check.in_range(5, 10),
    )
    df_sample = data.draw(dataframe_schema.strategy(size=size))
    if size == 0:
        assert df_sample.empty
    elif size is None:
        assert df_sample.empty or isinstance(
            dataframe_schema(df_sample), pd.DataFrame
        )
    else:
        assert isinstance(dataframe_schema(df_sample), pd.DataFrame)


@hypothesis.given(st.data())
def test_dataframe_example(data) -> None:
    """Test DataFrameSchema example method generate examples that pass."""
    schema = pa.DataFrameSchema({"column": pa.Column(int, pa.Check.gt(0))})
    df_sample = data.draw(schema.strategy(size=10))
    schema(df_sample)


@pytest.mark.parametrize("size", [3, 5, 10])
@hypothesis.given(st.data())
def test_dataframe_unique(size, data) -> None:
    """Test that DataFrameSchemas with unique columns are actually unique."""
    schema = pa.DataFrameSchema(
        {
            "col1": pa.Column(int),
            "col2": pa.Column(float),
            "col3": pa.Column(str),
            "col4": pa.Column(int),
        },
        unique=["col1", "col2", "col3"],
    )
    df_sample = data.draw(schema.strategy(size=size))
    schema(df_sample)


@pytest.mark.parametrize(
    "regex",
    [
        "col_[0-9]{1,4}",
        "[a-zA-Z]+_foobar",
        "[a-z]+_[0-9]+_[a-z]+",
    ],
)
@hypothesis.given(st.data(), st.integers(min_value=-5, max_value=5))
def test_dataframe_with_regex(regex: str, data, n_regex_columns: int) -> None:
    """Test DataFrameSchema strategy with regex columns"""
    dataframe_schema = pa.DataFrameSchema({regex: pa.Column(int, regex=True)})
    if n_regex_columns < 1:
        with pytest.raises(ValueError):
            dataframe_schema.strategy(size=5, n_regex_columns=n_regex_columns)
    else:
        df = dataframe_schema(
            data.draw(
                dataframe_schema.strategy(
                    size=5, n_regex_columns=n_regex_columns
                )
            )
        )
        assert df.shape[1] == n_regex_columns


@pytest.mark.parametrize("data_type", NUMERIC_DTYPES)
@hypothesis.given(st.data())
def test_dataframe_checks(data_type, data):
    """Test dataframe strategy with checks defined at the dataframe level."""
    min_value, max_value = data.draw(value_ranges(data_type))
    dataframe_schema = pa.DataFrameSchema(
        {f"{data_type}_col": pa.Column(data_type) for _ in range(5)},
        checks=pa.Check.in_range(min_value, max_value),
    )
    strat = dataframe_schema.strategy(size=5)
    example = data.draw(strat)
    dataframe_schema(example)


@pytest.mark.parametrize(
    "data_type", [pa.Int(), pa.Float, pa.String, pa.DateTime]
)
@hypothesis.given(st.data())
def test_dataframe_strategy_with_indexes(data_type, data):
    """Test dataframe strategy with index and multiindex components."""
    dataframe_schema_index = pa.DataFrameSchema(index=pa.Index(data_type))
    dataframe_schema_multiindex = pa.DataFrameSchema(
        index=pa.MultiIndex(
            [pa.Index(data_type, name=f"index{i}") for i in range(3)]
        )
    )

    dataframe_schema_index(data.draw(dataframe_schema_index.strategy(size=10)))
    example = data.draw(dataframe_schema_multiindex.strategy(size=10))
    dataframe_schema_multiindex(example)


@hypothesis.given(st.data())
def test_index_strategy(data) -> None:
    """Test Index schema component strategy."""
    data_type = pa.Int()
    index_schema = pa.Index(data_type, unique=True, name="index")
    strat = index_schema.strategy(size=10)
    example = data.draw(strat)

    assert (~example.duplicated()).all()
    actual_data_type = pandas_engine.Engine.dtype(example.dtype)
    assert data_type.check(actual_data_type)
    index_schema(pd.DataFrame(index=example))


def test_index_example() -> None:
    """
    Test Index schema component example method generates examples that pass.
    """
    data_type = pa.Int()
    index_schema = pa.Index(data_type, unique=True)
    for _ in range(10):
        index_schema(pd.DataFrame(index=index_schema.example()))


@hypothesis.given(st.data())
def test_multiindex_strategy(data) -> None:
    """Test MultiIndex schema component strategy."""
    data_type = pa.Float()
    multiindex = pa.MultiIndex(
        indexes=[
            pa.Index(data_type, unique=True, name="level_0"),
            pa.Index(data_type, nullable=True),
            pa.Index(data_type),
        ]
    )
    strat = multiindex.strategy(size=10)
    example = data.draw(strat)
    for i in range(example.nlevels):
        actual_data_type = pandas_engine.Engine.dtype(
            example.get_level_values(i).dtype
        )
        assert data_type.check(actual_data_type)

    with pytest.raises(pa.errors.BaseStrategyOnlyError):
        strategies.multiindex_strategy(
            data_type, strategies.pandas_dtype_strategy(data_type)
        )


def test_multiindex_example() -> None:
    """
    Test MultiIndex schema component example method generates examples that
    pass.
    """
    data_type = pa.Float()
    multiindex = pa.MultiIndex(
        indexes=[
            pa.Index(data_type, unique=True, name="level_0"),
            pa.Index(data_type, nullable=True),
            pa.Index(data_type),
            pa.Index(pd.DatetimeTZDtype(tz="UTC", unit="ns")),
        ]
    )
    for _ in range(10):
        example = multiindex.example()
        multiindex(pd.DataFrame(index=example))


@pytest.mark.parametrize("data_type", NULLABLE_DTYPES)
@hypothesis.given(st.data())
def test_field_element_strategy(data_type, data):
    """Test strategy for generating elements in columns/indexes."""
    strategy = strategies.field_element_strategy(data_type)
    element = data.draw(strategy)

    expected_type = strategies.to_numpy_dtype(data_type).type
    if strategies.pandas_strategies._is_datetime_tz(data_type):
        assert isinstance(element, pd.Timestamp)
        assert element.tz == data_type.tz
    else:
        assert element.dtype.type == expected_type

    with pytest.raises(pa.errors.BaseStrategyOnlyError):
        strategies.field_element_strategy(
            data_type, strategies.pandas_dtype_strategy(data_type)
        )


@pytest.mark.parametrize("data_type", NULLABLE_DTYPES)
@pytest.mark.parametrize(
    "field_strategy",
    [strategies.index_strategy, strategies.series_strategy],
)
@pytest.mark.parametrize("nullable", [True, False])
@hypothesis.given(st.data())
def test_check_nullable_field_strategy(
    data_type, field_strategy, nullable, data
):
    """Test strategies for generating nullable column/index data."""
    size = 5

    if (
        str(data_type) == "float16"
        and field_strategy.__name__ == "index_strategy"
    ):
        pytest.xfail("float16 is not supported for indexes")

    strat = field_strategy(data_type, nullable=nullable, size=size)
    example = data.draw(strat)

    if nullable:
        assert example.isna().sum() >= 0
    else:
        assert example.notna().all()


@pytest.mark.parametrize("data_type", NULLABLE_DTYPES)
@pytest.mark.parametrize("nullable", [True, False])
@hypothesis.given(st.data())
def test_check_nullable_dataframe_strategy(data_type, nullable, data):
    """Test strategies for generating nullable DataFrame data."""
    size = 5
    # pylint: disable=no-value-for-parameter
    strat = strategies.dataframe_strategy(
        columns={"col": pa.Column(data_type, nullable=nullable, name="col")},
        size=size,
    )
    example = data.draw(strat)
    if nullable:
        assert example.isna().sum(axis=None).item() >= 0
    else:
        assert example.notna().all(axis=None)


@pytest.mark.parametrize(
    "schema, warning",
    [
        [
            pa.SeriesSchema(
                pa.Int(),
                checks=[
                    pa.Check(lambda x: x > 0, element_wise=True),
                    pa.Check(lambda x: x > -10, element_wise=True),
                ],
            ),
            "Element-wise",
        ],
        [
            pa.SeriesSchema(
                pa.Int(),
                checks=[
                    pa.Check(lambda s: s > -10000),
                    pa.Check(lambda s: s > -9999),
                ],
            ),
            "Vectorized",
        ],
    ],
)
@hypothesis.given(st.data())
def test_series_strategy_undefined_check_strategy(
    schema: pa.SeriesSchema, warning: str, data
) -> None:
    """Test case where series check strategy is undefined."""
    with pytest.warns(
        UserWarning, match=f"{warning} check doesn't have a defined strategy"
    ):
        strat = schema.strategy(size=5)
    example = data.draw(strat)
    schema(example)


@pytest.mark.parametrize(
    "schema, warning",
    [
        [
            pa.DataFrameSchema(
                columns={"column": pa.Column(int)},
                checks=[
                    pa.Check(lambda x: x > 0, element_wise=True),
                    pa.Check(lambda x: x > -10, element_wise=True),
                ],
            ),
            "Element-wise",
        ],
        [
            pa.DataFrameSchema(
                columns={
                    "column": pa.Column(
                        int,
                        checks=[
                            pa.Check(lambda s: s > -10000),
                            pa.Check(lambda s: s > -9999),
                        ],
                    )
                },
            ),
            "Column",
        ],
        # schema with regex column and custom undefined strategy
        [
            pa.DataFrameSchema(
                columns={
                    "[0-9]+": pa.Column(
                        int,
                        checks=[pa.Check(lambda s: True)],
                        regex=True,
                    )
                },
            ),
            "Column",
        ],
        [
            pa.DataFrameSchema(
                columns={"column": pa.Column(int)},
                checks=[
                    pa.Check(lambda s: s > -10000),
                    pa.Check(lambda s: s > -9999),
                ],
            ),
            "Dataframe",
        ],
    ],
)
@hypothesis.given(st.data())
def test_dataframe_strategy_undefined_check_strategy(
    schema: pa.DataFrameSchema, warning: str, data
) -> None:
    """Test case where dataframe check strategy is undefined."""
    strat = schema.strategy(size=5)
    with pytest.warns(
        UserWarning, match=f"{warning} check doesn't have a defined strategy"
    ):
        example = data.draw(strat)
    schema(example)


@pytest.mark.xfail(reason="https://github.com/unionai-oss/pandera/issues/1220")
@pytest.mark.parametrize("register_check", [True, False])
@hypothesis.given(st.data())
def test_defined_check_strategy(
    register_check: bool,
    data: st.DataObject,
):
    """
    Strategy specified for custom check is actually used when generating column
    examples.
    """

    def custom_strategy(pandera_dtype, strategy=None, *, min_val, max_val):
        """Custom strategy for range check."""
        if strategy is None:
            return st.floats(min_value=min_val, max_value=max_val).map(
                # the map isn't strictly necessary, but shows an example of
                # using the pandera_dtype argument
                strategies.to_numpy_dtype(pandera_dtype).type
            )
        return strategy.filter(lambda val: 0 <= val <= 10)

    if "custom_check_with_strategy" in pa.Check.REGISTERED_CUSTOM_CHECKS:
        del pa.Check.REGISTERED_CUSTOM_CHECKS["custom_check_with_strategy"]

    @pa.extensions.register_check_method(
        strategy=custom_strategy,
        statistics=["min_val", "max_val"],
    )
    def custom_check_with_strategy(pandas_obj, *, min_val, max_val):
        """Custom range check."""
        if isinstance(pandas_obj, pd.Series):
            return pandas_obj.between(min_val, max_val)
        return pandas_obj.applymap(lambda x: min_val <= x <= max_val)

    if register_check:
        check = Check.custom_check_with_strategy(0, 10)
    else:
        check = Check(
            custom_check_with_strategy,
            strategy=custom_strategy,
            min_val=0,
            max_val=10,
        )

    # test with column and dataframe schema
    col_schema = pa.Column(dtype="float64", checks=check, name="col_name")
    df_schema_df_level_check = pa.DataFrameSchema(
        columns={
            "col1": pa.Column(float),
            "col2": pa.Column(float),
            "col3": pa.Column(float),
        },
        checks=check,
    )
    df_schema_col_level_check = pa.DataFrameSchema(
        columns={"col1": col_schema}
    )

    for schema in (
        col_schema,
        df_schema_df_level_check,
        df_schema_col_level_check,
    ):
        size = data.draw(st.none() | st.integers(0, 3), label="size")
        with catch_warnings(record=True) as record:
            sample = data.draw(schema.strategy(size=size), label="s")  # type: ignore
            # We specifically test against warnings here, as they might indicate
            # the defined strategy isn't being used.
            # See https://github.com/unionai-oss/pandera/issues/1220
            assert len(record) == 0
        if size is not None:
            assert sample.shape[0] == size
        validated = schema.validate(sample)
        assert isinstance(validated, pd.DataFrame)


def test_unsatisfiable_checks():
    """Test that unsatisfiable checks raise an exception."""
    schema = pa.DataFrameSchema(
        columns={
            "col1": pa.Column(int, checks=[pa.Check.gt(0), pa.Check.lt(0)])
        }
    )
    for _ in range(5):
        with pytest.raises(hypothesis.errors.Unsatisfiable):
            schema.example(size=10)


class Schema(pa.DataFrameModel):
    """DataFrame model for strategy testing."""

    col1: pa.typing.Series[int]
    col2: pa.typing.Series[float]
    col3: pa.typing.Series[str]


@hypothesis.given(st.data())
def test_schema_model_strategy(data) -> None:
    """Test that strategy can be created from a DataFrameModel."""
    strat = Schema.strategy(size=10)
    sample_data = data.draw(strat)
    Schema.validate(sample_data)


@hypothesis.given(st.data())
def test_schema_model_strategy_df_check(data) -> None:
    """Test that schema with custom checks produce valid data."""

    class SchemaWithDFCheck(Schema):
        """Schema with a custom dataframe-level check with no strategy."""

        @pa.dataframe_check
        @classmethod
        def non_empty(cls, df: pd.DataFrame) -> bool:
            """Checks that dataframe is not empty."""
            return not df.empty

    strat = SchemaWithDFCheck.strategy(size=10)
    sample_data = data.draw(strat)
    Schema.validate(sample_data)


def test_schema_model_example() -> None:
    """Test that examples can be drawn from a DataFrameModel."""
    sample_data = Schema.example(size=10)
    Schema.validate(sample_data)  # type: ignore[arg-type]


def test_schema_component_with_no_pdtype() -> None:
    """
    Test that SchemaDefinitionError is raised if trying to create a strategy
    where pandas_dtype property is not specified.
    """
    for schema_component_strategy in [
        strategies.column_strategy,
        strategies.index_strategy,
    ]:
        with pytest.raises(pa.errors.SchemaDefinitionError):
            schema_component_strategy(pandera_dtype=None)  # type: ignore


@pytest.mark.parametrize(
    "check_arg", [pd.Timestamp("2006-01-01"), np.datetime64("2006-01-01")]
)
@hypothesis.given(st.data())
def test_datetime_example(check_arg, data) -> None:
    """Test Column schema example method generate examples of
    timezone-naive datetimes that pass."""

    for checks in [
        pa.Check.le(check_arg),
        pa.Check.ge(check_arg),
        pa.Check.eq(check_arg),
        pa.Check.isin([check_arg]),
    ]:
        column_schema = pa.Column(
            "datetime", checks=checks, name="test_datetime"
        )
        column_schema(data.draw(column_schema.strategy()))


@pytest.mark.parametrize(
    "dtype, check_arg",
    [
        [
            pd.DatetimeTZDtype(tz="UTC"),
            pd.Timestamp("2006-01-01", tz="UTC"),
        ],
        [
            pd.DatetimeTZDtype(tz="CET"),
            pd.Timestamp("2006-01-01", tz="CET"),
        ],
    ],
)
@hypothesis.given(st.data())
def test_datetime_tz_example(dtype, check_arg, data) -> None:
    """Test Column schema example method generate examples of
    timezone-aware datetimes that pass."""
    for checks in [
        pa.Check.le(check_arg),
        pa.Check.ge(check_arg),
        pa.Check.eq(check_arg),
        pa.Check.isin([check_arg]),
    ]:
        column_schema = pa.Column(
            dtype,
            checks=checks,
            name="test_datetime_tz",
        )
        synth_data = data.draw(column_schema.strategy())
        column_schema(synth_data)


@pytest.mark.parametrize(
    "dtype",
    (pd.Timedelta,),
)
@pytest.mark.parametrize(
    "check_arg",
    [
        # nanoseconds
        pd.Timedelta(int(1e9), unit="nanoseconds"),
        np.timedelta64(int(1e9), "ns"),
        # microseconds
        pd.Timedelta(int(1e6), unit="microseconds"),
        datetime.timedelta(microseconds=int(1e6)),
        # milliseconds
        pd.Timedelta(int(1e3), unit="milliseconds"),
        np.timedelta64(int(1e3), "ms"),
        datetime.timedelta(milliseconds=int(1e3)),
        # seconds
        pd.Timedelta(1, unit="s"),
        np.timedelta64(1, "s"),
        datetime.timedelta(seconds=1),
        # minutes
        pd.Timedelta(1, unit="m"),
        np.timedelta64(1, "m"),
        datetime.timedelta(minutes=1),
        # hours
        pd.Timedelta(1, unit="h"),
        np.timedelta64(1, "h"),
        datetime.timedelta(hours=1),
        # days
        pd.Timedelta(1, unit="day"),
        np.timedelta64(1, "D"),
        datetime.timedelta(days=1),
        # weeks
        pd.Timedelta(1, unit="W"),
        np.timedelta64(1, "W"),
        datetime.timedelta(weeks=1),
    ],
)
@hypothesis.given(st.data())
def test_timedelta(dtype, check_arg, data):
    """
    Test Column schema example method generate examples of timedeltas
    that pass tests.
    """
    for checks in [
        pa.Check.le(check_arg),
        pa.Check.ge(check_arg),
        pa.Check.eq(check_arg),
        pa.Check.isin([check_arg]),
        pa.Check.in_range(check_arg, check_arg + check_arg),
    ]:
        column_schema = pa.Column(
            dtype,
            checks=checks,
            name="test_datetime_tz",
        )
        column_schema(data.draw(column_schema.strategy()))


@pytest.mark.parametrize("dtype", [int, float, str])
@hypothesis.given(st.data())
def test_empty_nullable_schema(dtype, data):
    """Test that empty nullable schema strategy draws empty examples."""
    schema = pa.DataFrameSchema({"myval": pa.Column(dtype, nullable=True)})
    assert data.draw(schema.strategy(size=0)).empty
