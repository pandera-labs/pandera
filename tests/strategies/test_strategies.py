# pylint: disable=undefined-variable,redefined-outer-name,invalid-name,undefined-loop-variable  # noqa
"""Unit tests for pandera data generating strategies."""
import operator
import re
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import pandera as pa
import pandera.strategies as strategies
from pandera.checks import _CheckBase, register_check_statistics
from pandera.dtypes_ import is_category, is_complex, is_float
from pandera.engines import pandas_engine

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


SUPPORTED_DTYPES = set()
for data_type in pandas_engine.Engine.get_registered_dtypes():
    if (
        # valid hypothesis.strategies.floats <=64
        getattr(data_type, "bit_width", -1) > 64
        or is_category(data_type)
        or data_type
        in (pandas_engine.Interval, pandas_engine.Period, pandas_engine.Sparse)
    ):
        continue
    SUPPORTED_DTYPES.add(pandas_engine.Engine.dtype(data_type))

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

NUMERIC_RANGE_CONSTANT = 10
DATE_RANGE_CONSTANT = np.timedelta64(NUMERIC_RANGE_CONSTANT, "D")
COMPLEX_RANGE_CONSTANT = np.complex64(
    complex(NUMERIC_RANGE_CONSTANT, NUMERIC_RANGE_CONSTANT)
)


@pytest.mark.parametrize("data_type", [pa.Category])
def test_unsupported_pandas_dtype_strategy(data_type):
    """Test unsupported pandas dtype strategy raises error."""
    with pytest.raises(
        TypeError,
        match="data generation for the Category dtype is currently unsupported",
    ):
        strategies.pandas_dtype_strategy(data_type)


@pytest.mark.parametrize("data_type", SUPPORTED_DTYPES)
@hypothesis.given(st.data())
def test_pandas_dtype_strategy(data_type, data):
    """Test that series can be constructed from pandas dtype."""

    strategy = strategies.pandas_dtype_strategy(data_type)
    example = data.draw(strategy)

    expected_type = strategies.to_numpy_dtype(data_type).type
    assert example.dtype.type == expected_type

    chained_strategy = strategies.pandas_dtype_strategy(data_type, strategy)
    chained_example = data.draw(chained_strategy)
    assert chained_example.dtype.type == expected_type


@pytest.mark.parametrize("data_type", NUMERIC_DTYPES)
@hypothesis.given(st.data())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
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
    """Strategy to generate value range based on PandasDtype"""
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
        .filter(lambda x: x[0] < x[1])
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
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
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
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
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
            **base_st_kwargs,
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
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
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
def test_str_pattern_checks(str_strat, pattern_fn, chained, data, pattern):
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
        .map(sorted)
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
def test_register_check_strategy(data):
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
    class CustomCheck(_CheckBase):
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
    result = data.draw(check.strategy(pa.Int()))
    assert result == 100


def test_register_check_strategy_exception():
    """Check method needs statistics attr to register a strategy."""

    def custom_strat():
        pass

    class CustomCheck(_CheckBase):
        """Custom check class."""

        @classmethod
        @strategies.register_check_strategy(custom_strat)
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

    with pytest.raises(
        AttributeError,
        match="check object doesn't have a defined statistics property",
    ):
        CustomCheck.custom_check()


@hypothesis.given(st.data())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_series_strategy(data):
    """Test SeriesSchema strategy."""
    series_schema = pa.SeriesSchema(pa.Int(), pa.Check.gt(0))
    series_schema(data.draw(series_schema.strategy()))


def test_series_example():
    """Test SeriesSchema example method generate examples that pass."""
    series_schema = pa.SeriesSchema(pa.Int(), pa.Check.gt(0))
    for _ in range(10):
        series_schema(series_schema.example())


@hypothesis.given(st.data())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_column_strategy(data):
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
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
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
    # with pytest.raises(pa.errors.BaseStrategyOnlyError):
    #     strategies.dataframe_strategy(
    #         data_type, strategies.pandas_dtype_strategy(data_type)
    #     )


def test_dataframe_example():
    """Test DataFrameSchema example method generate examples that pass."""
    schema = pa.DataFrameSchema(
        {"column": pa.Column(pa.Int(), pa.Check.gt(0))}
    )
    for _ in range(10):
        schema(schema.example())


@pytest.mark.parametrize(
    "regex",
    [
        "col_[0-9]{1,4}",
        "[a-zA-Z]+_foobar",
        "[a-z]+_[0-9]+_[a-z]+",
    ],
)
@hypothesis.given(st.data(), st.integers(min_value=-5, max_value=5))
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_dataframe_with_regex(regex, data, n_regex_columns):
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
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
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
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_dataframe_strategy_with_indexes(data_type, data):
    """Test dataframe strategy with index and multiindex components."""
    dataframe_schema_index = pa.DataFrameSchema(index=pa.Index(data_type))
    dataframe_schema_multiindex = pa.DataFrameSchema(
        index=pa.MultiIndex(
            [pa.Index(data_type, name=f"index{i}") for i in range(3)]
        )
    )

    dataframe_schema_index(data.draw(dataframe_schema_index.strategy(size=10)))
    dataframe_schema_multiindex(
        data.draw(dataframe_schema_multiindex.strategy(size=10))
    )


@hypothesis.given(st.data())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_index_strategy(data):
    """Test Index schema component strategy."""
    data_type = pa.Int()
    index_schema = pa.Index(data_type, allow_duplicates=False, name="index")
    strat = index_schema.strategy(size=10)
    example = data.draw(strat)

    assert (~example.duplicated()).all()
    actual_data_type = pandas_engine.Engine.dtype(example.dtype)
    assert data_type.check(actual_data_type)
    index_schema(pd.DataFrame(index=example))


def test_index_example():
    """
    Test Index schema component example method generates examples that pass.
    """
    data_type = pa.Int()
    index_schema = pa.Index(data_type, allow_duplicates=False)
    for _ in range(10):
        index_schema(pd.DataFrame(index=index_schema.example()))


@hypothesis.given(st.data())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_multiindex_strategy(data):
    """Test MultiIndex schema component strategy."""
    data_type = pa.Float()
    multiindex = pa.MultiIndex(
        indexes=[
            pa.Index(data_type, allow_duplicates=False, name="level_0"),
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


def test_multiindex_example():
    """
    Test MultiIndex schema component example method generates examples that
    pass.
    """
    data_type = pa.Float()
    multiindex = pa.MultiIndex(
        indexes=[
            pa.Index(data_type, allow_duplicates=False, name="level_0"),
            pa.Index(data_type, nullable=True),
            pa.Index(data_type),
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
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_check_nullable_field_strategy(
    data_type, field_strategy, nullable, data
):
    """Test strategies for generating nullable column/index data."""
    size = 5
    strat = field_strategy(data_type, nullable=nullable, size=size)
    example = data.draw(strat)

    if nullable:
        assert example.isna().any()
    else:
        assert example.notna().all()


@pytest.mark.parametrize("data_type", NULLABLE_DTYPES)
@pytest.mark.parametrize("nullable", [True, False])
@hypothesis.given(st.data())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
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
        assert example.isna().any(axis=None)
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
@hypothesis.settings(
    suppress_health_check=[
        hypothesis.HealthCheck.filter_too_much,
        hypothesis.HealthCheck.too_slow,
    ],
)
@hypothesis.given(st.data())
def test_series_strategy_undefined_check_strategy(schema, warning, data):
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
                columns={"column": pa.Column(pa.Int())},
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
                        pa.Int(),
                        checks=[
                            pa.Check(lambda s: s > -10000),
                            pa.Check(lambda s: s > -9999),
                        ],
                    )
                },
            ),
            "Column",
        ],
        [
            pa.DataFrameSchema(
                columns={"column": pa.Column(pa.Int())},
                checks=[
                    pa.Check(lambda s: s > -10000),
                    pa.Check(lambda s: s > -9999),
                ],
            ),
            "Dataframe",
        ],
    ],
)
@hypothesis.settings(
    suppress_health_check=[
        hypothesis.HealthCheck.filter_too_much,
        hypothesis.HealthCheck.too_slow,
    ],
)
@hypothesis.given(st.data())
def test_dataframe_strategy_undefined_check_strategy(schema, warning, data):
    """Test case where dataframe check strategy is undefined."""
    strat = schema.strategy(size=5)
    with pytest.warns(
        UserWarning, match=f"{warning} check doesn't have a defined strategy"
    ):
        example = data.draw(strat)
    schema(example)


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


@pytest.fixture(scope="module")
def schema_model():
    """Schema model fixture."""

    class Schema(pa.SchemaModel):
        """Schema model for strategy testing."""

        col1: pa.typing.Series[int]
        col2: pa.typing.Series[float]
        col3: pa.typing.Series[str]

    return Schema


@hypothesis.given(st.data())
@hypothesis.settings(
    suppress_health_check=[hypothesis.HealthCheck.too_slow],
)
def test_schema_model_strategy(schema_model, data):
    """Test that strategy can be created from a SchemaModel."""
    strat = schema_model.strategy(size=10)
    sample_data = data.draw(strat)
    schema_model.validate(sample_data)


def test_schema_model_example(schema_model):
    """Test that examples can be drawn from a SchemaModel."""
    sample_data = schema_model.example(size=10)
    schema_model.validate(sample_data)


def test_schema_component_with_no_pdtype():
    """
    Test that SchemaDefinitionError is raised if trying to create a strategy
    where pandas_dtype property is not specified.
    """
    for schema_component_strategy in [
        strategies.column_strategy,
        strategies.index_strategy,
    ]:
        with pytest.raises(pa.errors.SchemaDefinitionError):
            schema_component_strategy(pandera_dtype=None)
