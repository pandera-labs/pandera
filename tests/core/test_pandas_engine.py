"""Test pandas engine."""

import datetime as dt
from typing import Tuple, List, Optional, Any, Set

import hypothesis
import hypothesis.extra.pandas as pd_st
import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pyarrow
import pytest
import pytz
from hypothesis import given

from pandera import Field, DataFrameModel, errors
from pandera.engines import pandas_engine
from pandera.errors import ParserError, SchemaError

UNSUPPORTED_DTYPE_CLS: Set[Any] = set()

# `string[pyarrow]` gets parsed to type `string` by pandas
if pandas_engine.PYARROW_INSTALLED and pandas_engine.PANDAS_2_0_0_PLUS:
    UNSUPPORTED_DTYPE_CLS.add(pandas_engine.ArrowString)


@pytest.mark.parametrize(
    "data_type",
    [
        data_type
        for data_type in pandas_engine.Engine.get_registered_dtypes()
        if data_type not in UNSUPPORTED_DTYPE_CLS
    ],
)
def test_pandas_data_type(data_type):
    """Test numpy engine DataType base class."""
    if data_type.type is None:
        # don't test data types that require parameters e.g. Category
        return

    pandas_engine.Engine.dtype(data_type)
    pandas_engine.Engine.dtype(data_type.type)
    pandas_engine.Engine.dtype(
        getattr(data_type.type, "__name__", None)
        or getattr(data_type.type, "name", None)
        or data_type.type
    )

    with pytest.warns(UserWarning):
        pd_dtype = pandas_engine.DataType(data_type.type)
    with pytest.warns(UserWarning):
        pd_dtype_from_str = pandas_engine.DataType(str(data_type.type))
    assert pd_dtype == pd_dtype_from_str
    assert not pd_dtype.check("foo")


@pytest.mark.parametrize(
    "data_type_cls", list(pandas_engine.Engine.get_registered_dtypes())
)
def test_pandas_data_type_coerce(data_type_cls):
    """
    Test that pandas data type coercion will raise a ParserError. on failure.
    """
    try:
        data_type = data_type_cls()
    except TypeError:
        # don't test data types that require parameters
        return

    try:
        data_type.try_coerce(pd.Series(["1", "2", "a"]))
    except ParserError as exc:
        assert exc.failure_cases.shape[0] > 0


@pytest.mark.parametrize(
    "data_type_cls", list(pandas_engine.Engine.get_registered_dtypes())
)
def test_pandas_data_type_check(data_type_cls):
    """
    Test that pandas data type check results can be reduced.
    """
    try:
        data_type = data_type_cls()
    except TypeError:
        # don't test data types that require parameters
        return

    try:
        data_container = pd.Series([], dtype=data_type.type)
    except TypeError:
        # don't test complex data types, e.g. PythonDict, PythonTuple, etc
        return

    check_result = data_type.check(
        pandas_engine.Engine.dtype(data_container.dtype), data_container
    )
    assert isinstance(check_result, bool) or isinstance(
        check_result.all(), (bool, np.bool_)
    )


CATEGORIES = ["A", "B", "C"]


@given(st.lists(st.sampled_from(CATEGORIES), min_size=5))
def test_pandas_category_dtype(data):
    """Test pandas_engine.Category correctly coerces valid categorical data."""
    data = pd.Series(data)
    dtype = pandas_engine.Category(CATEGORIES)
    coerced_data = dtype.coerce(data)
    assert dtype.check(coerced_data.dtype)

    for _, value in data.items():
        coerced_value = dtype.coerce_value(value)
        assert coerced_value in CATEGORIES


@given(st.lists(st.sampled_from(["X", "Y", "Z"]), min_size=5))
def test_pandas_category_dtype_error(data):
    """Test pandas_engine.Category raises TypeErrors on invalid data."""
    data = pd.Series(data)
    dtype = pandas_engine.Category(CATEGORIES)

    with pytest.raises(TypeError):
        dtype.coerce(data)

    for _, value in data.items():
        with pytest.raises(TypeError):
            dtype.coerce_value(value)


@given(st.lists(st.sampled_from([1, 0, 1.0, 0.0, True, False]), min_size=5))
def test_pandas_boolean_native_type(data):
    """Test native pandas bool type correctly coerces valid bool-like data."""
    data = pd.Series(data)
    dtype = pandas_engine.Engine.dtype("boolean")

    # the BooleanDtype can't handle Series of non-boolean, mixed dtypes
    if data.dtype == "object":
        with pytest.raises(TypeError):
            dtype.coerce(data)
    else:
        coerced_data = dtype.coerce(data)
        assert dtype.check(coerced_data.dtype)

    for _, value in data.items():
        dtype.coerce_value(value)


@given(st.lists(st.sampled_from(["A", "True", "False", 5, -1]), min_size=5))
def test_pandas_boolean_native_type_error(data):
    """Test native pandas bool type raises TypeErrors on non-bool-like data."""
    data = pd.Series(data)
    dtype = pandas_engine.Engine.dtype("boolean")

    with pytest.raises(TypeError):
        dtype.coerce(data)

    for _, value in data.items():
        with pytest.raises(TypeError):
            dtype.coerce_value(value)


@hypothesis.settings(max_examples=1000)
@pytest.mark.parametrize("timezone_aware", [True, False])
@given(
    data=pd_st.series(
        dtype="datetime64[ns]",
        index=pd_st.range_indexes(min_size=5, max_size=10),
    ),
    timezone=st.sampled_from(pytz.all_timezones),
)
def test_pandas_datetimetz_dtype(timezone_aware, data, timezone):
    """
    Test that pandas timezone-aware datetime correctly handles timezone-aware
    and non-timezone-aware data.
    """
    timezone = pytz.timezone(timezone)
    tz_localize_kwargs = {"ambiguous": "NaT"}

    expected_failure = False
    if timezone_aware:
        data = data.dt.tz_localize(pytz.utc)
    else:
        assert data.dt.tz is None
        try:
            data.dt.tz_localize(timezone, **tz_localize_kwargs)
        except pytz.exceptions.NonExistentTimeError:
            expected_failure = True

    # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
    dtype = pandas_engine.Engine.dtype(
        pandas_engine.DateTime(
            tz=timezone, tz_localize_kwargs=tz_localize_kwargs
        )
    )
    if expected_failure:
        with pytest.raises(pytz.exceptions.NonExistentTimeError):
            dtype.coerce(data)
    else:
        coerced_data = dtype.coerce(data)
        assert coerced_data.dt.tz == timezone


def generate_test_cases_time_zone_agnostic() -> List[
    Tuple[
        List[dt.datetime],
        Optional[dt.tzinfo],
        bool,
        List[dt.datetime],
        bool,
    ]
]:
    """
    Generate test parameter combinations for a given list of datetime lists.

    Returns:
        List of tuples:
        - List of input datetimes
        - tz for DateTime constructor
        - coerce flag for Field constructor
        - expected output datetimes
        - raises flag (True if an exception is expected, False otherwise)
    """
    datetimes = [
        # multi tz and tz naive
        [
            pytz.timezone("America/New_York").localize(
                dt.datetime(2023, 3, 1, 4)
            ),
            pytz.timezone("America/Los_Angeles").localize(
                dt.datetime(2023, 3, 1, 5)
            ),
            dt.datetime(2023, 3, 1, 5),  # naive datetime
        ],
        # multi tz
        [
            pytz.timezone("America/New_York").localize(
                dt.datetime(2023, 3, 1, 4)
            ),
            pytz.timezone("America/Los_Angeles").localize(
                dt.datetime(2023, 3, 1, 5)
            ),
        ],
        # tz naive
        [dt.datetime(2023, 3, 1, 4), dt.datetime(2023, 3, 1, 5)],
        # single tz
        [
            pytz.timezone("America/New_York").localize(
                dt.datetime(2023, 3, 1, 4)
            ),
            pytz.timezone("America/New_York").localize(
                dt.datetime(2023, 3, 1, 5)
            ),
        ],
    ]

    test_cases = []

    for datetime_list in datetimes:
        for coerce in [True, False]:
            for tz in [
                None,
                pytz.timezone("America/Chicago"),
                pytz.FixedOffset(120),  # 120 minutes = 2 hours offset
            ]:
                # Determine if the test should raise an exception
                # Should raise error when:
                # * coerce is False but there is a timezone-naive datetime
                # * coerce is True but tz is not set
                has_naive_datetime = any(
                    dt.tzinfo is None for dt in datetime_list
                )
                raises = (not coerce and has_naive_datetime) or (
                    coerce and tz is None
                )

                # Generate expected output
                if raises:
                    expected_output = None  # No expected output since an exception will be raised
                else:
                    if coerce:
                        # Replace naive datetimes with localized ones
                        expected_output_naive = [
                            tz.localize(dtime) if tz is not None else dtime
                            for dtime in datetime_list
                            if dtime.tzinfo is None
                        ]

                        # Convert timezone-aware datetimes to the desired timezone
                        expected_output_aware = [
                            dtime.astimezone(
                                tz
                            )  # Use .astimezone() for aware datetimes
                            for dtime in datetime_list
                            if dtime.tzinfo is not None
                        ]
                        expected_output = (
                            expected_output_naive + expected_output_aware
                        )
                    else:
                        # ignore tz
                        expected_output = datetime_list

                test_case = (
                    datetime_list,
                    tz,
                    coerce,
                    expected_output,
                    raises,
                )
                test_cases.append(test_case)

    # define final test cases with improper type
    datetime_list = [
        pytz.timezone("America/New_York").localize(
            dt.datetime(
                2023,
                3,
                1,
                4,
            )
        ),
        "hello world",
    ]
    tz = None
    expected_output = None
    raises = True

    bad_type_coerce = (datetime_list, tz, True, expected_output, raises)
    bad_type_no_coerce = (datetime_list, tz, False, expected_output, raises)
    test_cases.extend([bad_type_coerce, bad_type_no_coerce])  # type: ignore

    return test_cases  # type: ignore


@pytest.mark.parametrize(
    "examples, tz, coerce, expected_output, raises",
    generate_test_cases_time_zone_agnostic(),
)
def test_dt_time_zone_agnostic(examples, tz, coerce, expected_output, raises):
    """Test that time_zone_agnostic works as expected"""

    # Testing using a pandera DataFrameModel rather than directly calling dtype coerce or validate because with
    # time_zone_agnostic, dtype is set dynamically based on the input data
    class SimpleSchema(DataFrameModel):
        # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        datetime_column: pandas_engine.DateTime(
            time_zone_agnostic=True, tz=tz
        ) = Field(coerce=coerce)

    data = pd.DataFrame({"datetime_column": examples})

    if raises:
        with pytest.raises((SchemaError, errors.ParserError)):
            SimpleSchema.validate(data)
    else:
        validated_df = SimpleSchema.validate(data)
        assert sorted(validated_df["datetime_column"].tolist()) == sorted(
            expected_output
        )


@hypothesis.settings(max_examples=1000)
@pytest.mark.parametrize("to_df", [True, False])
@given(
    data=pd_st.series(
        dtype="datetime64[ns]",
        index=pd_st.range_indexes(min_size=5, max_size=10),
    )
)
def test_pandas_date_coerce_dtype(to_df, data):
    """Test that pandas Date dtype coerces to datetime.date object."""

    data = data.to_frame() if to_df else data

    dtype = pandas_engine.Engine.dtype(pandas_engine.Date())
    coerced_data = dtype.coerce(data)

    if to_df:
        assert (coerced_data.dtypes == "object").all() or (
            coerced_data.isna().all(axis=None)
            and (coerced_data.dtypes == "datetime64[ns]").all()
        )

        assert (
            coerced_data.applymap(lambda x: isinstance(x, dt.date))
            | coerced_data.isna()
        ).all(axis=None)
        return

    assert (coerced_data.dtype == "object") or (
        coerced_data.isna().all() and coerced_data.dtype == "datetime64[ns]"
    )
    assert (
        coerced_data.map(lambda x: isinstance(x, dt.date))
        | coerced_data.isna()
    ).all()


pandas_arrow_dtype_cases = (
    (pd.Series([["a", "b", "c"]]), pyarrow.list_(pyarrow.string())),
    (pd.Series([["a", "b"]]), pyarrow.list_(pyarrow.string(), 2)),
    (
        pd.Series([{"foo": 1, "bar": "a"}]),
        pyarrow.struct([("foo", pyarrow.int64()), ("bar", pyarrow.string())]),
    ),
    (pd.Series([None, pd.NA, np.nan]), pyarrow.null),
    (pd.Series([None, dt.date(1970, 1, 1)]), pyarrow.date32),
    (pd.Series([None, dt.date(1970, 1, 1)]), pyarrow.date64),
    (pd.Series([1, 2]), pyarrow.duration("ns")),
    (pd.Series([1, 1e3, 1e6, 1e9, None]), pyarrow.time32("ms")),
    (pd.Series([1, 1e3, 1e6, 1e9, None]), pyarrow.time64("ns")),
    (
        pd.Series(
            [
                [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
                [{"key": "c", "value": 3}],
            ]
        ),
        pyarrow.map_(pyarrow.string(), pyarrow.int32()),
    ),
    (pd.Series(["foo", "barbaz", None]), pyarrow.binary()),
    (pd.Series(["foo", "bar", "baz", None]), pyarrow.binary(3)),
    (pd.Series(["foo", "barbaz", None]), pyarrow.large_binary()),
    (pd.Series(["1", "1.0", "foo", "bar", None]), pyarrow.large_string()),
    (
        pd.Series(["a", "b", "c"]),
        pyarrow.dictionary(pyarrow.int64(), pyarrow.string()),
    ),
)


@pytest.mark.parametrize(("data", "dtype"), pandas_arrow_dtype_cases)
def test_pandas_arrow_dtype(data, dtype):
    """Test pyarrow dtype."""
    if not (
        pandas_engine.PYARROW_INSTALLED and pandas_engine.PANDAS_2_0_0_PLUS
    ):
        pytest.skip("Support of pandas 2.0.0+ with pyarrow only")
    dtype = pandas_engine.Engine.dtype(dtype)

    coerced_data = dtype.coerce(data)
    assert coerced_data.dtype == dtype.type


pandas_arrow_dtype_error_cases = (
    (pd.Series([["a", "b", "c"]]), pyarrow.list_(pyarrow.int64())),
    (pd.Series([["a", "b"]]), pyarrow.list_(pyarrow.string(), 3)),
    (
        pd.Series([{"foo": 1, "bar": "a"}]),
        pyarrow.struct([("foo", pyarrow.string()), ("bar", pyarrow.int64())]),
    ),
    (pd.Series(["a", "1"]), pyarrow.null),
    (pd.Series(["a", dt.date(1970, 1, 1), "1970-01-01"]), pyarrow.date32),
    (pd.Series(["a", dt.date(1970, 1, 1), "1970-01-01"]), pyarrow.date64),
    (pd.Series(["a"]), pyarrow.duration("ns")),
    (pd.Series(["a", "b"]), pyarrow.time32("ms")),
    (pd.Series(["a", "b"]), pyarrow.time64("ns")),
    (
        pd.Series(
            [
                [{"key": "a", "value": 1}, {"key": "b", "value": 2}],
                [{"key": "c", "value": 3}],
            ]
        ),
        pyarrow.map_(pyarrow.int32(), pyarrow.string()),
    ),
    (pd.Series([1, "foo", None]), pyarrow.binary()),
    (pd.Series(["foo", "bar", "baz", None]), pyarrow.binary(2)),
    (pd.Series([1, "foo", "barbaz", None]), pyarrow.large_binary()),
    (pd.Series([1, 1.0, "foo", "bar", None]), pyarrow.large_string()),
    (
        pd.Series([1.0, 2.0, 3.0]),
        pyarrow.dictionary(pyarrow.int64(), pyarrow.float64()),
    ),
    (
        pd.Series(["a", "b", "c"]),
        pyarrow.dictionary(pyarrow.int64(), pyarrow.int64()),
    ),
)


@pytest.mark.parametrize(("data", "dtype"), pandas_arrow_dtype_error_cases)
def test_pandas_arrow_dtype_error(data, dtype):
    """Test pyarrow dtype raises Error on bad data."""
    if not (
        pandas_engine.PYARROW_INSTALLED and pandas_engine.PANDAS_2_0_0_PLUS
    ):
        pytest.skip("Support of pandas 2.0.0+ with pyarrow only")
    dtype = pandas_engine.Engine.dtype(dtype)

    with pytest.raises(
        (
            pyarrow.ArrowInvalid,
            pyarrow.ArrowTypeError,
            NotImplementedError,
            ValueError,
            AssertionError,
        )
    ):
        coerced_data = dtype.coerce(data)
        assert coerced_data.dtype == dtype.type
