"""Built-in checks for pandas."""

import sys
import operator
import re
from typing import Any, Iterable, Optional, TypeVar, Union, cast

import pandas as pd

import pandera.strategies as st
from pandera.api.extensions import register_builtin_check


MODIN_IMPORTED = "modin" in sys.modules
PYSPARK_IMPORTED = "pyspark" in sys.modules


# TODO: create a separate module for each framework: dask, modin, pyspark
# so checks are registered for the correct framework.
if MODIN_IMPORTED and not PYSPARK_IMPORTED:  # pragma: no cover
    import modin.pandas as mpd

    PandasData = Union[pd.Series, pd.DataFrame, mpd.Series, mpd.DataFrame]
elif not MODIN_IMPORTED and PYSPARK_IMPORTED:  # pragma: no cover
    import pyspark.pandas as ppd

    PandasData = Union[pd.Series, pd.DataFrame, ppd.Series, ppd.DataFrame]  # type: ignore[misc]
elif MODIN_IMPORTED and PYSPARK_IMPORTED:  # pragma: no cover
    import modin.pandas as mpd
    import pyspark.pandas as ppd

    PandasData = Union[  # type: ignore[misc]
        pd.Series,
        pd.DataFrame,
        mpd.Series,
        mpd.DataFrame,
        ppd.Series,
        ppd.DataFrame,
    ]
else:  # pragma: no cover
    PandasData = Union[pd.Series, pd.DataFrame]  # type: ignore[misc]


T = TypeVar("T")


# TODO: remove aliases, it's not needed anymore since aliases are defined in the
# Check class.
@register_builtin_check(
    aliases=["eq"],
    strategy=st.eq_strategy,
    error="equal_to({value})",
)
def equal_to(data: PandasData, value: Any) -> PandasData:
    """Ensure all elements of a data container equal a certain value.

    :param value: values in this pandas data structure must be
        equal to this value.
    """
    return data == value


@register_builtin_check(
    aliases=["ne"],
    strategy=st.ne_strategy,
    error="not_equal_to({value})",
)
def not_equal_to(data: PandasData, value: Any) -> PandasData:
    """Ensure no elements of a data container equals a certain value.

    :param value: This value must not occur in the checked
        :class:`pandas.Series`.
    """
    return data != value


@register_builtin_check(
    aliases=["gt"],
    strategy=st.gt_strategy,
    error="greater_than({min_value})",
)
def greater_than(data: PandasData, min_value: Any) -> PandasData:
    """
    Ensure values of a data container are strictly greater than a minimum
    value.

    :param min_value: Lower bound to be exceeded. Must be a type comparable
        to the dtype of the :class:`pandas.Series` to be validated (e.g. a
        numerical type for float or int and a datetime for datetime).
    """
    return data > min_value


@register_builtin_check(
    aliases=["ge"],
    strategy=st.ge_strategy,
    error="greater_than_or_equal_to({min_value})",
)
def greater_than_or_equal_to(data: PandasData, min_value: Any) -> PandasData:
    """Ensure all values are greater or equal a certain value.

    :param min_value: Allowed minimum value for values of a series. Must be
        a type comparable to the dtype of the :class:`pandas.Series` to be
        validated.
    """
    return data >= min_value


@register_builtin_check(
    aliases=["lt"],
    strategy=st.lt_strategy,
    error="less_than({max_value})",
)
def less_than(data: PandasData, max_value: Any) -> PandasData:
    """Ensure values of a series are strictly below a maximum value.

    :param max_value: All elements of a series must be strictly smaller
        than this. Must be a type comparable to the dtype of the
        :class:`pandas.Series` to be validated.
    """
    if max_value is None:
        raise ValueError("max_value must not be None")
    return data < max_value


@register_builtin_check(
    aliases=["le"],
    strategy=st.le_strategy,
    error="less_than_or_equal_to({max_value})",
)
def less_than_or_equal_to(data: PandasData, max_value: Any) -> PandasData:
    """Ensure values of a series are strictly below a maximum value.

    :param max_value: Upper bound not to be exceeded. Must be a type
        comparable to the dtype of the :class:`pandas.Series` to be
        validated.
    """
    if max_value is None:
        raise ValueError("max_value must not be None")
    return data <= max_value


@register_builtin_check(
    aliases=["between"],
    strategy=st.in_range_strategy,
    error="in_range({min_value}, {max_value})",
)
def in_range(
    data: PandasData,
    min_value: T,
    max_value: T,
    include_min: bool = True,
    include_max: bool = True,
):
    """Ensure all values of a series are within an interval.

    Both endpoints must be a type comparable to the dtype of the
    :class:`pandas.Series` to be validated.

    :param min_value: Left / lower endpoint of the interval.
    :param max_value: Right / upper endpoint of the interval. Must not be
        smaller than min_value.
    :param include_min: Defines whether min_value is also an allowed value
        (the default) or whether all values must be strictly greater than
        min_value.
    :param include_max: Defines whether min_value is also an allowed value
        (the default) or whether all values must be strictly smaller than
        max_value.
    """
    # Using functions from operator module to keep conditions out of the
    # closure
    left_op = operator.le if include_min else operator.lt
    right_op = operator.ge if include_max else operator.gt
    return left_op(min_value, data) & right_op(max_value, data)  # type: ignore


@register_builtin_check(
    strategy=st.isin_strategy,
    error="isin({allowed_values})",
)
def isin(data: PandasData, allowed_values: Iterable) -> PandasData:
    """Ensure only allowed values occur within a series.

    This checks whether all elements of a :class:`pandas.Series`
    are part of the set of elements of allowed values. If allowed
    values is a string, the set of elements consists of all distinct
    characters of the string. Thus only single characters which occur
    in allowed_values at least once can meet this condition. If you
    want to check for substrings use :meth:`Check.str_contains`.

    :param allowed_values: The set of allowed values. May be any iterable.
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.isin(allowed_values)


@register_builtin_check(
    strategy=st.notin_strategy,
    error="notin({forbidden_values})",
)
def notin(data: PandasData, forbidden_values: Iterable) -> PandasData:
    """Ensure some defined values don't occur within a series.

    Like :meth:`Check.isin` this check operates on single characters if
    it is applied on strings. If forbidden_values is a string, it is understood
    as set of prohibited characters. Any string of length > 1 can't be in it by
    design.

    :param forbidden_values: The set of values which should not occur. May
        be any iterable.
    :param raise_warning: if True, check raises SchemaWarning instead of
        SchemaError on validation.
    """
    return ~data.isin(forbidden_values)


@register_builtin_check(
    strategy=st.str_matches_strategy,
    error="str_matches('{pattern}')",
)
def str_matches(
    data: PandasData,
    pattern: Union[str, re.Pattern],
) -> PandasData:
    """Ensure that string values match a regular expression.

    :param pattern: Regular expression pattern to use for matching
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.match(cast(str, pattern), na=False)


@register_builtin_check(
    strategy=st.str_contains_strategy,
    error="str_contains('{pattern}')",
)
def str_contains(
    data: PandasData,
    pattern: Union[str, re.Pattern],
) -> PandasData:
    """Ensure that a pattern can be found within each row.

    :param pattern: Regular expression pattern to use for searching
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.contains(cast(str, pattern), na=False)


@register_builtin_check(
    strategy=st.str_startswith_strategy,
    error="str_startswith('{string}')",
)
def str_startswith(data: PandasData, string: str) -> PandasData:
    """Ensure that all values start with a certain string.

    :param string: String all values should start with
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.startswith(string, na=False)


@register_builtin_check(
    strategy=st.str_endswith_strategy, error="str_endswith('{string}')"
)
def str_endswith(data: PandasData, string: str) -> PandasData:
    """Ensure that all values end with a certain string.

    :param string: String all values should end with
    :param kwargs: key-word arguments passed into the `Check` initializer.
    """
    return data.str.endswith(string, na=False)


@register_builtin_check(
    strategy=st.str_length_strategy,
    error="str_length({min_value}, {max_value})",
)
def str_length(
    data: PandasData,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> PandasData:
    """Ensure that the length of strings is within a specified range.

    :param min_value: Minimum length of strings. No minimum by default.
    :param max_value: Maximum length of strings. No maximum by default.
    """
    str_len = data.str.len()
    if min_value is None and max_value is None:
        raise ValueError(
            "At least a minimum or a maximum need to be specified. Got "
            "None."
        )
    if max_value is None:
        return str_len >= min_value  # type: ignore[operator]
    elif min_value is None:
        return str_len <= max_value
    return (str_len <= max_value) & (str_len >= min_value)


@register_builtin_check(
    error="unique_values_eq({values})",
)
def unique_values_eq(data: PandasData, values: Iterable):
    """Ensure that unique values in the data object contain all values.

    .. note::
        In contrast with :func:`isin`, this check makes sure that all the items
        in the ``values`` iterable are contained within the series.

    :param values: The set of values that must be present. Maybe any iterable.
    """
    return set(data.unique()) == values  # type: ignore[return-value,operator]
