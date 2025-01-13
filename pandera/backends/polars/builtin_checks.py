"""Built-in checks for polars."""

import re
from typing import Any, Iterable, Optional, TypeVar, Union

import polars as pl

from pandera.api.extensions import register_builtin_check
from pandera.api.polars.types import PolarsData

T = TypeVar("T")


@register_builtin_check(
    aliases=["eq"],
    error="equal_to({value})",
)
def equal_to(data: PolarsData, value: Any) -> pl.LazyFrame:
    """Ensure all elements of a data container equal a certain value.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param value: Values in this Polars data structure must be
        equal to this value.
    """
    return data.lazyframe.select(pl.col(data.key).eq(value))


@register_builtin_check(
    aliases=["ne"],
    error="not_equal_to({value})",
)
def not_equal_to(data: PolarsData, value: Any) -> pl.LazyFrame:
    """Ensure no elements of a data container equals a certain value.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param value: This value must not occur in the checked data structure.
    """
    return data.lazyframe.select(pl.col(data.key).ne(value))


@register_builtin_check(
    aliases=["gt"],
    error="greater_than({min_value})",
)
def greater_than(data: PolarsData, min_value: Any) -> pl.LazyFrame:
    """
    Ensure values of a data container are strictly greater than a minimum
    value.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param min_value: Lower bound to be exceeded. Must be
        a type comparable to the dtype of the series datatype of Polars.
    """
    return data.lazyframe.select(pl.col(data.key).gt(min_value))


@register_builtin_check(
    aliases=["ge"],
    error="greater_than_or_equal_to({min_value})",
)
def greater_than_or_equal_to(data: PolarsData, min_value: Any) -> pl.LazyFrame:
    """Ensure all values are greater or equal a certain value.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param min_value: Allowed minimum value for values of a series. Must be
        a type comparable to the dtype of the series datatype of Polars.
    """
    return data.lazyframe.select(pl.col(data.key).ge(min_value))


@register_builtin_check(
    aliases=["lt"],
    error="less_than({max_value})",
)
def less_than(data: PolarsData, max_value: Any) -> pl.LazyFrame:
    """Ensure values of a series are strictly below a maximum value.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param max_value: All elements of a series must be strictly smaller
        than this. Must be a type comparable to the dtype of the series datatype of Polars.
    """
    return data.lazyframe.select(pl.col(data.key).lt(max_value))


@register_builtin_check(
    aliases=["le"],
    error="less_than_or_equal_to({max_value})",
)
def less_than_or_equal_to(data: PolarsData, max_value: Any) -> pl.LazyFrame:
    """Ensure values of a series are strictly below a maximum value.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param max_value: Upper bound not to be exceeded. Must be a type comparable to the dtype of the
    series datatype of Polars
    """
    return data.lazyframe.select(pl.col(data.key).le(max_value))


@register_builtin_check(
    aliases=["between"],
    error="in_range({min_value}, {max_value})",
)
def in_range(
    data: PolarsData,
    min_value: T,
    max_value: T,
    include_min: bool = True,
    include_max: bool = True,
) -> pl.LazyFrame:
    """Ensure all values of a series are within an interval.

    Both endpoints must be a type comparable to the dtype of the
    series datatype of Polars

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
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
    col = pl.col(data.key)
    is_in_min = col.ge(min_value) if include_min else col.gt(min_value)
    is_in_max = col.le(max_value) if include_max else col.lt(max_value)

    return data.lazyframe.select(is_in_min.and_(is_in_max))


@register_builtin_check(
    error="isin({allowed_values})",
)
def isin(data: PolarsData, allowed_values: Iterable) -> pl.LazyFrame:
    """Ensure only allowed values occur within a series.

    This checks whether all elements of a :class:`polars.Series`
    are part of the set of elements of allowed values. If allowed
    values is a string, the set of elements consists of all distinct
    characters of the string. Thus only single characters which occur
    in allowed_values at least once can meet this condition. If you
    want to check for substrings use :meth:`Check.str_contains`.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param allowed_values: The set of allowed values. May be any iterable.
    """
    return data.lazyframe.select(pl.col(data.key).is_in(allowed_values))


@register_builtin_check(
    error="notin({forbidden_values})",
)
def notin(data: PolarsData, forbidden_values: Iterable) -> pl.LazyFrame:
    """Ensure some defined values don't occur within a series.

    Like :meth:`Check.isin` this check operates on single characters if
    it is applied on strings. If forbidden_values is a string, it is understood
    as set of prohibited characters. Any string of length > 1 can't be in it by
    design.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The keys
        to access the dataframe is "dataframe" and column name using "key".
    :param forbidden_values: The set of values which should not occur. May
        be any iterable.
    """
    return data.lazyframe.select(
        pl.col(data.key).is_in(forbidden_values).not_()
    )


@register_builtin_check(
    error="str_matches('{pattern}')",
)
def str_matches(
    data: PolarsData,
    pattern: Union[str, re.Pattern],
) -> pl.LazyFrame:
    """Ensure that string starts with a match of a regular expression pattern.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param pattern: Regular expression pattern to use for matching.
    """
    pattern = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
    if not pattern.startswith("^"):
        pattern = f"^{pattern}"
    return data.lazyframe.select(
        pl.col(data.key).str.contains(pattern=pattern)
    )


@register_builtin_check(
    error="str_contains('{pattern}')",
)
def str_contains(
    data: PolarsData,
    pattern: Union[str, re.Pattern],
) -> pl.LazyFrame:
    """Ensure that a pattern can be found in the string.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param pattern: Regular expression pattern to use for searching.
    """

    pattern = pattern.pattern if isinstance(pattern, re.Pattern) else pattern
    return data.lazyframe.select(
        pl.col(data.key).str.contains(pattern=pattern, literal=False)
    )


@register_builtin_check(
    error="str_startswith('{string}')",
)
def str_startswith(data: PolarsData, string: str) -> pl.LazyFrame:
    """Ensure that all values start with a certain string.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param string: String all values should start with.
    """

    return data.lazyframe.select(pl.col(data.key).str.starts_with(string))


@register_builtin_check(error="str_endswith('{string}')")
def str_endswith(data: PolarsData, string: str) -> pl.LazyFrame:
    """Ensure that all values end with a certain string.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param string: String all values should end with.
    """
    return data.lazyframe.select(pl.col(data.key).str.ends_with(string))


@register_builtin_check(
    error="str_length({min_value}, {max_value})",
)
def str_length(
    data: PolarsData,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> pl.LazyFrame:
    """Ensure that the length of strings is within a specified range.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param min_value: Minimum length of strings (inclusive). (default: no minimum)
    :param max_value: Maximum length of strings (inclusive). (default: no maximum)
    """
    if min_value is None and max_value is None:
        raise ValueError(
            "Must provide at least on of 'min_value' and 'max_value'"
        )

    n_chars = pl.col(data.key).str.len_chars()
    if min_value is None:
        expr = n_chars.le(max_value)
    elif max_value is None:
        expr = n_chars.ge(min_value)
    else:
        expr = n_chars.is_between(min_value, max_value)

    return data.lazyframe.select(expr)


@register_builtin_check(
    error="unique_values_eq({values})",
)
def unique_values_eq(data: PolarsData, values: Iterable) -> bool:
    """Ensure that unique values in the data object contain all values.

    .. note::
        In contrast with :func:`isin`, this check makes sure that all the items
        in the ``values`` iterable are contained within the series.

    :param data: NamedTuple PolarsData contains the dataframe and column name for the check. The key
        to access the dataframe is "dataframe", and the key the to access the column name is "key".
    :param values: The set of values that must be present. May be any iterable.
    """

    return (
        set(data.lazyframe.collect().get_column(data.key).unique()) == values
    )
