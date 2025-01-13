"""Built-in checks for PySpark."""

from typing import Any, Iterable, TypeVar

import pyspark.sql.types as pst
from pyspark.sql.functions import col

import pandera.strategies as st
from pandera.api.extensions import register_builtin_check
from pandera.api.pyspark.types import PysparkDataframeColumnObject
from pandera.backends.pyspark.decorators import register_input_datatypes
from pandera.backends.pyspark.utils import convert_to_list

T = TypeVar("T")
ALL_NUMERIC_TYPE = [
    pst.LongType,
    pst.IntegerType,
    pst.ByteType,
    pst.ShortType,
    pst.DoubleType,
    pst.DecimalType,
    pst.FloatType,
]
ALL_DATE_TYPE = [pst.DateType, pst.TimestampType]
BOLEAN_TYPE = pst.BooleanType
BINARY_TYPE = pst.BinaryType
STRING_TYPE = pst.StringType


@register_builtin_check(
    aliases=["eq"],
    error="equal_to({value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(
        ALL_NUMERIC_TYPE, ALL_DATE_TYPE, STRING_TYPE, BINARY_TYPE, BOLEAN_TYPE
    )
)
def equal_to(data: PysparkDataframeColumnObject, value: Any) -> bool:
    """Ensure all elements of a data container equal a certain value.

    :param data: PysparkDataframeColumnObject column object which is a contains dataframe and column name to do the check
    :param value: values in this DataFrame data structure must be
        equal to this value.
    """
    cond = col(data.column_name) == value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["ne"],
    strategy=st.ne_strategy,
    error="not_equal_to({value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(
        ALL_NUMERIC_TYPE, ALL_DATE_TYPE, STRING_TYPE, BINARY_TYPE, BOLEAN_TYPE
    )
)
def not_equal_to(data: PysparkDataframeColumnObject, value: Any) -> bool:
    """Ensure no elements of a data container equals a certain value.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param value: This value must not occur in the checked
    """
    cond = col(data.column_name) != value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["gt"],
    error="greater_than({min_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def greater_than(data: PysparkDataframeColumnObject, min_value: Any) -> bool:
    """
    Ensure values of a data container are strictly greater than a minimum
    value.
    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param min_value: Lower bound to be exceeded.
    """
    cond = col(data.column_name) > min_value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["ge"],
    strategy=st.ge_strategy,
    error="greater_than_or_equal_to({min_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def greater_than_or_equal_to(
    data: PysparkDataframeColumnObject, min_value: Any
) -> bool:
    """Ensure all values are greater or equal a certain value.
    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param min_value: Allowed minimum value for values of a series. Must be
        a type comparable to the dtype of the column datatype of pyspark
    """
    cond = col(data.column_name) >= min_value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["lt"],
    strategy=st.lt_strategy,
    error="less_than({max_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def less_than(data: PysparkDataframeColumnObject, max_value: Any) -> bool:
    """Ensure values of a series are strictly below a maximum value.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param max_value: All elements of a series must be strictly smaller
        than this. Must be a type comparable to the dtype of the column datatype of pyspark
    """
    # test case exists but not detected by pytest so no cover added
    if max_value is None:  # pragma: no cover
        raise ValueError("max_value must not be None")
    cond = col(data.column_name) < max_value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["le"],
    strategy=st.le_strategy,
    error="less_than_or_equal_to({max_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def less_than_or_equal_to(
    data: PysparkDataframeColumnObject, max_value: Any
) -> bool:
    """Ensure values of a series are strictly below a maximum value.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param max_value: Upper bound not to be exceeded. Must be
        a type comparable to the dtype of the column datatype of pyspark
    """
    # test case exists but not detected by pytest so no cover added
    if max_value is None:  # pragma: no cover
        raise ValueError("max_value must not be None")
    cond = col(data.column_name) <= max_value
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    aliases=["between"],
    strategy=st.in_range_strategy,
    error="in_range({min_value}, {max_value})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(ALL_NUMERIC_TYPE, ALL_DATE_TYPE)
)
def in_range(
    data: PysparkDataframeColumnObject,
    min_value: T,
    max_value: T,
    include_min: bool = True,
    include_max: bool = True,
):
    """Ensure all values of a column are within an interval.

    Both endpoints must be a type comparable to the dtype of the
    :class:`pyspark.sql.function.col` to be validated.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
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
    cond_right = (
        col(data.column_name) >= min_value
        if include_min
        else col(data.column_name) > min_value
    )
    cond_left = (
        col(data.column_name) <= max_value
        if include_max
        else col(data.column_name) < max_value
    )
    return data.dataframe.filter(~(cond_right & cond_left)).limit(1).count() == 0  # type: ignore


@register_builtin_check(
    strategy=st.isin_strategy,
    error="isin({allowed_values})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(
        ALL_NUMERIC_TYPE, ALL_DATE_TYPE, STRING_TYPE, BINARY_TYPE
    )
)
def isin(data: PysparkDataframeColumnObject, allowed_values: Iterable) -> bool:
    """Ensure only allowed values occur within a series.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    This checks whether all elements of a :class:`pyspark.sql.function.col`
    are part of the set of elements of allowed values. If allowed
    values is a string, the set of elements consists of all distinct
    characters of the string. Thus only single characters which occur
    in allowed_values at least once can meet this condition. If you
    want to check for substrings use :meth:`Check.str_contains`.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param allowed_values: The set of allowed values. May be any iterable.
    """
    return (
        data.dataframe.filter(
            ~col(data.column_name).isin(list(allowed_values))
        )
        .limit(1)
        .count()
        == 0
    )


@register_builtin_check(
    strategy=st.notin_strategy,
    error="notin({forbidden_values})",
)
@register_input_datatypes(
    acceptable_datatypes=convert_to_list(
        ALL_NUMERIC_TYPE, ALL_DATE_TYPE, STRING_TYPE, BINARY_TYPE
    )
)
def notin(
    data: PysparkDataframeColumnObject, forbidden_values: Iterable
) -> bool:
    """Ensure some defined values don't occur within a series.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    Like :meth:`Check.isin` this check operates on single characters if
    it is applied on strings. If forbidden_values is a string, it is understood
    as set of prohibited characters. Any string of length > 1 can't be in it by
    design.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param forbidden_values: The set of values which should not occur. May
        be any iterable.
    """
    return (
        data.dataframe.filter(
            col(data.column_name).isin(list(forbidden_values))
        )
        .limit(1)
        .count()
        == 0
    )


@register_builtin_check(
    strategy=st.str_contains_strategy,
    error="str_contains('{pattern}')",
)
@register_input_datatypes(acceptable_datatypes=convert_to_list(STRING_TYPE))
def str_contains(data: PysparkDataframeColumnObject, pattern: str) -> bool:
    """Ensure that a pattern can be found within each row.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param pattern: Regular expression pattern to use for searching
    """
    return (
        data.dataframe.filter(~col(data.column_name).rlike(pattern))
        .limit(1)
        .count()
        == 0
    )


@register_builtin_check(
    error="str_startswith('{string}')",
)
@register_input_datatypes(acceptable_datatypes=convert_to_list(STRING_TYPE))
def str_startswith(data: PysparkDataframeColumnObject, string: str) -> bool:
    """Ensure that all values start with a certain string.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param string: String all values should start with
    """
    cond = col(data.column_name).startswith(string)
    return data.dataframe.filter(~cond).limit(1).count() == 0


@register_builtin_check(
    strategy=st.str_endswith_strategy, error="str_endswith('{string}')"
)
@register_input_datatypes(acceptable_datatypes=convert_to_list(STRING_TYPE))
def str_endswith(data: PysparkDataframeColumnObject, string: str) -> bool:
    """Ensure that all values end with a certain string.

    Remember it can be a compute intensive check on large dataset. So, use it with caution.

    :param data: NamedTuple PysparkDataframeColumnObject contains the dataframe and column name for the check. The keys
                to access the dataframe is "dataframe" and column name using "column_name".
    :param string: String all values should end with
    """
    cond = col(data.column_name).endswith(string)
    return data.dataframe.filter(~cond).limit(1).count() == 0
