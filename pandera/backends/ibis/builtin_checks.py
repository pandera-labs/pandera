"""Built-in checks for Ibis."""

from typing import Any, TypeVar

import ibis.expr.types as ir

from pandera.api.extensions import register_builtin_check
from pandera.api.ibis.types import IbisData

T = TypeVar("T")


@register_builtin_check(
    aliases=["eq"],
    error="equal_to({value})",
)
def equal_to(data: IbisData, value: Any) -> ir.Table:
    """Ensure all elements of a data container equal a certain value.

    :param data: NamedTuple IbisData contains the table and column name for the check. The key
        to access the table is "table", and the key to access the column name is "key".
    :param value: Values in this Ibis data structure must be
        equal to this value.
    """
    return data.table[data.key] == value
