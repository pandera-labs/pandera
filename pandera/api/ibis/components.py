"""Core Ibis schema component specifications."""

from typing import Any, Optional, Type

import ibis.expr.types as ir

from pandera.api.base.types import CheckList
from pandera.api.dataframe.components import ComponentSchema
from pandera.api.ibis.types import IbisDtypeInputTypes
from pandera.backends.ibis.register import register_ibis_backends
from pandera.engines import ibis_engine
from pandera.utils import is_regex


class Column(ComponentSchema[ir.Table]):
    """Validate types and properties of table columns."""

    def __init__(
        self,
        dtype: IbisDtypeInputTypes = None,
        checks: Optional[CheckList] = None,
        nullable: bool = False,
        unique: bool = False,
        coerce: bool = False,
        required: bool = True,
        name: Optional[str] = None,
        regex: bool = False,
        title: Optional[str] = None,
        description: Optional[str] = None,
        default: Optional[Any] = None,
        metadata: Optional[dict] = None,
        drop_invalid_rows: bool = False,
        **column_kwargs,
    ) -> None:
        """Create column validator object.

        :param dtype: datatype of the column. The datatype for type-checking
            a dataframe. All `Ibis datatypes <TODO(deepyaman): Insert link.>`__,
            supported built-in Python types that are supported by Ibis,
            and the pandera Ibis engine :ref:`datatypes <ibis-dtypes>`.
        :param checks: checks to verify validity of the column
        :param nullable: Whether or not column can contain null values.
        :param unique: whether column values should be unique
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype. This has no effect on columns
            where ``dtype=None``.
        :param required: Whether or not column is allowed to be missing
        :param name: column name in dataframe to validate. Names in the format
            '^{regex_pattern}$' are treated as regular expressions. During
            validation, this schema will be applied to any columns matching this
            pattern.
        :param regex: whether the ``name`` attribute should be treated as a
            regex pattern to apply to multiple columns in a dataframe. If the
            name is a regular expression, this attribute will automatically be
            set to True.
        :param title: A human-readable label for the column.
        :param description: An arbitrary textual description of the column.
        :param default: The default value for missing values in the column.
        :param metadata: An optional key value data.
        :param drop_invalid_rows: if True, drop invalid rows on validation.

        :raises SchemaInitError: if impossible to build schema from parameters

        :example:

        >>> import pandas as pd
        >>> import pandera as pa
        >>>
        >>>
        >>> schema = pa.DataFrameSchema({
        ...     "column": pa.Column(str)
        ... })
        >>>
        >>> schema.validate(pd.DataFrame({"column": ["foo", "bar"]}))
          column
        0    foo
        1    bar

        See :ref:`here<column>` for more usage details.
        """

        super().__init__(
            dtype=dtype,
            checks=checks,
            nullable=nullable,
            unique=unique,
            coerce=coerce,
            name=name,
            title=title,
            description=description,
            default=default,
            metadata=metadata,
            drop_invalid_rows=drop_invalid_rows,
            **column_kwargs,
        )
        self.required = required
        self.regex = regex
        self.name = name

        # self.set_regex()  # TODO(deepyaman): Implement method.

    # pylint: disable=unused-argument
    @staticmethod
    def register_default_backends(check_obj_cls: Type):
        register_ibis_backends()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value) -> None:
        self._dtype = ibis_engine.Engine.dtype(value) if value else None

    @property
    def selector(self):
        if self.name is not None and not is_regex(self.name) and self.regex:
            return f"^{self.name}$"
        return self.name

    def set_name(self, name: str):
        """Set or modify the name of a column object.

        :param str name: the name of the column object
        """
        self.name = name
        return self
