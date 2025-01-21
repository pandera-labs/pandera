"""Core pyspark dataframe container specification."""

from __future__ import annotations

import copy
import os
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Type,
    Union,
    cast,
    overload,
)

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructField, StructType

from pandera import errors
from pandera.api.base.error_handler import ErrorHandler
from pandera.api.base.schema import BaseSchema
from pandera.api.base.types import StrictType
from pandera.api.checks import Check
from pandera.api.pyspark.types import CheckList, PySparkDtypeInputTypes
from pandera.backends.pyspark.register import register_pyspark_backends
from pandera.config import get_config_context
from pandera.dtypes import DataType, UniqueSettings
from pandera.engines import pyspark_engine

if TYPE_CHECKING:
    import pandera.api.pyspark.components

N_INDENT_SPACES = 4


class DataFrameSchema(BaseSchema):  # pylint: disable=too-many-public-methods
    """A light-weight PySpark DataFrame validator."""

    def __init__(
        self,
        columns: Optional[  # type: ignore [name-defined]
            Dict[Any, pandera.api.pyspark.components.Column]  # type: ignore [name-defined]
        ] = None,
        checks: Optional[CheckList] = None,
        dtype: PySparkDtypeInputTypes = None,
        coerce: bool = False,
        strict: StrictType = False,
        name: Optional[str] = None,
        ordered: bool = False,
        unique: Optional[Union[str, List[str]]] = None,
        report_duplicates: UniqueSettings = "all",
        unique_column_names: bool = False,
        title: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Initialize DataFrameSchema validator.

        :param columns: a dict where keys are column names and values are
            Column objects specifying the datatypes and properties of a
            particular column.
        :type columns: mapping of column names and column schema component.
        :param checks: dataframe-wide checks.
        :param dtype: datatype of the dataframe. This overrides the data
            types specified in any of the columns. If a string is specified,
            then assumes one of the valid pyspark string values:
            https://spark.apache.org/docs/latest/sql-ref-datatypes.html.
        :param coerce: whether or not to coerce all of the columns on
            validation. This has no effect on columns where
            ``dtype=None``
        :param strict: ensure that all and only the columns defined in the
            schema are present in the dataframe. If set to 'filter',
            only the columns in the schema will be passed to the validated
            dataframe. If set to filter and columns defined in the schema
            are not present in the dataframe, will throw an error.
        :param name: name of the schema.
        :param ordered: whether or not to validate the columns order.
        :param unique: a list of columns that should be jointly unique.
        :param report_duplicates: how to report unique errors
            - `exclude_first`: report all duplicates except first occurence
            - `exclude_last`: report all duplicates except last occurence
            - `all`: (default) report all duplicates
        :param unique_column_names: whether or not column names must be unique.
        :param title: A human-readable label for the schema.
        :param description: An arbitrary textual description of the schema.
        :param metadata: An optional key-value data.

        :raises SchemaInitError: if impossible to build schema from parameters

        :examples:

        >>> import pandera.pyspark as psa
        >>> import pyspark.sql.types as pt
        >>>
        >>> schema = psa.DataFrameSchema({
        ...     "str_column": psa.Column(str),
        ...     "float_column": psa.Column(float),
        ...     "int_column": psa.Column(int),
        ...     "date_column": psa.Column(pt.DateType),
        ... })

        Use the pyspark API to define checks, which takes a function with
        the signature: ``ps.Dataframe -> Union[bool]`` where the
        output contains boolean values.

        >>> schema_withchecks = psa.DataFrameSchema({
        ...     "probability": psa.Column(
        ...         pt.DoubleType(), psa.Check.greater_than(0)),
        ...
        ...     # check that the "category" column contains a few discrete
        ...     # values, and the majority of the entries are dogs.
        ...     "category": psa.Column(
        ...         pt.StringType(), psa.Check.str_startswith("B"),
        ...            ),
        ... })

        See :ref:`here<DataFrameSchemas>` for more usage details.

        """

        if columns is None:
            columns = {}
        _validate_columns(columns)
        columns = _columns_renamed(columns)

        if checks is None:
            checks = []
        if isinstance(checks, (Check)):
            checks = [checks]

        super().__init__(
            dtype=dtype,
            checks=checks,
            name=name,
            title=title,
            description=description,
            metadata=metadata,
        )

        self.columns: Dict[Any, pandera.api.pyspark.components.Column] = (  # type: ignore [name-defined]
            {} if columns is None else columns
        )

        if strict not in (
            False,
            True,
            "filter",
        ):
            raise errors.SchemaInitError(
                "strict parameter must equal either `True`, `False`, "
                "or `'filter'`."
            )

        self.strict: Union[bool, str] = strict
        self._coerce = coerce
        self.ordered = ordered
        self._unique = [unique] if isinstance(unique, str) else unique
        self.report_duplicates = report_duplicates
        self.unique_column_names = unique_column_names

        self.metadata = metadata

    @staticmethod
    def register_default_backends(check_obj_cls: Type):
        register_pyspark_backends()

    @property
    def coerce(self) -> bool:
        """Whether to coerce series to specified type."""
        if isinstance(self.dtype, DataType):
            return self.dtype.auto_coerce or self._coerce
        return self._coerce

    @coerce.setter
    def coerce(self, value: bool) -> None:
        """Set coerce attribute"""
        self._coerce = value

    @property
    def unique(self):
        """List of columns that should be jointly unique."""
        return self._unique

    @unique.setter
    def unique(self, value: Optional[Union[str, List[str]]]) -> None:
        """Set unique attribute."""
        self._unique = [value] if isinstance(value, str) else value

    @property
    def dtypes(self) -> Dict[str, DataType]:
        # pylint:disable=anomalous-backslash-in-string
        """
        A dict where the keys are column names and values are
        :class:`~pandera.dtypes.DataType` s for the column. Excludes columns
        where `regex=True`.

        :returns: dictionary of columns and their associated dtypes.
        """
        regex_columns = [
            name for name, col in self.columns.items() if col.regex
        ]
        if regex_columns:
            warnings.warn(
                "Schema has columns specified as regex column names: "
                f"{regex_columns}. Use the `get_dtypes` to get the datatypes "
                "for these columns.",
                UserWarning,
            )
        return {n: c.dtype for n, c in self.columns.items() if not c.regex}

    def get_metadata(self) -> Optional[dict]:
        """Provide metadata for columns and schema level"""
        res: Dict[Any, Any] = {"columns": {}}
        for k in self.columns.keys():
            res["columns"][k] = self.columns[k].properties["metadata"]

        res["dataframe"] = self.metadata

        meta = {}
        meta[self.name] = res
        return meta

    def get_dtypes(self, dataframe: DataFrame) -> Dict[str, DataType]:
        """
        Same as the ``dtype`` property, but expands columns where
        ``regex == True`` based on the supplied dataframe.

        :returns: dictionary of columns and their associated dtypes.
        """
        regex_dtype = {}
        for _, column in self.columns.items():
            if column.regex:
                regex_dtype.update(
                    {
                        c: column.dtype
                        for c in column.get_backend(
                            dataframe
                        ).get_regex_columns(
                            column,
                            dataframe,
                        )
                    }
                )
        return {
            **{n: c.dtype for n, c in self.columns.items() if not c.regex},
            **regex_dtype,
        }

    @property
    def dtype(
        self,
    ) -> DataType:
        """Get the dtype property."""
        return self._dtype  # type: ignore

    @dtype.setter
    def dtype(self, value: PySparkDtypeInputTypes) -> None:
        """Set the pyspark dtype property."""
        self._dtype = (
            pyspark_engine.Engine.dtype(value) if value else None
        )  # pylint:disable=no-value-for-parameter

    def coerce_dtype(self, check_obj: DataFrame) -> DataFrame:
        return self.get_backend(check_obj).coerce_dtype(check_obj, schema=self)

    def validate(
        self,
        check_obj: DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = True,
        inplace: bool = False,
    ):
        """Check if all columns in a dataframe have a column in the Schema.

        :param check_obj: DataFrame object i.e. the dataframe to be validated.
        :param head: Not used since spark has no concept of head or tail
        :param tail: Not used since spark has no concept of head or tail
        :param sample: validate a random sample of n% rows. Value ranges from
                0-1, for example 10% rows can be sampled using setting value as 0.1.
                refer below documentation.
                https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrame.sample.html
        :param random_state: random seed for the ``sample`` argument.
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        :returns: validated ``DataFrame``

        :raises SchemaError: when ``DataFrame`` violates built-in or custom
            checks.

        :example:

        Calling ``schema.validate`` returns the dataframe.


        >>> import pandera.pyspark as psa
        >>> from pyspark.sql import SparkSession
        >>> import pyspark.sql.types as T
        >>> spark = SparkSession.builder.getOrCreate()
        >>>
        >>> data = [("Bread", 9), ("Butter", 15)]
        >>> spark_schema = T.StructType(
        ...         [
        ...             T.StructField("product", T.StringType(), False),
        ...             T.StructField("price", T.IntegerType(), False),
        ...         ],
        ...     )
        >>> df = spark.createDataFrame(data=data, schema=spark_schema)
        >>>
        >>> schema_withchecks = psa.DataFrameSchema(
        ...         columns={
        ...             "product": psa.Column("str", checks=psa.Check.str_startswith("B")),
        ...             "price": psa.Column("int", checks=psa.Check.gt(5)),
        ...         },
        ...         name="product_schema",
        ...         description="schema for product info",
        ...         title="ProductSchema",
        ...     )
        >>>
        >>> schema_withchecks.validate(df).take(2)
            [Row(product='Bread', price=9), Row(product='Butter', price=15)]
        """
        if not get_config_context().validation_enabled:
            return check_obj
        error_handler = ErrorHandler(lazy)

        return self._validate(
            check_obj=check_obj,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
            error_handler=error_handler,
        )

    def _validate(
        self,
        check_obj: DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
        error_handler: ErrorHandler = None,
    ):
        return self.get_backend(check_obj).validate(
            check_obj=check_obj,
            schema=self,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
            error_handler=error_handler,
        )

    def __call__(
        self,
        dataframe: DataFrame,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = True,
        inplace: bool = False,
    ):
        """Alias for :func:`DataFrameSchema.validate` method.

        :param dataframe: DataFrame object i.e. the dataframe to be validated.
        :param head: Not used since spark has no concept of head or tail.
        :type head: int
        :param tail: Not used since spark has no concept of head or tail.
        :type tail: int
        :param sample: validate a random sample of n% rows. Value ranges from
                0-1, for example 10% rows can be sampled using setting value as 0.1.
                refer below documentation.
                https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrame.sample.html
        :param lazy: if True, lazily evaluates dataframe against all validation
            checks and raises a ``SchemaErrors``. Otherwise, raise
            ``SchemaError`` as soon as one occurs.
        :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
        """
        return self.validate(
            dataframe, head, tail, sample, random_state, lazy, inplace
        )

    def __repr__(self) -> str:
        """Represent string for logging."""
        return (
            f"<Schema {self.__class__.__name__}("
            f"columns={self.columns}, "
            f"checks={self.checks}, "
            f"coerce={self.coerce}, "
            f"dtype={self._dtype}, "
            f"strict={self.strict}, "
            f"name={self.name}, "
            f"ordered={self.ordered}, "
            f"unique_column_names={self.unique_column_names}, "
            f"title={self.title}, "
            f"description='{self.description}, "
            f"metadata='{self.metadata}, "
            ")>"
        )

    def __str__(self) -> str:
        """Represent string for user inspection."""

        def _format_multiline(json_str, arg):
            return "\n".join(
                f"{indent}{line}" if i != 0 else f"{indent}{arg}={line}"
                for i, line in enumerate(json_str.split("\n"))
            )

        indent = " " * N_INDENT_SPACES
        if self.columns:
            columns_str = f"{indent}columns={{\n"
            for colname, col in self.columns.items():
                columns_str += f"{indent * 2}'{colname}': {col}\n"
            columns_str += f"{indent}}}"
        else:
            columns_str = f"{indent}columns={{}}"

        if self.checks:
            checks_str = f"{indent}checks=[\n"
            for check in self.checks:
                checks_str += f"{indent * 2}{check}\n"
            checks_str += f"{indent}]"
        else:
            checks_str = f"{indent}checks=[]"

        return (
            f"<Schema {self.__class__.__name__}(\n"
            f"{columns_str},\n"
            f"{checks_str},\n"
            f"{indent}coerce={self.coerce},\n"
            f"{indent}dtype={self._dtype},\n"
            f"{indent}strict={self.strict}\n"
            f"{indent}name={self.name},\n"
            f"{indent}ordered={self.ordered},\n"
            f"{indent}unique_column_names={self.unique_column_names},\n"
            f"{indent}metadata={self.metadata}, \n"
            ")>"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        def _compare_dict(obj):
            return {k: v for k, v in obj.__dict__.items() if k}

        return _compare_dict(self) == _compare_dict(other)

    @classmethod
    def __get_validators__(cls):
        yield cls._pydantic_validate

    @classmethod
    def _pydantic_validate(cls, schema: Any) -> DataFrameSchema:
        """Verify that the input is a compatible DataFrameSchema."""
        if not isinstance(schema, cls):  # type: ignore
            raise TypeError(f"{schema} is not a {cls}.")

        return cast("DataFrameSchema", schema)

    #####################
    # Schema IO Methods #
    #####################

    def to_script(self, fp: Union[str, Path] = None) -> DataFrameSchema:
        """Create DataFrameSchema from yaml file.

        :param path: str, Path to write script
        :returns: dataframe schema.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.to_script(self, fp)

    @classmethod
    def from_yaml(cls, yaml_schema) -> DataFrameSchema:
        """Create DataFrameSchema from yaml file.

        :param yaml_schema: str, Path to yaml schema, or serialized yaml
            string.
        :returns: dataframe schema.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.from_yaml(yaml_schema)

    def to_yaml(self, stream: Optional[os.PathLike] = None) -> Optional[str]:
        """Write DataFrameSchema to yaml file.

        :param stream: file stream to write to. If None, dumps to string.
        :returns: yaml string if stream is None, otherwise returns None.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.to_yaml(self, stream=stream)

    @classmethod
    def from_json(cls, source) -> DataFrameSchema:
        """Create DataFrameSchema from json file.

        :param source: str, Path to json schema, or serialized yaml
            string.
        :returns: dataframe schema.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.from_json(source)

    @overload
    def to_json(
        self, target: None = None, **kwargs
    ) -> str:  # pragma: no cover
        ...

    @overload
    def to_json(
        self, target: os.PathLike, **kwargs
    ) -> None:  # pragma: no cover
        ...

    def to_json(
        self, target: Optional[os.PathLike] = None, **kwargs
    ) -> Optional[str]:
        """Write DataFrameSchema to json file.

        :param target: file target to write to. If None, dumps to string.
        :returns: json string if target is None, otherwise returns None.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import,redefined-outer-name
        import pandera.io

        return pandera.io.to_json(self, target, **kwargs)

    def to_structtype(self) -> StructType:
        """Recover fields of DataFrameSchema as a Pyspark StructType object.

        As the output of this method will be used to specify a read schema in Pyspark
            (avoiding automatic schema inference), the False `nullable` properties are
            just ignored, as this check will be executed by the Pandera validations
            after a dataset is read.

        :returns: StructType object with current schema fields.
        """
        fields = [
            StructField(column, self.columns[column].dtype.type, True)
            for column in self.columns
        ]
        return StructType(fields)

    def to_ddl(self) -> str:
        """Recover fields of DataFrameSchema as a Pyspark DDL string.

        :returns: String with current schema fields, in compact DDL format.
        """
        # `StructType.toDDL()` is only available in internal java classes
        spark = SparkSession.builder.getOrCreate()
        # Create a base dataframe from where we access underlying Java classes
        empty_df_with_schema = spark.createDataFrame([], self.to_structtype())

        return empty_df_with_schema._jdf.schema().toDDL()


def _validate_columns(
    column_dict: dict[Any, pandera.api.pyspark.components.Column],  # type: ignore [name-defined]
) -> None:
    for column_name, column in column_dict.items():
        for check in column.checks:
            if check.groupby is None or callable(check.groupby):
                continue
            nonexistent_groupby_columns = [
                c for c in check.groupby if c not in column_dict
            ]
            if nonexistent_groupby_columns:
                raise errors.SchemaInitError(
                    f"groupby argument {nonexistent_groupby_columns} in "
                    f"Check for Column {column_name} not "
                    "specified in the DataFrameSchema."
                )


def _columns_renamed(
    columns: dict[Any, pandera.api.pyspark.components.Column],  # type: ignore [name-defined]
) -> dict[Any, pandera.api.pyspark.components.Column]:  # type: ignore [name-defined]
    def renamed(column, new_name):
        column = copy.deepcopy(column)
        column.set_name(new_name)
        return column

    return {
        column_name: renamed(column, column_name)
        for column_name, column in columns.items()
    }
