"""Module for reading and writing schema objects."""

import json
import warnings
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

import pandera.errors
from pandera import dtypes
from pandera.api.checks import Check
from pandera.api.pandas.components import Column
from pandera.api.pandas.container import DataFrameSchema
from pandera.engines import pandas_engine
from pandera.schema_statistics import get_dataframe_schema_statistics

try:
    import black
    import yaml
    from frictionless import Schema as FrictionlessSchema
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "IO and formatting requires 'pyyaml', 'black' and 'frictionless'"
        "to be installed.\n"
        "You can install pandera together with the IO dependencies with:\n"
        "pip install pandera[io]\n"
    ) from exc


DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _get_dtype_string_alias(dtype: pandas_engine.DataType) -> str:
    """Get string alias of the datatype for serialization.

    Calling pandas_engine.Engine.dtype(<string_alias>) should be a valid
    operation
    """
    str_alias = str(dtype)
    try:
        pandas_engine.Engine.dtype(str_alias)
    except TypeError as e:  # pragma: no cover
        raise TypeError(
            f"string alias {str_alias} for datatype "
            f"'{dtype.__module__}.{dtype.__class__.__name__}' not "
            "recognized."
        ) from e
    return f'"{dtype}"'


def _serialize_check_stats(check_stats, dtype=None):
    """Serialize check statistics into json/yaml-compatible format."""

    def handle_stat_dtype(stat):
        if pandas_engine.Engine.dtype(dtypes.DateTime).check(
            dtype
        ) and hasattr(stat, "strftime"):
            # try serializing stat as a string if it's datetime-like,
            # otherwise return original value
            return stat.strftime(DATETIME_FORMAT)
        elif pandas_engine.Engine.dtype(dtypes.Timedelta).check(dtype):
            # try serializing stat into an int in nanoseconds if it's
            # timedelta-like, otherwise return original value
            return getattr(stat, "value", stat)

        return stat

    # Extract check options if they exist
    check_options = (
        check_stats.pop("options", {}) if isinstance(check_stats, dict) else {}
    )

    # Handle unary checks
    if isinstance(check_stats, dict) and len(check_stats) == 1:
        value = handle_stat_dtype(list(check_stats.values())[0])
        if check_options:
            return {"value": value, "options": check_options}
        return value

    # Handle dictionary case
    if isinstance(check_stats, dict):
        serialized_check_stats = {}
        for arg, stat in check_stats.items():
            serialized_check_stats[arg] = handle_stat_dtype(stat)
        if check_options:
            serialized_check_stats["options"] = check_options
        return serialized_check_stats

    return handle_stat_dtype(check_stats)


def _serialize_dataframe_stats(dataframe_checks):
    """
    Serialize global dataframe check statistics into json/yaml-compatible
    format.
    """
    serialized_checks = {}

    for check_name, check_stats in dataframe_checks.items():
        # The case that `check_name` is not registered is handled in
        # `parse_checks` so we know that `check_name` exists.

        # infer dtype of statistics and serialize them
        serialized_checks[check_name] = _serialize_check_stats(check_stats)

    return serialized_checks


def _serialize_component_stats(component_stats):
    """
    Serialize column or index statistics into json/yaml-compatible format.
    """
    serialized_checks = None
    if component_stats["checks"] is not None:
        serialized_checks = {}
        for check_name, check_stats in component_stats["checks"].items():
            serialized_checks[check_name] = _serialize_check_stats(
                check_stats, component_stats["dtype"]
            )

    dtype = component_stats.get("dtype")
    if dtype:
        dtype = str(dtype)

    description = component_stats.get("description")
    title = component_stats.get("title")

    return {
        "title": title,
        "description": description,
        "dtype": dtype,
        "nullable": component_stats["nullable"],
        "checks": serialized_checks,
        **{
            key: component_stats.get(key)
            for key in [
                "name",
                "unique",
                "coerce",
                "required",
                "regex",
            ]
            if key in component_stats
        },
    }


def serialize_schema(dataframe_schema):
    """Serialize dataframe schema into into json/yaml-compatible format."""
    from pandera import __version__  # pylint: disable=import-outside-toplevel

    statistics = get_dataframe_schema_statistics(dataframe_schema)

    columns, index, checks = None, None, None
    if statistics["columns"] is not None:
        columns = {
            col_name: _serialize_component_stats(column_stats)
            for col_name, column_stats in statistics["columns"].items()
        }

    if statistics["index"] is not None:
        index = [
            _serialize_component_stats(index_stats)
            for index_stats in statistics["index"]
        ]

    if statistics["checks"] is not None:
        checks = _serialize_dataframe_stats(statistics["checks"])

    return {
        "schema_type": "dataframe",
        "version": __version__,
        "columns": columns,
        "checks": checks,
        "index": index,
        "dtype": dataframe_schema.dtype,
        "coerce": dataframe_schema.coerce,
        "strict": dataframe_schema.strict,
        "name": dataframe_schema.name,
        "ordered": dataframe_schema.ordered,
        "unique": dataframe_schema.unique,
        "report_duplicates": dataframe_schema.report_duplicates,
        "unique_column_names": dataframe_schema.unique_column_names,
        "add_missing_columns": dataframe_schema.add_missing_columns,
        "title": dataframe_schema.title,
        "description": dataframe_schema.description,
        "drop_invalid_rows": dataframe_schema.drop_invalid_rows,
    }


def _deserialize_check_stats(check, serialized_check_stats, dtype=None):
    """Deserialize check statistics and reconstruct check with options."""

    def handle_stat_dtype(stat):
        try:
            if pandas_engine.Engine.dtype(dtypes.DateTime).check(dtype):
                return pd.to_datetime(stat, format=DATETIME_FORMAT)
            elif pandas_engine.Engine.dtype(dtypes.Timedelta).check(dtype):
                # serialize to int in nanoseconds
                return pd.to_timedelta(stat, unit="ns")
        except (TypeError, ValueError):
            return stat
        return stat

    # Extract options if they exist
    options = {}
    if isinstance(serialized_check_stats, dict):
        # handle case where serialized check stats are in the form of a
        # dictionary mapping Check arg names to values.
        options = serialized_check_stats.pop("options", {})
        # Handle special case for unary checks with options
        if (
            "value" in serialized_check_stats
            and len(serialized_check_stats) == 1
        ):
            serialized_check_stats = serialized_check_stats["value"]

    # Create check with original logic
    if isinstance(serialized_check_stats, dict):
        check_stats = {}
        for arg, stat in serialized_check_stats.items():
            check_stats[arg] = handle_stat_dtype(stat)
        check_instance = check(**check_stats)
    else:
        # otherwise assume unary check function signature
        check_instance = check(handle_stat_dtype(serialized_check_stats))

    # Apply options if they exist
    if options:
        for option_name, option_value in options.items():
            setattr(check_instance, option_name, option_value)

    return check_instance


def _deserialize_component_stats(serialized_component_stats):
    dtype = serialized_component_stats.get("dtype")
    if dtype:
        dtype = pandas_engine.Engine.dtype(dtype)

    description = serialized_component_stats.get("description")
    title = serialized_component_stats.get("title")

    checks = serialized_component_stats.get("checks")
    if checks is not None:
        checks = [
            _deserialize_check_stats(
                getattr(Check, check_name), check_stats, dtype
            )
            for check_name, check_stats in checks.items()
        ]
    return {
        "title": title,
        "description": description,
        "dtype": dtype,
        "checks": checks,
        **{
            key: serialized_component_stats.get(key)
            for key in [
                "name",
                "nullable",
                "unique",
                "coerce",
                "required",
                "regex",
            ]
            if key in serialized_component_stats
        },
    }


def deserialize_schema(serialized_schema):
    """
    De-serialize the schema from a JSON-able mapping.

    :param serialized_schema: a mapping representing the schema
    :returns:
        the schema de-serialized into :class:`~pandera.api.pandas.container.DataFrameSchema`
    """
    # pylint: disable=import-outside-toplevel
    from pandera import Index, MultiIndex

    # GH#475
    serialized_schema = serialized_schema if serialized_schema else {}

    if not isinstance(serialized_schema, Mapping):
        raise pandera.errors.SchemaDefinitionError(
            "Schema representation must be a mapping."
        )

    columns = serialized_schema.get("columns")
    index = serialized_schema.get("index")
    checks = serialized_schema.get("checks")

    if columns is not None:
        columns = {
            col_name: Column(**_deserialize_component_stats(column_stats))
            for col_name, column_stats in columns.items()
        }

    if index is not None:
        index = [
            _deserialize_component_stats(index_component)
            for index_component in index
        ]

    if checks is not None:
        # handles unregistered checks by raising AttributeErrors from getattr
        checks = [
            _deserialize_check_stats(getattr(Check, check_name), check_stats)
            for check_name, check_stats in checks.items()
        ]

    if index is None:
        pass
    elif len(index) == 1:
        index = Index(**index[0])
    else:
        index = MultiIndex(
            indexes=[Index(**index_properties) for index_properties in index]
        )
    return DataFrameSchema(
        columns=columns,
        checks=checks,
        index=index,
        dtype=serialized_schema.get("dtype", None),
        coerce=serialized_schema.get("coerce", False),
        strict=serialized_schema.get("strict", False),
        name=serialized_schema.get("name", None),
        ordered=serialized_schema.get("ordered", False),
        unique=serialized_schema.get("unique", None),
        report_duplicates=serialized_schema.get("report_duplicates", "all"),
        unique_column_names=serialized_schema.get(
            "unique_column_names", False
        ),
        add_missing_columns=serialized_schema.get(
            "add_missing_columns", False
        ),
        title=serialized_schema.get("title", None),
        description=serialized_schema.get("description", None),
        drop_invalid_rows=serialized_schema.get("drop_invalid_rows", False),
    )


def from_yaml(yaml_schema):
    """Create :class:`~pandera.api.pandas.container.DataFrameSchema` from yaml file.

    :param yaml_schema: str or Path to yaml schema, or serialized yaml string.
    :returns: dataframe schema.
    """
    try:
        with Path(yaml_schema).open("r", encoding="utf-8") as f:
            serialized_schema = yaml.safe_load(f)
    except (TypeError, OSError):
        serialized_schema = yaml.safe_load(yaml_schema)
    return deserialize_schema(serialized_schema)


def to_yaml(dataframe_schema, stream=None):
    """Write :class:`~pandera.api.pandas.container.DataFrameSchema` to yaml file.

    :param dataframe_schema: schema to write to file or dump to string.
    :param stream: file stream to write to. If None, dumps to string.
    :returns: yaml string if stream is None, otherwise returns None.
    """
    statistics = serialize_schema(dataframe_schema)

    def _write_yaml(obj, stream):
        return yaml.safe_dump(obj, stream=stream, sort_keys=False)

    try:
        with Path(stream).open("w", encoding="utf-8") as f:
            _write_yaml(statistics, f)
    except (TypeError, OSError):
        return _write_yaml(statistics, stream)


def from_json(source):
    """
    Create :class:`~pandera.api.pandas.container.DataFrameSchema` from json file.

    :param source:
        Depending on the type, source is assumed to be:

        1) str or Path to a file containing json schema (if the file exists),
        2) str as a JSON-encoded schema, or
        3) stream that we can read from containing the schema as JSON-encoded
           string.

    :returns: dataframe schema.
    """
    if isinstance(source, str):
        try:
            serialized_schema = json.loads(source)
        except json.decoder.JSONDecodeError:
            with Path(source).open(encoding="utf-8") as f:
                serialized_schema = json.load(fp=f)
    elif isinstance(source, Path):
        with source.open(encoding="utf-8") as f:
            serialized_schema = json.load(fp=f)
    else:
        serialized_schema = json.load(fp=source)

    return deserialize_schema(serialized_schema)


def to_json(dataframe_schema, target=None, **kwargs):
    """
    Write :class:`~pandera.api.pandas.container.DataFrameSchema` to json file.

    :param dataframe_schema: schema to write to file or dump to string.
    :param target: file path or stream to write to. If None, returns a
        dump to string.
    :param kwargs: keyword arguments to pass into :func:`json.dump`
    :returns: json string if stream is None, otherwise returns None.
    """
    serialized_schema = serialize_schema(dataframe_schema)

    if target is None:
        return json.dumps(serialized_schema, sort_keys=False, **kwargs)

    if isinstance(target, (str, Path)):
        with Path(target).open("w", encoding="utf-8") as f:
            json.dump(serialized_schema, fp=f, sort_keys=False, **kwargs)
    else:
        json.dump(serialized_schema, fp=target, sort_keys=False, **kwargs)


SCRIPT_TEMPLATE = """
from pandera import (
    DataFrameSchema, Column, Check, Index, MultiIndex
)

schema = DataFrameSchema(
    columns={{{columns}}},
    checks={checks},
    index={index},
    dtype={dtype},
    coerce={coerce},
    strict={strict},
    name={name},
    ordered={ordered},
    unique={unique},
    report_duplicates={report_duplicates},
    unique_column_names={unique_column_names},
    add_missing_columns={add_missing_columns},
    title={title},
    description={description},
)
"""

COLUMN_TEMPLATE = """
Column(
    dtype={dtype},
    checks={checks},
    nullable={nullable},
    unique={unique},
    coerce={coerce},
    required={required},
    regex={regex},
    description={description},
    title={title},
)
"""

INDEX_TEMPLATE = """
Index(
    dtype={dtype},
    checks={checks},
    nullable={nullable},
    coerce={coerce},
    name={name},
    description={description},
    title={title},
)
"""

MULTIINDEX_TEMPLATE = """
MultiIndex(indexes=[{indexes}])
"""


def _format_checks(checks_dict):
    """Format checks into string representation including options."""
    if checks_dict is None:
        return "None"

    checks = []
    for check_name, check_kwargs in checks_dict.items():
        if check_kwargs is None:
            warnings.warn(
                f"Check {check_name} cannot be serialized. "
                "This check will be ignored"
            )
            continue

        # Handle options separately
        options = (
            check_kwargs.pop("options", {})
            if isinstance(check_kwargs, dict)
            else {}
        )

        # Format main check arguments
        if isinstance(check_kwargs, dict):
            args = ", ".join(
                f"{k}={v.__repr__()}" for k, v in check_kwargs.items()
            )
        else:
            args = check_kwargs.__repr__()

        # Add options to arguments if they exist
        if options:
            if args:
                args += ", "
            args += ", ".join(
                f"{k}={v.__repr__()}" for k, v in options.items()
            )

        checks.append(f"Check.{check_name}({args})")

    return f"[{', '.join(checks)}]"


def _format_index(index_statistics):
    index = []
    for properties in index_statistics:
        dtype = properties.get("dtype")
        description = properties.get("description")
        title = properties.get("title")
        index_code = INDEX_TEMPLATE.format(
            dtype=(None if dtype is None else _get_dtype_string_alias(dtype)),
            checks=(
                "None"
                if properties["checks"] is None
                else _format_checks(properties["checks"])
            ),
            nullable=properties["nullable"],
            coerce=properties["coerce"],
            name=(
                "None"
                if properties["name"] is None
                else f"\"{properties['name']}\""
            ),
            description=(None if description is None else f'"{description}"'),
            title=(None if title is None else f'"{title}"'),
        )
        index.append(index_code.strip())

    if len(index) == 1:
        return index[0]

    return MULTIINDEX_TEMPLATE.format(indexes=",".join(index)).strip()


def _format_script(script):
    formatter = partial(black.format_str, mode=black.FileMode(line_length=80))
    return formatter(script)


def to_script(dataframe_schema, path_or_buf=None):
    """Write :class:`~pandera.api.pandas.container.DataFrameSchema` to a python script.

    :param dataframe_schema: schema to write to file or dump to string.
    :param path_or_buf: filepath or buf stream to write to. If None, outputs
        string representation of the script.
    :returns: yaml string if stream is None, otherwise returns None.
    """
    statistics = get_dataframe_schema_statistics(dataframe_schema)

    columns = {}
    for colname, properties in statistics["columns"].items():
        dtype = properties.get("dtype")
        description = properties["description"]
        title = properties["title"]
        column_code = COLUMN_TEMPLATE.format(
            dtype=(None if dtype is None else _get_dtype_string_alias(dtype)),
            checks=_format_checks(properties["checks"]),
            nullable=properties["nullable"],
            unique=properties["unique"],
            coerce=properties["coerce"],
            required=properties["required"],
            regex=properties["regex"],
            description=(None if description is None else f'"{description}"'),
            title=(None if title is None else f'"{title}"'),
        )
        columns[colname] = column_code.strip()

    index = (
        None
        if statistics["index"] is None
        else _format_index(statistics["index"])
    )

    column_str = ", ".join(f"'{k}': {v}" for k, v in columns.items())

    script = SCRIPT_TEMPLATE.format(
        columns=column_str,
        checks=statistics["checks"],
        index=index,
        dtype=dataframe_schema.dtype,
        coerce=dataframe_schema.coerce,
        strict=dataframe_schema.strict,
        name=dataframe_schema.name.__repr__(),
        ordered=dataframe_schema.ordered,
        unique=dataframe_schema.unique,
        report_duplicates=f'"{dataframe_schema.report_duplicates}"',
        unique_column_names=dataframe_schema.unique_column_names,
        add_missing_columns=dataframe_schema.add_missing_columns,
        title=dataframe_schema.title,
        description=dataframe_schema.description,
    ).strip()

    # add pandas imports to handle datetime and timedelta.
    if "Timedelta" in script:
        script = "from pandas import Timedelta\n" + script
    if "Timestamp" in script:
        script = "from pandas import Timestamp\n" + script

    formatted_script = _format_script(script)

    if path_or_buf is None:
        return formatted_script

    with Path(path_or_buf).open("w", encoding="utf-8") as f:
        f.write(formatted_script)


class FrictionlessFieldParser:
    """Parses frictionless data schema field specifications so we can convert
    them to an equivalent Pandera :class:`~pandera.api.pandas.components.Column`
    schema.

    For this implementation, we are using field names, constraints and types
    but leaving other frictionless parameters out (e.g. foreign keys, type
    formats, titles, descriptions).

    :param field: a field object from a frictionless schema.
    :param primary_keys: the primary keys from a frictionless schema. These
        are used to ensure primary key fields are treated properly - no
        duplicates, no missing values etc.
    """

    def __init__(self, field, primary_keys) -> None:
        self.constraints = field.constraints or {}
        self.primary_keys = primary_keys
        self.name = field.name
        self.type = field.get("type", "string")

    @property
    def dtype(self) -> str:
        """Determine what type of field this is, so we can feed that into
        :class:`~pandera.dtypes.DataType`. If no type is specified in the
        frictionless schema, we default to string values.

        :returns: the pandas-compatible representation of this field type as a
            string.
        """
        types = {
            "string": "string",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "object": "object",
            "array": "object",
            "date": "string",
            "time": "string",
            "datetime": "datetime64[ns]",
            "year": "int",
            "yearmonth": "string",
            "duration": "timedelta64[ns]",
            "geopoint": "object",
            "geojson": "object",
            "any": "string",
        }
        return (
            "category"
            if self.constraints.get("enum", None)
            else types[self.type]
        )

    @property
    def checks(self) -> Optional[Dict]:
        """Convert a set of frictionless schema field constraints into checks.

        This parses the standard set of frictionless constraints which can be
        found
        `here <https://specs.frictionlessdata.io/table-schema/#constraints>`_
        and maps them into the equivalent pandera checks.

        :returns: a dictionary of pandera :class:`~pandera.api.checks.Check`
            objects which capture the standard constraint logic of a
            frictionless schema field.
        """
        if not self.constraints:
            return None
        constraints = self.constraints.copy()
        checks = {}

        def _combine_constraints(check_name, min_constraint, max_constraint):
            """Catches bounded constraints where we need to combine a min and max
            pair of constraints into a single check."""
            if min_constraint in constraints and max_constraint in constraints:
                checks[check_name] = {
                    "min_value": constraints.pop(min_constraint),
                    "max_value": constraints.pop(max_constraint),
                }

        _combine_constraints("in_range", "minimum", "maximum")
        _combine_constraints("str_length", "minLength", "maxLength")

        for constraint_type, constraint_value in constraints.items():
            if constraint_type == "maximum":
                checks["less_than_or_equal_to"] = constraint_value
            elif constraint_type == "minimum":
                checks["greater_than_or_equal_to"] = constraint_value
            elif constraint_type == "maxLength":
                checks["str_length"] = {
                    "min_value": None,
                    "max_value": constraint_value,
                }
            elif constraint_type == "minLength":
                checks["str_length"] = {
                    "min_value": constraint_value,
                    "max_value": None,
                }
            elif constraint_type == "pattern":
                checks["str_matches"] = rf"^{constraint_value}$"
            elif constraint_type == "enum":
                checks["isin"] = constraint_value
        return checks or None

    @property
    def nullable(self) -> bool:
        """Determine whether this field can contain missing values.

        If a field is a primary key, this will return ``False``."""
        if self.name in self.primary_keys:
            return False
        return not self.constraints.get("required", False)

    @property
    def unique(self) -> bool:
        """Determine whether this field can contain duplicate values.

        If a field is a primary key, this will return ``True``.
        """

        # only set column-level uniqueness property if `primary_keys` contains
        # more than one field name.
        if len(self.primary_keys) == 1 and self.name in self.primary_keys:
            return True
        return self.constraints.get("unique", False)

    @property
    def coerce(self) -> bool:
        """Determine whether values within this field should be coerced.

        This currently returns ``True`` for all fields within a frictionless
        schema.
        """
        return True

    @property
    def required(self) -> bool:
        """Determine whether this field must exist within the data.

        This currently returns ``True`` for all fields within a frictionless
        schema.
        """
        return True

    @property
    def regex(self) -> bool:
        """Determine whether this field name should be used for regex matches.

        This currently returns ``False`` for all fields within a frictionless
        schema.
        """
        return False

    def to_pandera_column(self) -> Dict:
        """Export this field to a column spec dictionary."""
        return {
            "checks": self.checks,
            "coerce": self.coerce,
            "nullable": self.nullable,
            "unique": self.unique,
            "dtype": self.dtype,
            "required": self.required,
            "name": self.name,
            "regex": self.regex,
        }


def from_frictionless_schema(
    schema: Union[str, Path, Dict, FrictionlessSchema]
) -> DataFrameSchema:
    # pylint: disable=line-too-long,anomalous-backslash-in-string
    r"""Create a :class:`~pandera.api.pandas.container.DataFrameSchema` from either a
    frictionless json/yaml schema file saved on disk, or from a frictionless
    schema already loaded into memory.

    Each field from the frictionless schema will be converted to a pandera
    column specification using :class:`~pandera.io.pandas_io.FrictionlessFieldParser`
    to map field characteristics to pandera column specifications.

    :param schema: the frictionless schema object (or a
        string/Path to the location on disk of a schema specification) to
        parse.
    :returns: dataframe schema with frictionless field specs converted to
        pandera column checks and constraints for use as normal.

    :example:

    Here, we're defining a very basic frictionless schema in memory before
    parsing it and then querying the resulting
    :class:`~pandera.api.pandas.container.DataFrameSchema` object as per any other Pandera
    schema:

    >>> from pandera.io import from_frictionless_schema
    >>>
    >>> FRICTIONLESS_SCHEMA = {
    ...     "fields": [
    ...         {
    ...             "name": "column_1",
    ...             "type": "integer",
    ...             "constraints": {"minimum": 10, "maximum": 99}
    ...         },
    ...         {
    ...             "name": "column_2",
    ...             "type": "string",
    ...             "constraints": {"maxLength": 10, "pattern": "\\S+"}
    ...         },
    ...     ],
    ...     "primaryKey": "column_1"
    ... }
    >>> schema = from_frictionless_schema(FRICTIONLESS_SCHEMA)
    >>> schema.columns["column_1"].checks
    [<Check in_range: in_range(10, 99)>]
    >>> schema.columns["column_1"].required
    True
    >>> schema.columns["column_1"].unique
    True
    >>> schema.columns["column_2"].checks
    [<Check str_length: str_length(None, 10)>, <Check str_matches: str_matches('^\S+$')>]
    """
    if not isinstance(schema, FrictionlessSchema):
        schema = FrictionlessSchema(schema)

    assembled_schema = {
        "columns": {
            field.name: FrictionlessFieldParser(
                field, schema.primary_key
            ).to_pandera_column()
            for field in schema.fields
        },
        "index": None,
        "checks": None,
        "coerce": True,
        "strict": True,
        # only set dataframe-level uniqueness if the frictionless primary
        # key property specifies more than one field
        "unique": (
            None if len(schema.primary_key) == 1 else list(schema.primary_key)
        ),
    }
    return deserialize_schema(assembled_schema)
