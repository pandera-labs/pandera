```{eval-rst}
.. currentmodule:: pandera
```

(lazy-validation)=

# Lazy Validation

*New in version 0.4.0*

By default, when you call the `validate` method on schema or schema component
objects, a {class}`~pandera.errors.SchemaError` is raised as soon as one of the
assumptions specified in the schema is falsified. For example, for a
{class}`~pandera.api.pandas.container.DataFrameSchema` object, the following situations will raise an
exception:

- a column specified in the schema is not present in the dataframe.
- if `strict=True`, a column in the dataframe is not specified in the schema.
- the `data type` does not match.
- if `coerce=True`, the dataframe column cannot be coerced into the specified
  `data type`.
- the {class}`~pandera.api.checks.Check` specified in one of the columns returns `False` or
  a boolean series containing at least one `False` value.

For example:

```{eval-rst}
.. testcode:: non_lazy_validation

   import pandas as pd
   import pandera as pa

   from pandera import Check, Column, DataFrameSchema

   df = pd.DataFrame({"column": ["a", "b", "c"]})

   schema = pa.DataFrameSchema({"column": Column(int)})
   schema.validate(df)
```

```{eval-rst}
.. testoutput:: non_lazy_validation

    Traceback (most recent call last):
    ...
    SchemaError: expected series 'column' to have type int64, got object

```

For more complex cases, it is useful to see all of the errors raised during
the `validate` call so that you can debug the causes of errors on different
columns and checks. The `lazy` keyword argument in the `validate` method
of all schemas and schema components gives you the option of doing just this:

```{eval-rst}
.. testcode:: lazy_validation
    :skipif: SKIP_PANDAS_LT_V1

    import pandas as pd
    import pandera as pa

    from pandera import Check, Column, DataFrameSchema

    schema = pa.DataFrameSchema(
        columns={
            "int_column": Column(int),
            "float_column": Column(float, Check.greater_than(0)),
            "str_column": Column(str, Check.equal_to("a")),
            "date_column": Column(pa.DateTime),
        },
        strict=True
    )

    df = pd.DataFrame({
        "int_column": ["a", "b", "c"],
        "float_column": [0, 1, 2],
        "str_column": ["a", "b", "d"],
        "unknown_column": None,
    })

    schema.validate(df, lazy=True)
```

```{eval-rst}
.. testoutput:: lazy_validation
    :skipif: SKIP_PANDAS_LT_V1

    Traceback (most recent call last):
    ...

    SchemaErrors: {
        "SCHEMA": {
            "COLUMN_NOT_IN_SCHEMA": [
                {
                    "schema": null,
                    "column": null,
                    "check": "column_in_schema",
                    "error": "column 'unknown_column' not in DataFrameSchema {'int_column': <Schema Column(name=int_column, type=DataType(int64))>, 'float_column': <Schema Column(name=float_column, type=DataType(float64))>, 'str_column': <Schema Column(name=str_column, type=DataType(str))>, 'date_column': <Schema Column(name=date_column, type=DataType(datetime64[ns]))>}"
                }
            ],
            "COLUMN_NOT_IN_DATAFRAME": [
                {
                    "schema": null,
                    "column": null,
                    "check": "column_in_dataframe",
                    "error": "column 'date_column' not in dataframe. Columns in dataframe: ['int_column', 'float_column', 'str_column', 'unknown_column']"
                }
            ],
            "WRONG_DATATYPE": [
                {
                    "schema": null,
                    "column": "int_column",
                    "check": "dtype('int64')",
                    "error": "expected series 'int_column' to have type int64, got object"
                },
                {
                    "schema": null,
                    "column": "float_column",
                    "check": "dtype('float64')",
                    "error": "expected series 'float_column' to have type float64, got int64"
                }
            ]
        },
        "DATA": {
            "DATAFRAME_CHECK": [
                {
                    "schema": null,
                    "column": "float_column",
                    "check": "greater_than(0)",
                    "error": "Column 'float_column' failed element-wise validator number 0: greater_than(0) failure cases: 0"
                },
                {
                    "schema": null,
                    "column": "str_column",
                    "check": "equal_to(a)",
                    "error": "Column 'str_column' failed element-wise validator number 0: equal_to(a) failure cases: b, d"
                }
            ]
        }
    }
```

As you can see from the output above, a {class}`~pandera.errors.SchemaErrors`
exception is raised with a summary of the error counts and failure cases
caught by the schema. This summary is called an {ref}`error-report`.

You can also inspect the failure cases in a more granular form:

```{eval-rst}
.. testcode:: lazy_validation
    :skipif: SKIP_PANDAS_LT_V1_OR_GT_V2

    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("Schema errors and failure cases:")
        print(err.failure_cases)
        print("\nDataFrame object that failed validation:")
        print(err.data)
```

```{eval-rst}
.. testoutput:: lazy_validation
    :skipif: SKIP_PANDAS_LT_V1_OR_GT_V2

    Schema errors and failure cases:
        schema_context        column                check check_number  \
    0  DataFrameSchema          None     column_in_schema         None
    1  DataFrameSchema          None  column_in_dataframe         None
    2           Column    int_column       dtype('int64')         None
    3           Column  float_column     dtype('float64')         None
    4           Column  float_column      greater_than(0)            0
    5           Column    str_column          equal_to(a)            0
    6           Column    str_column          equal_to(a)            0

         failure_case index
    0  unknown_column  None
    1     date_column  None
    2          object  None
    3           int64  None
    4               0     0
    5               b     1
    6               d     2

    DataFrame object that failed validation:
      int_column  float_column str_column unknown_column
    0          a             0          a           None
    1          b             1          b           None
    2          c             2          d           None
```
