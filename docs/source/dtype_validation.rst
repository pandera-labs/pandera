.. currentmodule:: pandera

.. _dtype_validation:

Data Type Validation
====================

The core utility of ``pandera`` is that it allows you to validate the types of
incoming raw data so that your data pipeline can fail early and not propagate
data corruption downstream to critical applications. These applications may
include analytics, statistical, and machine learning use cases that rely on
clean data for them to be valid.


How can I specify data types?
-----------------------------

With pandera schemas, there are multiple ways of specifying the data types of
columns, indexes, or even whole dataframes.

.. testcode:: dtype_validation

    import pandera as pa
    import pandas as pd

    # schema with datatypes at the column and index level
    schema_field_dtypes = pa.DataFrameSchema(
        {
            "column1": pa.Column(int),
            "column2": pa.Column(float),
            "column3": pa.Column(str),
        },
        index = pa.Index(int),
    )

    # schema with datatypes at the dataframe level, if all columns are the
    # same data type
    schema_df_dtypes = pa.DataFrameSchema(dtype=int)


The equivalent :py:class:`~pandera.api.pandas.model.DataFrameModel` would be:

.. testcode:: dtype_validation

    from pandera.typing import Series, Index

    class ModelFieldDtypes(pa.DataFrameModel):
        column1: Series[int]
        column2: Series[float]
        column3: Series[str]
        index: Index[int]

    class ModelDFDtypes(pa.DataFrameModel):
        class Config:
            dtype = int


Supported pandas datatypes
--------------------------

By default, pandera supports the validation of pandas dataframes, so pandera
schemas support any of the `data types <https://pandas.pydata.org/docs/user_guide/basics.html#dtypes>`__
that pandas supports:

- Built-in python types, e.g. ``int``, ``float``, ``str``, ``bool``, etc.
- `Numpy data types <https://numpy.org/doc/stable/user/basics.types.html>`__, e.g. ``numpy.int_``, ``numpy.bool__``, etc.
- Pandas-native data types, e.g. ``pd.StringDtype``, ``pd.BooleanDtype``, ``pd.DatetimeTZDtype``, etc.
- Any of the `string aliases <https://pandas.pydata.org/docs/user_guide/basics.html#dtypes>`__ supported by pandas.

We recommend using the built-in python datatypes for the common data types, but
it's really up to you to figure out how you want to express these types.
Additionally, you can use also the :ref:`pandera-defined datatypes <api-dtypes>`
if you want.

For example, the following schema expresses the equivalent integer types in
six different ways:

.. testcode:: dtype_validation

    import numpy as np

    integer_schema = pa.DataFrameSchema(
        {
            "builtin_python": pa.Column(int),
            "builtin_python": pa.Column("int"),
            "string_alias": pa.Column("int64"),
            "numpy_dtype": pa.Column(np.int64),
            "pandera_dtype": pa.Column(pa.Int),
            "pandera_dtype": pa.Column(pa.Int64),
        },
    )

.. note:: The default ``int`` type for Windows is 32-bit integers ``int32``.


Parameterized data types
------------------------

One thing to be aware of is the difference between declaring pure Python types
(i.e. classes) as the data type of a column vs parameterized types, which in
the case of pandas, are actually instances of special classes defined by pandas.
For example, using the object-based API, we can easily define a column as a
timezone-aware datatype:

.. testcode:: dtype_validation

    datetimeschema = pa.DataFrameSchema({
        "dt": pa.Column(pd.DatetimeTZDtype(unit="ns", tz="UTC"))
    })

However, since python's type annotations require types and not objects, to
express this same type with the class-based API, we need to use an
:py:class:`~typing.Annotated` type:

.. testcode:: dtype_validation

    try:
        from typing import Annotated  # python 3.9+
    except ImportError:
        from typing_extensions import Annotated

    class DateTimeModel(pa.DataFrameModel):
        dt: Series[Annotated[pd.DatetimeTZDtype, "ns", "UTC"]]

Or alternatively, you can pass in the ``dtype_kwargs`` into
:py:func:`~pandera.api.dataframe.model_components.Field`:

.. testcode:: dtype_validation

    class DateTimeModel(pa.DataFrameModel):
        dt: Series[pd.DatetimeTZDtype] = pa.Field(dtype_kwargs={"unit": "ns", "tz": "UTC"})

You can read more about the supported parameterized data types
:ref:`here <parameterized dtypes>`.


Data type coercion
------------------

Pandera is primarily a *validation* library: it only checks the schema metadata
or data values of the dataframe without changing anything about the dataframe
itself.

However, in many cases its useful to *parse*, i.e. transform the data values
to the data contract specified in the pandera schema. Currently, the only
transformation pandera does is type coercion, which can be done by passing in
the ``coerce=True`` argument to the schema or schema component objects:

- :py:class:`~pandera.api.pandas.components.Column`
- :py:class:`~pandera.api.pandas.components.Index`
- :py:class:`~pandera.api.pandas.components.MultiIndex`
- :py:class:`~pandera.api.pandas.container.DataFrameSchema`
- :py:class:`~pandera.api.pandas.arrays.SeriesSchema`

If this argument is provided, instead of simply checking the columns/index(es)
for the correct types, calling ``schema.validate`` will attempt to coerce the
incoming dataframe values into the specified data types.

It will then apply the dataframe-, column-, and index-level checks to the
data, all of which are purely *validators*.


.. _how_nullable_works:

How data types interact with ``nullable``
------------------------------------------

The ``nullable`` argument, which can be specified at the column-, index, or
``SeriesSchema``-level, is essentially a core pandera check. As such, it is
applied after the data type check/coercion step described in the previous
section. Therefore, datatypes that are inherently not nullable will fail even
if you specify ``nullable=True`` because pandera considers type checks a
first-class check that's distinct from any downstream check that you may want
to apply to the data.


Support for the python ``typing`` module
----------------------------------------

*new in 0.15.0*

Pandera also supports a limited set of generic and special types :py:mod:`typing`
for you to validate columns containing ``object`` values:

- ``typing.Dict[K, V]``
- ``typing.List[T]``
- ``typing.Tuple[T, ...]``
- ``typing.TypedDict``
- ``typing.NamedTuple``

For example:

.. testcode:: dtype_validation

    from typing import Dict, List, Tuple, NamedTuple

    if sys.version_info >= (3, 12):
        from typing import TypedDict
        # use typing_extensions.TypedDict for python < 3.9 in order to support
        # run-time availability of optional/required fields
    else:
        from typing_extensions import TypedDict


    class PointDict(TypedDict):
        x: float
        y: float

    class PointTuple(NamedTuple):
        x: float
        y: float

    schema = pa.DataFrameSchema(
        {
            "dict_column": pa.Column(Dict[str, int]),
            "list_column": pa.Column(List[float]),
            "tuple_column": pa.Column(Tuple[int, str, float]),
            "typeddict_column": pa.Column(PointDict),
            "namedtuple_column": pa.Column(PointTuple),
        },
    )

    data = pd.DataFrame({
        "dict_column": [{"foo": 1, "bar": 2}],
        "list_column": [[1.0]],
        "tuple_column": [(1, "bar", 1.0)],
        "typeddict_column": [PointDict(x=2.1, y=4.8)],
        "namedtuple_column": [PointTuple(x=9.2, y=1.6)],
    })

    schema.validate(data)

Pandera uses `typeguard <https://typeguard.readthedocs.io/en/latest/>`__ for
data type validation and `pydantic <https://docs.pydantic.dev/latest/>` for
data value coercion, in the case that you've specified ``coerce=True`` at the
column-, index-, or dataframe-level.

.. note::

    For certain types like ``List[T]``, ``typeguard`` will only check the type
    of the first value, e.g. if you specify ``List[int]``, a data value of
    ``[1, "foo", 1.0]`` will still pass. Checking all values will be
    configurable in future  versions of pandera when ``typeguard > 4.*.*`` is
    supported.
