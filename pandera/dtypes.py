# pylint: disable=line-too-long
"""Schema datatypes."""

from enum import Enum


class PandasDtype(Enum):
    """Enumerate all valid pandas data types.

    ``pandera`` follows the
    `numpy data types <https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#arrays-dtypes>`_
    subscribed to by ``pandas`` and by default supports using the numpy data
    type string aliases to validate DataFrame or Series dtypes.

    This class simply enumerates the valid numpy dtypes for pandas arrays.
    For convenience ``PandasDtype`` enums can all be accessed in the top-level
    ``pandera`` name space via the same enum name.

    :examples:

    >>> import pandas as pd
    >>> import pandera as pa
    >>>
    >>>
    >>> pa.SeriesSchema(pa.Int).validate(pd.Series([1, 2, 3]))
    0    1
    1    2
    2    3
    dtype: int64
    >>> pa.SeriesSchema(pa.Float).validate(pd.Series([1.1, 2.3, 3.4]))
    0    1.1
    1    2.3
    2    3.4
    dtype: float64
    >>> pa.SeriesSchema(pa.String).validate(pd.Series(["a", "b", "c"]))
        0    a
    1    b
    2    c
    dtype: object

    You can also directly use the string alias for each data-type in the
    schema definition:

    >>> pa.SeriesSchema("int").validate(pd.Series([1, 2, 3]))
    0    1
    1    2
    2    3
    dtype: int64

    .. note::
        ``pandera`` also offers limited support for
        `pandas extension types <https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes>`_,
        however since the release of pandas 1.0.0 there are backwards
        incompatible extension types like the nullable ``Integer`` array and
        the dedicated ``String`` array. In theory the string aliases for these
        extension types should work when supplied to the ``pandas_dtype``
        argument when initializing ``pa.SeriesSchemaBase`` objects, but this
        is not currently tested.
    """

    Bool = "bool"  #: ``"bool"`` numpy dtype
    DateTime = "datetime64[ns]" #: ``"datetime64[ns]"`` numpy dtype
    Category = "category" #: pandas ``"categorical"`` datatype
    Float = "float"  #: ``"float"`` numpy dtype
    Float16 = "float16"  #: ``"float16"`` numpy dtype
    Float32 = "float32"  #: ``"float32"`` numpy dtype
    Float64 = "float64"  #: ``"float64"`` numpy dtype
    Int = "int"  #: ``"int"`` numpy dtype
    Int8 = "int8"  #: ``"int8"`` numpy dtype
    Int16 = "int16"  #: ``"int16"`` numpy dtype
    Int32 = "int32"  #: ``"int32"`` numpy dtype
    Int64 = "int64"  #: ``"int64"`` numpy dtype
    UInt8 = "uint8"  #: ``"uint8"`` numpy dtype
    UInt16 = "uint16"  #: ``"uint16"`` numpy dtype
    UInt32 = "uint32"  #: ``"uint32"`` numpy dtype
    UInt64 = "uint64"  #: ``"uint64"`` numpy dtype
    Object = "object"  #: ``"object"`` numpy dtype

    #: The string datatype doesn't map to a first-class pandas datatype and is
    #: representated as a numpy ``"object"`` array. This will change after
    #: pandera explicitly supports pandas 1.0+ and is currently handled
    #: internally by pandera as a special case.
    String = "string"
    Timedelta = "timedelta64[ns]"  #: ``"timedelta64[ns]"`` numpy dtype


NUMPY_INT_DTYPES = [
    "int", "int_", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
]
