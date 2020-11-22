# pylint: disable=no-member,too-many-public-methods
"""Schema datatypes."""

from enum import Enum

import numpy as np
import pandas as pd
from packaging import version

PandasExtensionType = pd.core.dtypes.base.ExtensionDtype

LEGACY_PANDAS = version.parse(pd.__version__).major < 1  # type: ignore
NUMPY_NONNULLABLE_INT_DTYPES = [
    "int",
    "int_",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
]

NUMPY_TYPES = frozenset(
    [item for sublist in np.sctypes.values() for item in sublist]
).union(frozenset([np.complex, np.int, np.uint, np.float]))

# for int and float dtype, delegate string representation to the
# default based on OS. In Windows, pandas defaults to int64 while numpy
# defaults to int32.
_DEFAULT_PANDAS_INT_TYPE = str(pd.Series([1]).dtype)
_DEFAULT_PANDAS_FLOAT_TYPE = str(pd.Series([1.0]).dtype)
_DEFAULT_PANDAS_COMPLEX_TYPE = str(pd.Series([complex(1)]).dtype)
_DEFAULT_NUMPY_INT_TYPE = str(np.dtype(int))
_DEFAULT_NUMPY_FLOAT_TYPE = str(np.dtype(float))


class PandasDtype(Enum):
    # pylint: disable=line-too-long
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

    Alternatively, you can use built-in python scalar types for integers,
    floats, booleans, and strings:

    >>> pa.SeriesSchema(int).validate(pd.Series([1, 2, 3]))
    0    1
    1    2
    2    3
    dtype: int64

    You can also use the pandas string aliases in the schema definition:

    >>> pa.SeriesSchema("int").validate(pd.Series([1, 2, 3]))
    0    1
    1    2
    2    3
    dtype: int64

    .. note::
        ``pandera`` also offers limited support for
        `pandas extension types <https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#dtypes>`_,
        however since the release of pandas 1.0.0 there are backwards
        incompatible extension types like the ``Integer`` array. The extension
        types, e.g. ``pd.IntDtype64()`` and their string alias should work
        when supplied to the ``pandas_dtype`` argument, unless otherwise
        specified below, but this functionality is only tested for
        pandas >= 1.0.0. Extension types in earlier versions are not guaranteed
        to work as the ``pandas_dtype`` argument in schemas or schema
        components.

    """

    Bool = "bool"  #: ``"bool"`` numpy dtype
    DateTime = "datetime64[ns]"  #: ``"datetime64[ns]"`` numpy dtype
    Timedelta = "timedelta64[ns]"  #: ``"timedelta64[ns]"`` numpy dtype
    Category = "category"  #: pandas ``"categorical"`` datatype
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
    INT8 = "Int8"  #: ``"Int8"`` pandas dtype:: pandas 0.24.0+
    INT16 = "Int16"  #: ``"Int16"`` pandas dtype: pandas 0.24.0+
    INT32 = "Int32"  #: ``"Int32"`` pandas dtype: pandas 0.24.0+
    INT64 = "Int64"  #: ``"Int64"`` pandas dtype: pandas 0.24.0+
    UINT8 = "UInt8"  #: ``"UInt8"`` pandas dtype: pandas 0.24.0+
    UINT16 = "UInt16"  #: ``"UInt16"`` pandas dtype: pandas 0.24.0+
    UINT32 = "UInt32"  #: ``"UInt32"`` pandas dtype: pandas 0.24.0+
    UINT64 = "UInt64"  #: ``"UInt64"`` pandas dtype: pandas 0.24.0+
    Object = "object"  #: ``"object"`` numpy dtype
    Complex = "complex"  #: ``"complex"`` numpy dtype
    Complex64 = "complex64"  #: ``"complex"`` numpy dtype
    Complex128 = "complex128"  #: ``"complex"`` numpy dtype
    Complex256 = "complex256"  #: ``"complex"`` numpy dtype
    Str = "str"  #: ``"str"`` numpy dtype

    #: ``"string"`` pandas dtypes: pandas 1.0.0+. For <1.0.0, this enum will
    #: fall back on the str-as-object-array representation.
    String = "string"

    @property
    def str_alias(self):
        """Get datatype string alias."""
        return {
            "int": _DEFAULT_PANDAS_INT_TYPE,
            "float": _DEFAULT_PANDAS_FLOAT_TYPE,
            "complex": _DEFAULT_PANDAS_COMPLEX_TYPE,
            "str": "object",
            "string": "object" if LEGACY_PANDAS else "string",
        }.get(self.value, self.value)

    @classmethod
    def from_str_alias(cls, str_alias: str) -> "PandasDtype":
        """Get PandasDtype from string alias.

        :param: pandas dtype string alias from
            https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#basics-dtypes
        :returns: pandas dtype
        """
        pandas_dtype = {
            "bool": cls.Bool,
            "datetime64[ns]": cls.DateTime,
            "timedelta64[ns]": cls.Timedelta,
            "category": cls.Category,
            "float": cls.Float,
            "float16": cls.Float16,
            "float32": cls.Float32,
            "float64": cls.Float64,
            "int": cls.Int,
            "int8": cls.Int8,
            "int16": cls.Int16,
            "int32": cls.Int32,
            "int64": cls.Int64,
            "uint8": cls.UInt8,
            "uint16": cls.UInt16,
            "uint32": cls.UInt32,
            "uint64": cls.UInt64,
            "Int8": cls.INT8,
            "Int16": cls.INT16,
            "Int32": cls.INT32,
            "Int64": cls.INT64,
            "UInt8": cls.UINT8,
            "UInt16": cls.UINT16,
            "UInt32": cls.UINT32,
            "UInt64": cls.UINT64,
            "object": cls.Object,
            "complex": cls.Complex,
            "complex64": cls.Complex64,
            "complex128": cls.Complex128,
            "complex256": cls.Complex256,
            "str": cls.Str,
            "string": cls.Str if LEGACY_PANDAS else cls.String,
        }.get(str_alias)

        if pandas_dtype is None:
            raise TypeError(
                "pandas dtype string alias '%s' not recognized" % str_alias
            )

        return pandas_dtype

    @classmethod
    def from_pandas_api_type(cls, pandas_api_type: str) -> "PandasDtype":
        """Get PandasDtype enum from pandas api type.

        :param pandas_api_type: string output from
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.infer_dtype.html
        :returns: pandas dtype
        """
        if pandas_api_type.startswith("mixed"):
            return cls.Object

        pandas_dtype = {
            "string": cls.Str,
            "floating": cls.Float,
            "integer": cls.Int,
            "categorical": cls.Category,
            "boolean": cls.Bool,
            "datetime64": cls.DateTime,
            "datetime": cls.DateTime,
            "timedelta64": cls.Timedelta,
            "timedelta": cls.Timedelta,
        }.get(pandas_api_type)

        if pandas_dtype is None:
            raise TypeError(
                "pandas api type '%s' not recognized" % pandas_api_type
            )

        return pandas_dtype

    @classmethod
    def from_python_type(cls, python_type: type) -> "PandasDtype":
        """Get PandasDtype enum from built-in python type.

        :param python_type: built-in python type. Allowable types are:
            str, int, float, and bool.
        """
        pandas_dtype = {
            bool: cls.Bool,
            str: cls.Str,
            int: cls.Int,
            float: cls.Float,
            object: cls.Object,
            complex: cls.Complex,
        }.get(python_type)

        if pandas_dtype is None:
            raise TypeError(
                "python type '%s' not recognized as pandas data type"
                % python_type
            )

        return pandas_dtype

    @classmethod
    def from_numpy_type(cls, numpy_type: np.dtype) -> "PandasDtype":
        """Get PandasDtype enum from numpy type.

        :param numpy_type: numpy data type.
        """
        return cls.from_str_alias(numpy_type.__name__)

    def __eq__(self, other):
        # pylint: disable=comparison-with-callable
        # see https://github.com/PyCQA/pylint/issues/2306
        if other is None:
            return False
        if isinstance(other, str):
            other = self.from_str_alias(other)
        if self.value == "string" and LEGACY_PANDAS:
            return PandasDtype.Str.value == other.value
        elif self.value == "string":
            return self.value == other.value
        return self.str_alias == other.str_alias

    def __hash__(self):
        if self is PandasDtype.Int:
            hash_obj = _DEFAULT_PANDAS_INT_TYPE
        elif self is PandasDtype.Float:
            hash_obj = _DEFAULT_PANDAS_FLOAT_TYPE
        else:
            hash_obj = self.str_alias
        return id(hash_obj)

    @property
    def numpy_dtype(self):
        """Get numpy data type."""
        if self is PandasDtype.Category:
            raise TypeError(
                "the pandas Categorical data type doesn't have a numpy "
                "equivalent."
            )

        dtype = {"string": np.dtype("str")}.get(self.value)

        # pylint: disable=comparison-with-callable
        if self.value == "string":
            dtype = np.dtype("str")
        else:
            dtype = np.dtype(self.value.lower())
        return dtype

    @property
    def is_int(self) -> bool:
        """Return True if PandasDtype is an integer."""
        return self.value.lower().startswith("int")

    @property
    def is_nullable_int(self) -> bool:
        """Return True if PandasDtype is a nullable integer."""
        return self.value.startswith("Int")

    @property
    def is_nonnullable_int(self) -> bool:
        """Return True if PandasDtype is a non-nullable integer."""
        return self.value.startswith("int")

    @property
    def is_uint(self) -> bool:
        """Return True if PandasDtype is an unsigned integer."""
        return self.value.lower().startswith("uint")

    @property
    def is_nullable_uint(self) -> bool:
        """Return True if PandasDtype is a nullable unsigned integer."""
        return self.value.startswith("UInt")

    @property
    def is_nonnullable_uint(self) -> bool:
        """Return True if PandasDtype is a non-nullable unsigned integer."""
        return self.value.startswith("uint")

    @property
    def is_float(self) -> bool:
        """Return True if PandasDtype is a float."""
        return self.value.startswith("float")

    @property
    def is_complex(self) -> bool:
        """Return True if PandasDtype is a complex number."""
        return self.value.startswith("complex")

    @property
    def is_bool(self) -> bool:
        """Return True if PandasDtype is a boolean."""
        return self is PandasDtype.Bool

    @property
    def is_string(self) -> bool:
        """Return True if PandasDtype is a string."""
        return self in [PandasDtype.Str, PandasDtype.String]

    @property
    def is_category(self) -> bool:
        """Return True if PandasDtype is a category."""
        return self is PandasDtype.Category

    @property
    def is_datetime(self) -> bool:
        """Return True if PandasDtype is a datetime."""
        return self is PandasDtype.DateTime

    @property
    def is_timedelta(self) -> bool:
        """Return True if PandasDtype is a timedelta."""
        return self is PandasDtype.Timedelta

    @property
    def is_object(self) -> bool:
        """Return True if PandasDtype is an object."""
        return self is PandasDtype.Object

    @property
    def is_continuous(self) -> bool:
        """Return True if PandasDtype is a continuous datatype."""
        return (
            self.is_int
            or self.is_uint
            or self.is_float
            or self.is_complex
            or self.is_datetime
            or self.is_timedelta
        )
