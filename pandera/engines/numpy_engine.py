import builtins
import datetime
from dataclasses import field
from typing import Any, List

import numpy as np

from .. import dtypes_
from ..dtypes_ import DisableInitMixin, immutable
from . import engine


@immutable(init=True)
class DataType(dtypes_.DataType):
    type: np.dtype = field(default=np.dtype("object"), repr=False)

    def __post_init__(self):
        object.__setattr__(self, "type", np.dtype(self.type))

    def coerce(self, arr: np.ndarray) -> np.ndarray:
        return arr.astype(self.type)

    def __str__(self) -> str:
        return self.type.name

    def __repr__(self) -> str:
        return f"DataType({self})"


class Engine(metaclass=engine.Engine, base_datatype=DataType):
    @classmethod
    def dtype(cls, data_type: Any) -> "DataType":
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            try:
                np_dtype = np.dtype(data_type).type
            except TypeError:
                raise TypeError(
                    f"data type '{data_type}' not understood by {cls.__name__}."
                ) from None
            try:
                return engine.Engine.dtype(cls, np_dtype)
            except TypeError:
                return DataType(data_type)


################################################################################
# boolean
################################################################################


@Engine.register_dtype(
    equivalents=["bool", bool, np.bool_, dtypes_.Bool, dtypes_.Bool()]
)
@immutable
class Bool(DisableInitMixin, DataType, dtypes_.Bool):
    """representation of a boolean data type."""

    type = np.dtype("bool")


def _build_number_equivalents(
    builtin_name: str, pandera_name: str, sizes: List[int]
) -> None:
    """Return a dict of equivalent builtin, numpy, pandera dtypes
    indexed by size in bit_width."""
    builtin_type = getattr(builtins, builtin_name, None)
    default_np_dtype = np.dtype(builtin_name)
    default_size = int(default_np_dtype.name.replace(builtin_name, ""))

    default_equivalents = [
        # e.g.: np.int64
        np.dtype(builtin_name).type,
        # e.g: pandera.dtypes.Int
        getattr(dtypes_, pandera_name),
    ]
    if builtin_type:
        default_equivalents.append(builtin_type)

    return {
        bit_width: set(
            (
                # e.g.: numpy.int64
                getattr(np, f"{builtin_name}{bit_width}"),
                # e.g.: pandera.dtypes.Int64
                getattr(dtypes_, f"{pandera_name}{bit_width}"),
                getattr(dtypes_, f"{pandera_name}{bit_width}")(),
                # e.g.: pandera.dtypes.Int(64)
                getattr(dtypes_, pandera_name)(),
            )
        )
        | set(default_equivalents if bit_width == default_size else [])
        for bit_width in sizes
    }


################################################################################
## signed integer
################################################################################

_int_equivalents = _build_number_equivalents(
    builtin_name="int", pandera_name="Int", sizes=[64, 32, 16, 8]
)


@Engine.register_dtype(equivalents=_int_equivalents[64])
@immutable
class Int64(DisableInitMixin, DataType, dtypes_.Int64):
    type = np.dtype("int64")
    bit_width: int = 64


@Engine.register_dtype(equivalents=_int_equivalents[32])
@immutable
class Int32(Int64):
    type = np.dtype("int32")
    bit_width: int = 32


@Engine.register_dtype(equivalents=_int_equivalents[16])
@immutable
class Int16(Int32):
    type = np.dtype("int16")
    bit_width: int = 16


@Engine.register_dtype(equivalents=_int_equivalents[8])
@immutable
class Int8(Int16):
    type = np.dtype("int8")
    bit_width: int = 8


################################################################################
## unsigned integer
################################################################################

_uint_equivalents = _build_number_equivalents(
    builtin_name="uint",
    pandera_name="UInt",
    sizes=[64, 32, 16, 8],
)


@Engine.register_dtype(equivalents=_uint_equivalents[64])
@immutable
class UInt64(DisableInitMixin, DataType, dtypes_.UInt64):
    type = np.dtype("uint64")
    bit_width: int = 64


@Engine.register_dtype(equivalents=_uint_equivalents[32])
@immutable
class UInt32(UInt64):
    type = np.dtype("uint32")
    bit_width: int = 32


@Engine.register_dtype(equivalents=_uint_equivalents[16])
@immutable
class UInt16(UInt32):
    type = np.dtype("uint16")
    bit_width: int = 16


@Engine.register_dtype(equivalents=_uint_equivalents[8])
@immutable
class UInt8(UInt16):
    type = np.dtype("uint8")
    bit_width: int = 8


################################################################################
## float
################################################################################

_float_equivalents = _build_number_equivalents(
    builtin_name="float",
    pandera_name="Float",
    sizes=[128, 64, 32, 16],
)


@Engine.register_dtype(equivalents=_float_equivalents[128])
@immutable
class Float128(DisableInitMixin, DataType, dtypes_.Float128):
    type = np.dtype("float128")
    bit_width: int = 128


@Engine.register_dtype(equivalents=_float_equivalents[64])
@immutable
class Float64(Float128):
    type = np.dtype("float64")
    bit_width: int = 64


@Engine.register_dtype(equivalents=_float_equivalents[32])
@immutable
class Float32(Float64):
    type = np.dtype("float32")
    bit_width: int = 32


@Engine.register_dtype(equivalents=_float_equivalents[16])
@immutable
class Float16(Float32):
    type = np.dtype("float16")
    bit_width: int = 16


################################################################################
## complex
################################################################################

_complex_equivalents = _build_number_equivalents(
    builtin_name="complex",
    pandera_name="Complex",
    sizes=[256, 128, 64],
)


@Engine.register_dtype(equivalents=_complex_equivalents[256])
@immutable
class Complex256(DisableInitMixin, DataType, dtypes_.Complex256):
    type = np.dtype("complex256")
    bit_width: int = 256


@Engine.register_dtype(equivalents=_complex_equivalents[128])
@immutable
class Complex128(Complex256):
    type = np.dtype("complex128")
    bit_width: int = 128


@Engine.register_dtype(equivalents=_complex_equivalents[64])
@immutable
class Complex64(Complex128):
    type = np.dtype("complex64")
    bit_width: int = 64


################################################################################
# string
################################################################################


@Engine.register_dtype(equivalents=["str", "string", str, np.str_])
@immutable
class String(DisableInitMixin, DataType, dtypes_.String):
    type = np.dtype("str")

    def coerce(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(object)
        notna = ~arr.isna()
        arr[notna] = arr[notna].astype(str)
        return arr

    def check(self, datatype: "dtypes_.DataType") -> bool:
        return isinstance(datatype, (Object, type(self)))


################################################################################
# object
################################################################################


@Engine.register_dtype(equivalents=["object", "O", object, np.object_])
@immutable
class Object(DisableInitMixin, DataType):
    type = np.dtype("object")


Object = Object

################################################################################
# time
################################################################################


@Engine.register_dtype(
    equivalents=[
        datetime.datetime,
        np.datetime64,
        dtypes_.Timestamp,
        dtypes_.Timestamp(),
    ]
)
@immutable
class DateTime64(DisableInitMixin, DataType, dtypes_.Timestamp):
    type = np.dtype("datetime64")


@Engine.register_dtype(
    equivalents=[
        datetime.datetime,
        np.timedelta64,
        dtypes_.Timedelta,
        dtypes_.Timedelta(),
    ]
)
@immutable
class Timedelta64(DisableInitMixin, DataType, dtypes_.Timedelta):
    type = np.dtype("timedelta64")
