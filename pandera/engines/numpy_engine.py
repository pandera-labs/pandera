"""Numpy engine and data types."""
# docstrings are inherited
# pylint:disable=missing-class-docstring,too-many-ancestors
import builtins
import dataclasses
import datetime
import inspect
import warnings
from typing import Any, Dict, List, Union

import numpy as np

from .. import dtypes, errors
from ..dtypes import immutable
from ..system import MAC_M1_PLATFORM, WINDOWS_PLATFORM
from . import engine, utils
from .type_aliases import PandasObject


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing Numpy data types."""

    type: np.dtype = dataclasses.field(
        default=np.dtype("object"), repr=False, init=False
    )
    """Native numpy dtype boxed by the data type."""

    def __init__(self, dtype: Any):
        super().__init__()
        object.__setattr__(self, "type", np.dtype(dtype))
        dtype_cls = dtype if inspect.isclass(dtype) else dtype.__class__
        warnings.warn(
            f"'{dtype_cls}' support is not guaranteed.\n"
            + "Usage Tip: Consider writing a custom "
            + "pandera.dtypes.DataType or opening an issue at "
            + "https://github.com/pandera-dev/pandera"
        )

    def __post_init__(self):
        # this method isn't called if __init__ is defined
        object.__setattr__(
            self, "type", np.dtype(self.type)
        )  # pragma: no cover

    def coerce(
        self, data_container: Union[PandasObject, np.ndarray]
    ) -> Union[PandasObject, np.ndarray]:
        try:
            return data_container.astype(self.type)
        except (ValueError, TypeError) as exc:
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}",
                failure_cases=utils.numpy_pandas_coerce_failure_cases(
                    data_container, self.type
                ),
            ) from exc

    def __str__(self) -> str:
        return self.type.name

    def __repr__(self) -> str:
        return f"DataType({self})"


class Engine(  # pylint:disable=too-few-public-methods
    metaclass=engine.Engine, base_pandera_dtypes=DataType
):
    """Numpy data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        """Convert input into a numpy-compatible
        Pandera :class:`~pandera.dtypes.DataType` object."""
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            try:
                np_dtype = np.dtype(data_type).type
            except TypeError:
                raise TypeError(
                    f"data type '{data_type}' not understood by "
                    f"{cls.__name__}."
                ) from None

            try:
                return engine.Engine.dtype(cls, np_dtype)
            except TypeError:
                return DataType(data_type)


###############################################################################
# boolean
###############################################################################


@Engine.register_dtype(
    equivalents=["bool", bool, np.bool_, dtypes.Bool, dtypes.Bool()]
)
@immutable
class Bool(DataType, dtypes.Bool):
    type = np.dtype("bool")


def _build_number_equivalents(
    builtin_name: str, pandera_name: str, sizes: List[int]
) -> Dict[int, List[Union[type, str, np.dtype, dtypes.DataType]]]:
    """Return a dict of equivalent builtin, numpy, pandera dtypes
    indexed by size in bit_width."""
    builtin_type = getattr(builtins, builtin_name, None)
    default_np_dtype = np.dtype(builtin_name)
    default_size = int(default_np_dtype.name.replace(builtin_name, ""))

    default_equivalents = [
        # e.g.: np.int64
        np.dtype(builtin_name).type,
        # e.g: pandera.dtypes.Int
        getattr(dtypes, pandera_name),
    ]
    if builtin_type:
        default_equivalents.append(builtin_type)

    return {
        bit_width: list(
            set(
                (
                    # e.g.: numpy.int64
                    getattr(np, f"{builtin_name}{bit_width}"),
                    # e.g.: pandera.dtypes.Int64
                    getattr(dtypes, f"{pandera_name}{bit_width}"),
                    getattr(dtypes, f"{pandera_name}{bit_width}")(),
                    # e.g.: pandera.dtypes.Int(64)
                    getattr(dtypes, pandera_name)(),
                )
            )
            | set(default_equivalents if bit_width == default_size else [])
        )
        for bit_width in sizes
    }


###############################################################################
# signed integer
###############################################################################

_int_equivalents = _build_number_equivalents(
    builtin_name="int", pandera_name="Int", sizes=[64, 32, 16, 8]
)


@Engine.register_dtype(equivalents=_int_equivalents[64])
@immutable
class Int64(DataType, dtypes.Int64):

    type = np.dtype("int64")
    bit_width: int = 64


@Engine.register_dtype(equivalents=_int_equivalents[32])
@immutable
class Int32(Int64):
    type = np.dtype("int32")  # type: ignore
    bit_width: int = 32


@Engine.register_dtype(equivalents=_int_equivalents[16])
@immutable
class Int16(Int32):
    type = np.dtype("int16")  # type: ignore
    bit_width: int = 16


@Engine.register_dtype(equivalents=_int_equivalents[8])
@immutable
class Int8(Int16):
    type = np.dtype("int8")  # type: ignore
    bit_width: int = 8


###############################################################################
# unsigned integer
###############################################################################

_uint_equivalents = _build_number_equivalents(
    builtin_name="uint",
    pandera_name="UInt",
    sizes=[64, 32, 16, 8],
)


@Engine.register_dtype(equivalents=_uint_equivalents[64])
@immutable
class UInt64(DataType, dtypes.UInt64):
    type = np.dtype("uint64")
    bit_width: int = 64


@Engine.register_dtype(equivalents=_uint_equivalents[32])
@immutable
class UInt32(UInt64):
    type = np.dtype("uint32")  # type: ignore
    bit_width: int = 32


@Engine.register_dtype(equivalents=_uint_equivalents[16])
@immutable
class UInt16(UInt32):
    type = np.dtype("uint16")  # type: ignore
    bit_width: int = 16


@Engine.register_dtype(equivalents=_uint_equivalents[8])
@immutable
class UInt8(UInt16):
    type = np.dtype("uint8")  # type: ignore
    bit_width: int = 8


###############################################################################
# float
###############################################################################

_float_equivalents = _build_number_equivalents(
    builtin_name="float",
    pandera_name="Float",
    sizes=(
        [64, 32, 16]
        if WINDOWS_PLATFORM or MAC_M1_PLATFORM
        else [128, 64, 32, 16]
    ),
)


if not WINDOWS_PLATFORM or MAC_M1_PLATFORM:
    # not supported in windows
    # https://github.com/winpython/winpython/issues/613
    @Engine.register_dtype(equivalents=_float_equivalents[128])
    @immutable
    class Float128(DataType, dtypes.Float128):
        type = np.dtype("float128")
        bit_width: int = 128

    @Engine.register_dtype(equivalents=_float_equivalents[64])
    @immutable
    class Float64(Float128):
        type = np.dtype("float64")
        bit_width: int = 64


else:

    @Engine.register_dtype(equivalents=_float_equivalents[64])
    @immutable
    class Float64(DataType, dtypes.Float64):  # type: ignore
        type = np.dtype("float64")
        bit_width: int = 64


@Engine.register_dtype(equivalents=_float_equivalents[32])
@immutable
class Float32(Float64):
    type = np.dtype("float32")  # type: ignore
    bit_width: int = 32


@Engine.register_dtype(equivalents=_float_equivalents[16])
@immutable
class Float16(Float32):
    type = np.dtype("float16")  # type: ignore
    bit_width: int = 16


###############################################################################
# complex
###############################################################################

_complex_equivalents = _build_number_equivalents(
    builtin_name="complex",
    pandera_name="Complex",
    sizes=[128, 64] if WINDOWS_PLATFORM or MAC_M1_PLATFORM else [256, 128, 64],
)


if not WINDOWS_PLATFORM or MAC_M1_PLATFORM:
    # not supported in windows
    # https://github.com/winpython/winpython/issues/613
    @Engine.register_dtype(equivalents=_complex_equivalents[256])
    @immutable
    class Complex256(DataType, dtypes.Complex256):
        type = np.dtype("complex256")
        bit_width: int = 256

    @Engine.register_dtype(equivalents=_complex_equivalents[128])
    @immutable
    class Complex128(Complex256):
        type = np.dtype("complex128")  # type: ignore
        bit_width: int = 128


else:

    @Engine.register_dtype(equivalents=_complex_equivalents[128])
    @immutable
    class Complex128(DataType, dtypes.Complex128):  # type: ignore
        type = np.dtype("complex128")  # type: ignore
        bit_width: int = 128


@Engine.register_dtype(equivalents=_complex_equivalents[64])
@immutable
class Complex64(Complex128):
    type = np.dtype("complex64")  # type: ignore
    bit_width: int = 64


###############################################################################
# string
###############################################################################


@Engine.register_dtype(equivalents=["str", "string", str, np.str_])
@immutable
class String(DataType, dtypes.String):
    type = np.dtype("str")

    def coerce(self, data_container: np.ndarray) -> np.ndarray:
        data_container = data_container.astype(object)
        try:
            notna = ~np.isnan(data_container)
        except TypeError:
            notna = np.ones_like(data_container).astype(bool)
        data_container[notna] = data_container[notna].astype(str)
        return data_container

    def check(self, pandera_dtype: "dtypes.DataType") -> bool:
        return isinstance(pandera_dtype, (Object, type(self)))


###############################################################################
# object
###############################################################################


@Engine.register_dtype(equivalents=["object", "O", object, np.object_])
@immutable
class Object(DataType):
    """Semantic representation of a :class:`numpy.object_`."""

    type = np.dtype("object")


###############################################################################
# time
###############################################################################


@Engine.register_dtype(
    equivalents=[
        datetime.datetime,
        np.datetime64,
        dtypes.Timestamp,
        dtypes.Timestamp(),
    ]
)
@immutable
class DateTime64(DataType, dtypes.Timestamp):
    type = np.dtype("datetime64")


@Engine.register_dtype(
    equivalents=[
        datetime.datetime,
        np.timedelta64,
        dtypes.Timedelta,
        dtypes.Timedelta(),
    ]
)
@immutable
class Timedelta64(DataType, dtypes.Timedelta):
    type = np.dtype("timedelta64[ns]")
