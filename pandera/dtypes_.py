"""Pandera data types."""
# pylint:disable=too-many-ancestors
import dataclasses
from abc import ABC
from typing import (
    Any,
    Callable,
    Iterable,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)


class DataType(ABC):
    """Base class of all Pandera data types."""

    def __init__(self):
        if self.__class__ is DataType:
            raise TypeError(
                f"{self.__class__.__name__} may not be instantiated."
            )

    def coerce(self, data_container: Any):
        """Coerce data container to the dtype."""
        raise NotImplementedError()

    def __call__(self, data_container: Any):
        """Coerce data container to the dtype."""
        return self.coerce(data_container)

    def check(self, pandera_dtype: "DataType") -> bool:
        """Check that pandera :class:`DataType`s are equivalent."""
        if not isinstance(pandera_dtype, DataType):
            return False
        return self == pandera_dtype

    def __repr__(self) -> str:
        return f"DataType({str(self)})"

    def __str__(self) -> str:
        raise NotImplementedError()

    def __hash__(self) -> int:
        raise NotImplementedError()


_Dtype = TypeVar("_Dtype", bound=DataType)
_DataTypeClass = Type[_Dtype]


def immutable(
    pandera_dtype_cls: Optional[_DataTypeClass] = None, **dataclass_kwargs: Any
) -> Union[_DataTypeClass, Callable[[_DataTypeClass], _DataTypeClass]]:
    """:func:`dataclasses.dataclass` decorator with different default values:
    `frozen=True`, `init=False`, `repr=False`.

    In addition, `init=False` disables inherited `__init__` method to ensure
    the DataType's default attributes are not altered during initialization.

    :param dtype: :class:`DataType` to decorate.
    :param dataclass_kwargs: Keywords arguments forwarded to
        :func:`dataclasses.dataclass`.
    :returns: Immutable :class:`~pandera.dtypes.DataType`
    """
    kwargs = {"frozen": True, "init": False, "repr": False}
    kwargs.update(dataclass_kwargs)

    def _wrapper(pandera_dtype_cls: _DataTypeClass) -> _DataTypeClass:
        immutable_dtype = dataclasses.dataclass(**kwargs)(pandera_dtype_cls)
        if not kwargs["init"]:

            def __init__(self):  # pylint:disable=unused-argument
                pass

            # delattr(immutable_dtype, "__init__") doesn't work because
            # super.__init__ would still exist.
            setattr(immutable_dtype, "__init__", __init__)

        return immutable_dtype

    if pandera_dtype_cls is None:
        return _wrapper

    return _wrapper(pandera_dtype_cls)


################################################################################
# boolean
################################################################################


@immutable
class Bool(DataType):
    """Semantic representation of a boolean data type."""

    def __str__(self) -> str:
        return "bool"


Boolean = Bool

################################################################################
# number
################################################################################


@immutable
class _Number(DataType):
    """Semantic representation of a numeric data type."""

    continuous: Optional[bool] = None
    exact: Optional[bool] = None

    def check(self, pandera_dtype: "DataType") -> bool:
        if self.__class__ is _Number:
            return isinstance(pandera_dtype, (Int, Float, Complex))
        return super().check(pandera_dtype)


@immutable
class _PhysicalNumber(_Number):

    bit_width: Optional[int] = None
    _base_name: Optional[str] = dataclasses.field(
        default=None, init=False, repr=False
    )

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, type(self)):
            return obj.bit_width == self.bit_width
        return super().__eq__(obj)

    def __str__(self) -> str:
        return f"{self._base_name}{self.bit_width}"


################################################################################
## signed integer
################################################################################


@immutable(eq=False)
class Int(_PhysicalNumber):  # type: ignore
    """Semantic representation of an integer data type."""

    _base_name = "int"
    continuous = False
    exact = True
    bit_width = 64
    signed: bool = dataclasses.field(default=True, init=False)


@immutable
class Int64(Int, _PhysicalNumber):
    """Semantic representation of an integer data type stored in 64 bits."""

    bit_width = 64


@immutable
class Int32(Int64):
    """Semantic representation of an integer data type stored in 32 bits."""

    bit_width = 32


@immutable
class Int16(Int32):
    """Semantic representation of an integer data type stored in 16 bits."""

    bit_width = 16


@immutable
class Int8(Int16):
    """Semantic representation of an integer data type stored in 8 bits."""

    bit_width = 8


################################################################################
## unsigned integer
################################################################################


@immutable
class UInt(Int):
    """Semantic representation of an unsigned integer data type."""

    _base_name = "uint"
    signed: bool = dataclasses.field(default=False, init=False)


@immutable
class UInt64(UInt):
    """Semantic representation of an unsigned integer data type stored
    in 64 bits."""

    bit_width = 64


@immutable
class UInt32(UInt64):
    """Semantic representation of an unsigned integer data type stored
    in 32 bits."""

    bit_width = 32


@immutable
class UInt16(UInt32):
    """Semantic representation of an unsigned integer data type stored
    in 16 bits."""

    bit_width = 16


@immutable
class UInt8(UInt16):
    """Semantic representation of an unsigned integer data type stored
    in 8 bits."""

    bit_width = 8


################################################################################
## float
################################################################################


@immutable(eq=False)
class Float(_PhysicalNumber):  # type: ignore
    """Semantic representation of a floating data type."""

    _base_name = "float"
    continuous = True
    exact = False
    bit_width = 64


@immutable
class Float128(Float):
    """Semantic representation of a floating data type stored in 128 bits."""

    bit_width = 128


@immutable
class Float64(Float128):
    """Semantic representation of a floating data type stored in 64 bits."""

    bit_width = 64


@immutable
class Float32(Float64):
    """Semantic representation of a floating data type stored in 32 bits."""

    bit_width = 32


@immutable
class Float16(Float32):
    """Semantic representation of a floating data type stored in 16 bits."""

    bit_width = 16


################################################################################
## complex
################################################################################


@immutable(eq=False)
class Complex(_PhysicalNumber):  # type: ignore
    """Semantic representation of a complex number data type."""

    _base_name = "complex"
    bit_width = 128


@immutable
class Complex256(Complex):
    """Semantic representation of a complex number data type stored
    in 256 bits."""

    bit_width = 256


@immutable
class Complex128(Complex):
    """Semantic representation of a complex number data type stored
    in 128 bits."""

    bit_width = 128


@immutable
class Complex64(Complex128):
    """Semantic representation of a complex number data type stored
    in 64 bits."""

    bit_width = 64


################################################################################
# nominal
################################################################################


@immutable(init=True)
class Category(DataType):  # type: ignore
    """Semantic representation of a categorical data type."""

    categories: Optional[Tuple[Any]] = None  # tuple to ensure safe hash
    ordered: bool = False

    def __init__(
        self, categories: Optional[Iterable[Any]] = None, ordered: bool = False
    ):
        # Define __init__ to avoid exposing pylint errors to end users.
        super().__init__()
        if categories is not None and not isinstance(categories, tuple):
            object.__setattr__(self, "categories", tuple(categories))
        object.__setattr__(self, "ordered", ordered)

    def check(self, pandera_dtype: "DataType") -> bool:
        if isinstance(pandera_dtype, Category) and (
            self.categories is None or pandera_dtype.categories is None
        ):
            # Category without categories is a superset of any Category
            # Allow end-users to not list categories when validating.
            return True

        return super().check(pandera_dtype)

    def __str__(self) -> str:
        return "category"


@immutable
class String(DataType):
    """Semantic representation of a string data type."""

    def __str__(self) -> str:
        return "string"


################################################################################
# time
################################################################################


@immutable
class Date(DataType):
    """Semantic representation of a date data type."""

    def __str__(self) -> str:
        return "date"


@immutable
class Timestamp(Date):
    """Semantic representation of a timestamp data type."""

    def __str__(self) -> str:
        return "timestamp"


DateTime = Timestamp


@immutable
class Timedelta(DataType):
    """Semantic representation of a delta time data type."""

    def __str__(self) -> str:
        return "timedelta"
