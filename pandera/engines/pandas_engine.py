"""Pandas engine and data types."""

# pylint:disable=too-many-ancestors

# docstrings are inherited
# pylint:disable=missing-class-docstring

# pylint doesn't know about __init__ generated with dataclass
# pylint:disable=unexpected-keyword-arg,no-value-for-parameter
import builtins
import dataclasses
import datetime
import decimal
import inspect
import sys
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import numpy as np
import pandas as pd
import typeguard
from pydantic import BaseModel, ValidationError, create_model

from pandera import dtypes, errors
from pandera.dtypes import immutable
from pandera.engines import PYDANTIC_V2, engine, numpy_engine, utils
from pandera.engines.type_aliases import (
    PandasDataType,
    PandasExtensionType,
    PandasObject,
)
from pandera.engines.utils import pandas_version
from pandera.system import FLOAT_128_AVAILABLE

if PYDANTIC_V2:
    from pydantic import RootModel

try:
    import pyarrow  # pylint: disable=unused-import

    PYARROW_INSTALLED = True
except ImportError:
    PYARROW_INSTALLED = False

try:
    from typeguard import CollectionCheckStrategy

    # This may be worth making configurable at the global level.
    type_types_kwargs = {
        "collection_check_strategy": CollectionCheckStrategy.ALL_ITEMS,
    }
    TYPEGUARD_COLLECTION_STRATEGY_AVAILABLE = True
    TYPEGUARD_ERROR = typeguard.TypeCheckError
except ImportError:
    warnings.warn(
        "Using typeguard < 3. Generic types like List[TYPE], Dict[TYPE, TYPE] "
        "will only validate the first element in the collection.",
        UserWarning,
    )
    type_types_kwargs = {}
    TYPEGUARD_COLLECTION_STRATEGY_AVAILABLE = False
    TYPEGUARD_ERROR = TypeError


PANDAS_1_2_0_PLUS = pandas_version().release >= (1, 2, 0)
PANDAS_1_3_0_PLUS = pandas_version().release >= (1, 3, 0)
PANDAS_2_0_0_PLUS = pandas_version().release >= (2, 0, 0)


# register different TypedDict type depending on python version
if sys.version_info >= (3, 12):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict  # noqa


try:
    # python 3.8+
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore[assignment]


def is_extension_dtype(
    pd_dtype: PandasDataType,
) -> Union[bool, Iterable[bool]]:
    """Check if a value is a pandas extension type or instance of one."""
    return isinstance(pd_dtype, PandasExtensionType) or (
        isinstance(pd_dtype, type)
        and issubclass(pd_dtype, PandasExtensionType)
    )


def is_pyarrow_dtype(
    pd_dtype: PandasDataType,
) -> Union[bool, Iterable[bool]]:
    """Check if a value is a pandas pyarrow type or instance of one."""
    if not (PYARROW_INSTALLED and PANDAS_2_0_0_PLUS):
        return False

    return isinstance(pd_dtype, pd.ArrowDtype)


def is_geopandas_dtype(
    pd_dtype: Union[PandasDataType, str],
) -> Union[bool, Iterable[bool]]:
    """Check if a value is a geopandas extension type or instance of one."""
    try:
        from geopandas.array import GeometryDtype
    except ImportError:
        return False

    if pd_dtype == "geometry":
        return True
    return pd_dtype is GeometryDtype


@immutable(init=True)
class DataType(dtypes.DataType):
    """Base `DataType` for boxing Pandas data types."""

    type: Any = dataclasses.field(repr=False, init=False)
    """Native pandas dtype boxed by the data type."""

    def __init__(self, dtype: Any):
        super().__init__()
        try:
            type_ = pd.api.types.pandas_dtype(dtype)
        except TypeError:
            type_ = pd.api.types.pandas_dtype(object)

        object.__setattr__(self, "type", type_)
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
            self, "type", pd.api.types.pandas_dtype(self.type)
        )  # pragma: no cover

    def coerce(self, data_container: PandasObject) -> PandasObject:
        """Pure coerce without catching exceptions."""
        coerced = data_container.astype(self.type)
        if type(data_container).__module__.startswith("modin.pandas"):
            # NOTE: this is a hack to enable catching of errors in modin
            coerced.__str__()
        return coerced

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to a particular type."""
        # by default, the pandas Engine delegates to the underlying numpy
        # datatype to coerce a value to the correct type.
        return self.type.type(value)

    def try_coerce(self, data_container: PandasObject) -> PandasObject:
        try:
            coerced = self.coerce(data_container)
            if type(data_container).__module__.startswith("modin.pandas"):
                # NOTE: this is a hack to enable catching of errors in modin
                coerced.__str__()
        except Exception as exc:  # pylint:disable=broad-except
            if isinstance(exc, errors.ParserError):
                raise
            if self.type != np.dtype("object") and self != numpy_engine.Object:
                type_alias = self.type
            else:
                type_alias = str(self)
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {type_alias}",
                failure_cases=utils.numpy_pandas_coerce_failure_cases(
                    data_container, self
                ),
            ) from exc

        return coerced

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Optional[PandasObject] = None,
    ) -> Union[bool, Iterable[bool]]:
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
        except TypeError:
            return False

        # attempts to compare pandas native type if possible
        # to let subclass inherit check
        # (super will compare that DataType classes are exactly the same)
        try:
            return self.type == pandera_dtype.type or super().check(
                pandera_dtype
            )
        except TypeError:
            return super().check(pandera_dtype)

    def __str__(self) -> str:
        return str(self.type)

    def __repr__(self) -> str:
        return f"DataType({self})"


class Engine(  # pylint:disable=too-few-public-methods
    metaclass=engine.Engine,
    base_pandera_dtypes=(DataType, numpy_engine.DataType),
):
    """Pandas data type engine."""

    @classmethod
    def dtype(cls, data_type: Any) -> dtypes.DataType:
        """Convert input into a pandas-compatible
        Pandera :class:`~pandera.dtypes.DataType` object."""
        try:
            return engine.Engine.dtype(cls, data_type)
        except TypeError:
            if is_geopandas_dtype(data_type):
                # pylint: disable=cyclic-import,unused-import
                # register geopandas datatypes
                import pandera.engines.geopandas_engine

                np_or_pd_dtype = data_type
            elif is_pyarrow_dtype(data_type):
                # pylint: disable=cyclic-import
                # register pyarrow datatypes
                import pandera.engines.pyarrow_engine

                np_or_pd_dtype = data_type.pyarrow_dtype
            elif is_extension_dtype(data_type) and isinstance(data_type, type):
                try:
                    np_or_pd_dtype = data_type()
                    # Convert to str here because some pandas dtypes allow
                    # an empty constructor for compatibility but fail on
                    # str(). e.g: PeriodDtype
                    str(np_or_pd_dtype.name)
                except (TypeError, AttributeError) as err:
                    raise TypeError(
                        f" dtype {data_type} cannot be instantiated: {err}\n"
                        "Usage Tip: Use an instance or a string "
                        "representation."
                    ) from None
            else:
                # let pandas transform any acceptable value
                # into a numpy or pandas dtype.
                np_or_pd_dtype = pd.api.types.pandas_dtype(data_type)
                if isinstance(np_or_pd_dtype, np.dtype):
                    # cast alias to platform-agnostic dtype
                    # e.g.: np.intc -> np.int32
                    common_np_dtype = np.dtype(np_or_pd_dtype.name)
                    np_or_pd_dtype = common_np_dtype.type

            return engine.Engine.dtype(cls, np_or_pd_dtype)

    @classmethod
    def numpy_dtype(cls, pandera_dtype: dtypes.DataType) -> np.dtype:
        """Convert a Pandera :class:`~pandera.dtypes.DataType
        to a :class:`numpy.dtype`."""
        pandera_dtype: dtypes.DataType = engine.Engine.dtype(
            cls, pandera_dtype
        )

        alias = str(pandera_dtype).lower()
        if alias == "boolean":
            alias = "bool"
        elif alias.startswith("string"):
            alias = "str"

        try:
            return np.dtype(alias)
        except TypeError as err:
            raise TypeError(
                f"Data type '{pandera_dtype}' cannot be cast to a numpy dtype."
            ) from err


###############################################################################
# boolean
###############################################################################


Engine.register_dtype(
    numpy_engine.Bool,
    equivalents=["bool", bool, np.bool_, dtypes.Bool, dtypes.Bool()],
)


@Engine.register_dtype(
    equivalents=["boolean", pd.BooleanDtype, pd.BooleanDtype()],
)
@immutable
class BOOL(DataType, dtypes.Bool):
    """Semantic representation of a :class:`pandas.BooleanDtype`."""

    type = pd.BooleanDtype()
    _bool_like = frozenset({True, False})

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to specified datatime type."""
        if value not in self._bool_like:
            raise TypeError(
                f"value {value} cannot be coerced to type {self.type}"
            )
        return super().coerce_value(value)


###############################################################################
# number
###############################################################################


def _register_numpy_numbers(
    builtin_name: str, pandera_name: str, sizes: List[int]
) -> None:
    """Register pandera.engines.numpy_engine DataTypes
    with the pandas engine."""

    builtin_type = getattr(builtins, builtin_name, None)  # uint doesn't exist

    # default to int64 regardless of OS
    default_pd_dtype = {
        "int": np.dtype("int64"),
        "uint": np.dtype("uint64"),
    }.get(builtin_name, pd.Series([1], dtype=builtin_name).dtype)

    for bit_width in sizes:
        # e.g.: numpy.int64
        np_dtype = getattr(np, f"{builtin_name}{bit_width}")

        equivalents = set(
            (
                np_dtype,
                # e.g.: pandera.dtypes.Int64
                getattr(dtypes, f"{pandera_name}{bit_width}"),
                getattr(dtypes, f"{pandera_name}{bit_width}")(),
            )
        )

        if np_dtype == default_pd_dtype:
            equivalents |= set(
                (
                    default_pd_dtype,
                    builtin_name,
                    getattr(dtypes, pandera_name),
                    getattr(dtypes, pandera_name)(),
                )
            )
            if builtin_type:
                equivalents.add(builtin_type)

            # results from pd.api.types.infer_dtype
            if builtin_type is float:
                equivalents.add("floating")
                equivalents.add("mixed-integer-float")
            elif builtin_type is int:
                equivalents.add("integer")

        numpy_data_type = getattr(numpy_engine, f"{pandera_name}{bit_width}")
        Engine.register_dtype(numpy_data_type, equivalents=list(equivalents))


###############################################################################
# signed integer
###############################################################################

_register_numpy_numbers(
    builtin_name="int",
    pandera_name="Int",
    sizes=[64, 32, 16, 8],
)


@Engine.register_dtype(equivalents=[pd.Int64Dtype, pd.Int64Dtype()])
@immutable
class INT64(DataType, dtypes.Int):
    """Semantic representation of a :class:`pandas.Int64Dtype`."""

    type = pd.Int64Dtype()  # type: ignore[assignment]
    bit_width: int = 64


@Engine.register_dtype(equivalents=[pd.Int32Dtype, pd.Int32Dtype()])
@immutable
class INT32(INT64):
    """Semantic representation of a :class:`pandas.Int32Dtype`."""

    type = pd.Int32Dtype()  # type: ignore[assignment]
    bit_width: int = 32


@Engine.register_dtype(equivalents=[pd.Int16Dtype, pd.Int16Dtype()])
@immutable
class INT16(INT32):
    """Semantic representation of a :class:`pandas.Int16Dtype`."""

    type = pd.Int16Dtype()  # type: ignore[assignment]
    bit_width: int = 16


@Engine.register_dtype(equivalents=[pd.Int8Dtype, pd.Int8Dtype()])
@immutable
class INT8(INT16):
    """Semantic representation of a :class:`pandas.Int8Dtype`."""

    type = pd.Int8Dtype()  # type: ignore[assignment]
    bit_width: int = 8


###############################################################################
# unsigned integer
###############################################################################

_register_numpy_numbers(
    builtin_name="uint",
    pandera_name="UInt",
    sizes=[64, 32, 16, 8],
)


@Engine.register_dtype(equivalents=[pd.UInt64Dtype, pd.UInt64Dtype()])
@immutable
class UINT64(DataType, dtypes.UInt):
    """Semantic representation of a :class:`pandas.UInt64Dtype`."""

    type = pd.UInt64Dtype()
    bit_width: int = 64


@Engine.register_dtype(equivalents=[pd.UInt32Dtype, pd.UInt32Dtype()])
@immutable
class UINT32(UINT64):
    """Semantic representation of a :class:`pandas.UInt32Dtype`."""

    type = pd.UInt32Dtype()  # type: ignore[assignment]
    bit_width: int = 32


@Engine.register_dtype(equivalents=[pd.UInt16Dtype, pd.UInt16Dtype()])
@immutable
class UINT16(UINT32):
    """Semantic representation of a :class:`pandas.UInt16Dtype`."""

    type = pd.UInt16Dtype()  # type: ignore[assignment]
    bit_width: int = 16


@Engine.register_dtype(equivalents=[pd.UInt8Dtype, pd.UInt8Dtype()])
@immutable
class UINT8(UINT16):
    """Semantic representation of a :class:`pandas.UInt8Dtype`."""

    type = pd.UInt8Dtype()  # type: ignore[assignment]
    bit_width: int = 8


# ###############################################################################
# # float
# ###############################################################################

_register_numpy_numbers(
    builtin_name="float",
    pandera_name="Float",
    sizes=[128, 64, 32, 16] if FLOAT_128_AVAILABLE else [64, 32, 16],
)

if PANDAS_1_2_0_PLUS:

    @Engine.register_dtype(equivalents=[pd.Float64Dtype, pd.Float64Dtype()])
    @immutable
    class FLOAT64(DataType, dtypes.Float):
        """Semantic representation of a :class:`pandas.Float64Dtype`."""

        type = pd.Float64Dtype()
        bit_width: int = 64

    @Engine.register_dtype(equivalents=[pd.Float32Dtype, pd.Float32Dtype()])
    @immutable
    class FLOAT32(FLOAT64):
        """Semantic representation of a :class:`pandas.Float32Dtype`."""

        type = pd.Float32Dtype()  # type: ignore[assignment]
        bit_width: int = 32


# ###############################################################################
# # complex
# ###############################################################################

_register_numpy_numbers(
    builtin_name="complex",
    pandera_name="Complex",
    sizes=[256, 128, 64] if FLOAT_128_AVAILABLE else [128, 64],
)
###############################################################################
# decimal
###############################################################################


def _check_decimal(
    pandas_obj: pd.Series,
    precision: Optional[int] = None,
    scale: Optional[int] = None,
) -> pd.Series:
    series_cls = type(pandas_obj)  # support non-pandas series (modin, etc.)
    if pandas_obj.isnull().all():
        return series_cls(np.full_like(pandas_obj, True), dtype=np.bool_)

    is_decimal = pandas_obj.apply(
        lambda x: isinstance(x, decimal.Decimal)
    ).astype("bool") | pd.isnull(pandas_obj)
    if not is_decimal.any():
        return is_decimal

    decimals = pandas_obj[is_decimal]
    # fix for modin unamed series raises KeyError
    # https://github.com/modin-project/modin/issues/4317
    decimals.name = "decimals"  # type: ignore

    splitted = decimals.astype("string").str.split(".", n=1, expand=True)  # type: ignore
    if splitted.shape[1] < 2:
        splitted[1] = ""
    len_left = splitted[0].str.len().fillna(0)
    len_right = splitted[1].str.len().fillna(0)
    precisions = len_left + len_right

    scales = series_cls(
        np.full_like(decimals, np.nan), dtype=np.object_, index=decimals.index  # type: ignore
    )
    pos_left = len_left > 0
    scales[pos_left] = len_right[pos_left]
    scales[~pos_left] = 0

    is_valid = is_decimal
    if precision is not None:
        is_valid &= precisions <= precision
    if scale is not None:
        is_valid &= scales <= scale
    return is_valid


@Engine.register_dtype(
    equivalents=["decimal", decimal.Decimal, dtypes.Decimal]
)
@immutable(init=True)
class Decimal(DataType, dtypes.Decimal):
    # pylint:disable=line-too-long
    """Semantic representation of a :class:`decimal.Decimal`.


    .. note:: :class:`decimal.Decimal` is especially useful when exporting a pandas
        DataFrame to parquet files via `pyarrow <https://arrow.apache.org/docs/python/parquet.html>`_.
        Pyarrow will automatically convert the decimal objects contained in the `object`
        series to the corresponding `parquet Decimal type <https://github.com/apache/parquet-format/blob/master/LogicalTypes.md#decimal>`_.
    """

    type = np.dtype("object")
    rounding: str = dataclasses.field(
        default_factory=lambda: decimal.getcontext().rounding
    )
    """
    The `rounding mode <https://docs.python.org/3/library/decimal.html#rounding-modes>`__
    supported by the Python :py:class:`decimal.Decimal` class.
    """

    _exp: decimal.Decimal = dataclasses.field(init=False)
    _ctx: decimal.Context = dataclasses.field(init=False)

    def __init__(  # pylint:disable=super-init-not-called
        self,
        precision: int = dtypes.DEFAULT_PYTHON_PREC,
        scale: int = 0,
        rounding: Optional[str] = None,
    ) -> None:
        dtypes.Decimal.__init__(self, precision, scale, rounding)

    def coerce_value(self, value: Any) -> decimal.Decimal:
        """Coerce a value to a particular type."""

        if pd.isna(value):
            return cast(decimal.Decimal, pd.NA)

        dec = decimal.Decimal(str(value))
        return dec.quantize(self._exp, context=self._ctx)

    def coerce(self, data_container: PandasObject) -> PandasObject:
        return data_container.apply(self.coerce_value)  # type: ignore

    def check(  # type: ignore
        self,
        pandera_dtype: DataType,
        data_container: Optional[pd.Series] = None,
    ) -> Union[bool, Iterable[bool]]:
        if type(data_container).__module__.startswith("pyspark.pandas"):
            raise NotImplementedError(
                "Decimal is not yet supported for pyspark."
            )
        if not super().check(pandera_dtype, data_container):
            if data_container is None:
                return False
            else:
                return np.full_like(data_container, False, dtype=bool)
        if data_container is None:
            return True
        return _check_decimal(
            data_container, precision=self.precision, scale=self.scale
        )

    def __str__(self) -> str:
        return dtypes.Decimal.__str__(self)


# ###############################################################################
# # nominal
# ###############################################################################


@Engine.register_dtype(
    equivalents=[
        "category",
        "categorical",
        dtypes.Category,
        pd.CategoricalDtype,
    ]
)
@immutable(init=True)
class Category(DataType, dtypes.Category):
    """Semantic representation of a :class:`pandas.CategoricalDtype`."""

    type: pd.CategoricalDtype = dataclasses.field(default=None, init=False)  # type: ignore[assignment]  # noqa

    def __init__(  # pylint:disable=super-init-not-called
        self, categories: Optional[Iterable[Any]] = None, ordered: bool = False
    ) -> None:
        dtypes.Category.__init__(self, categories, ordered)
        object.__setattr__(
            self,
            "type",
            pd.CategoricalDtype(self.categories, self.ordered),  # type: ignore
        )

    def coerce(self, data_container: PandasObject) -> PandasObject:
        """Pure coerce without catching exceptions."""
        coerced = data_container.astype(self.type)
        if (coerced.isna() & data_container.notna()).any(axis=None):  # type: ignore[arg-type]
            raise TypeError(
                f"Data container cannot be coerced to type {self.type}"
            )
        return coerced

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to a particular type."""
        if value not in self.categories:  # type: ignore
            raise TypeError(
                f"value {value} cannot be coerced to type {self.type}"
            )
        return value

    @classmethod
    def from_parametrized_dtype(
        cls, cat: Union[dtypes.Category, pd.CategoricalDtype]
    ):
        """Convert a categorical to
        a Pandera :class:`pandera.dtypes.pandas_engine.Category`."""
        return cls(categories=cat.categories, ordered=cat.ordered)  # type: ignore
    
    def __str__(self) -> str:
        return repr(self.type)


if PANDAS_1_3_0_PLUS:

    @Engine.register_dtype(equivalents=["string", pd.StringDtype])
    @immutable(init=True)
    class STRING(DataType, dtypes.String):
        """Semantic representation of a :class:`pandas.StringDtype`."""

        type: pd.StringDtype = dataclasses.field(default=None, init=False)  # type: ignore[assignment]
        storage: Optional[Literal["python", "pyarrow"]] = "python"

        def __post_init__(self):
            if self.storage == "pyarrow" and not PYARROW_INSTALLED:
                raise ModuleNotFoundError(
                    "pyarrow needs to be installed when using the "
                    "string[pyarrow] pandas data type. Please "
                    "`pip install pyarrow` or "
                    "`conda install -c conda-forge pyarrow` before proceeding."
                )
            type_ = pd.StringDtype(self.storage)
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(cls, pd_dtype: pd.StringDtype):
            """Convert a :class:`pandas.StringDtype` to
            a Pandera :class:`pandera.engines.pandas_engine.STRING`."""
            return cls(pd_dtype.storage)  # type: ignore[attr-defined]

        def __str__(self) -> str:
            return repr(self.type)

else:

    @Engine.register_dtype(
        equivalents=["string", pd.StringDtype, pd.StringDtype()]  # type: ignore
    )  # type: ignore[no-redef] # python 3.7
    @immutable
    class STRING(DataType, dtypes.String):  # type: ignore[no-redef] # python 3.8+
        """Semantic representation of a :class:`pandas.StringDtype`."""

        type = pd.StringDtype()  # type: ignore


@Engine.register_dtype(
    equivalents=["str", str, dtypes.String, dtypes.String(), np.str_]
)
@immutable
class NpString(numpy_engine.String):
    """Specializes numpy_engine.String.coerce to handle pd.NA values."""

    def coerce(
        self,
        data_container: Union[PandasObject, np.ndarray],
    ) -> Union[PandasObject, np.ndarray]:
        def _to_str(obj):
            # NOTE: this is a hack to handle the following case:
            # pyspark.pandas.Index doesn't support .where method yet, use numpy
            reverter = None
            if type(obj).__module__.startswith("pyspark.pandas"):
                # pylint: disable=import-outside-toplevel
                import pyspark.pandas as ps

                if isinstance(obj, ps.Index):
                    obj = obj.to_series()
                    reverter = ps.Index
            else:
                obj = obj.astype(object)

            obj = (
                obj.astype(str)
                if obj.notna().all(axis=None)
                else obj.where(obj.isna(), obj.astype(str))
            )
            return obj if reverter is None else reverter(obj)

        return _to_str(data_container)

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Optional[PandasObject] = None,
    ) -> Union[bool, Iterable[bool]]:
        if data_container is None:
            return isinstance(pandera_dtype, (numpy_engine.Object, type(self)))

        # NOTE: this is a hack to handle the following case:
        # pyspark.pandas doesn't support types with a Series of type object
        if type(data_container).__module__.startswith("pyspark.pandas"):
            is_python_string = data_container.map(lambda x: str(type(x))).isin(  # type: ignore[operator]
                ["<class 'str'>", "<class 'numpy.str_'>"]
            )
        else:
            is_python_string = data_container.map(lambda x: isinstance(x, str))  # type: ignore[operator]
        return is_python_string.astype(bool) | data_container.isna()


Engine.register_dtype(
    numpy_engine.Object,
    equivalents=[
        "object",
        "object_",
        "object0",
        "O",
        "bytes",
        "mixed-integer",
        "mixed",
        "bytes",
        bytes,
        object,
        np.object_,
        np.bytes_,
    ],
)

# ###############################################################################
# # time
# ###############################################################################


_PandasDatetime = Union[np.datetime64, pd.DatetimeTZDtype]


@immutable(init=True)
class _BaseDateTime(DataType):
    to_datetime_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )

    @staticmethod
    def _get_to_datetime_fn(obj: Any) -> Callable:
        # NOTE: this is a hack to support pyspark.pandas. This needs to be
        # thoroughly tested, right now pyspark.pandas returns NA when a
        # dtype value can't be coerced into the target dtype.
        to_datetime_fn = pd.to_datetime
        if type(obj).__module__.startswith(
            "pyspark.pandas"
        ):  # pragma: no cover
            # pylint: disable=import-outside-toplevel
            import pyspark.pandas as ps

            to_datetime_fn = ps.to_datetime
        if type(obj).__module__.startswith("modin.pandas"):
            # pylint: disable=import-outside-toplevel
            import modin.pandas as mpd

            to_datetime_fn = mpd.to_datetime

        return to_datetime_fn


@Engine.register_dtype(
    equivalents=[
        "time",
        "datetime",
        "datetime64",
        datetime.datetime,
        np.datetime64,
        dtypes.Timestamp,
        dtypes.Timestamp(),
        pd.Timestamp,
    ]
)
@immutable(init=True)
class DateTime(_BaseDateTime, dtypes.Timestamp):
    """Semantic representation of a potentially timezone-aware datetime.

    Uses ``np.dtype("datetime64[ns]")`` for non-timezone aware datetimes and
    :class:`pandas.DatetimeTZDtype` for timezone-aware datetimes.
    """

    type: Optional[_PandasDatetime] = dataclasses.field(
        default=None, init=False
    )
    unit: str = "ns"
    """The precision of the datetime data. Currently limited to "ns"."""

    tz: Optional[datetime.tzinfo] = None
    """The timezone."""

    to_datetime_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )
    "Any additional kwargs passed to :func:`pandas.to_datetime` for coercion."

    tz_localize_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )
    "Keyword arguments passed to :func:`pandas.Series.dt.tz_localize` for coercion."

    _default_tz_localize_kwargs = {
        "ambiguous": "infer",
    }

    def __post_init__(self):
        if self.tz is None:
            type_ = np.dtype("datetime64[ns]")
        else:
            type_ = pd.DatetimeTZDtype(self.unit, self.tz)
            # DatetimeTZDtype converted tz to tzinfo for us
            object.__setattr__(self, "tz", type_.tz)

        object.__setattr__(self, "type", type_)

    def _coerce(
        self, data_container: PandasObject, pandas_dtype: Any
    ) -> PandasObject:
        to_datetime_fn = self._get_to_datetime_fn(data_container)
        _tz_localize_kwargs = {
            **self._default_tz_localize_kwargs,
            **self.tz_localize_kwargs,
        }

        def _to_datetime(col: PandasObject) -> PandasObject:
            col = to_datetime_fn(col, **self.to_datetime_kwargs)
            pdtype_tz = getattr(pandas_dtype, "tz", None)
            coltype_tz = getattr(col.dtype, "tz", None)
            if pdtype_tz is not None or coltype_tz is not None:
                if hasattr(col, "dt"):
                    if col.dt.tz is None:
                        # localize datetime column so that it's timezone-aware
                        col = col.dt.tz_localize(
                            pdtype_tz,
                            **_tz_localize_kwargs,
                        )
                    else:
                        col = col.dt.tz_convert(pdtype_tz)
                elif (
                    hasattr(col, "tz")
                    and col.tz != pdtype_tz
                    and hasattr(col, "tz_localize")
                ):
                    if col.tz is None:
                        # localize datetime index so that it's timezone-aware
                        col = col.tz_localize(
                            pdtype_tz,
                            **_tz_localize_kwargs,
                        )
                    else:
                        col = col.tz_convert(pdtype_tz)
            return col.astype(pandas_dtype)

        if isinstance(data_container, pd.DataFrame):
            # pd.to_datetime transforms a df input into a series.
            # We actually want to coerce every columns.
            return data_container.transform(_to_datetime)

        return _to_datetime(data_container)

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.DatetimeTZDtype):
        """Convert a :class:`pandas.DatetimeTZDtype` to
        a Pandera :class:`pandera.engines.pandas_engine.DateTime`."""
        return cls(unit=pd_dtype.unit, tz=pd_dtype.tz)  # type: ignore

    def coerce(self, data_container: PandasObject) -> PandasObject:
        return self._coerce(data_container, pandas_dtype=self.type)

    def coerce_value(self, value: Any) -> Any:
        """Coerce an value to specified datatime type."""
        return self._get_to_datetime_fn(value)(
            value, **self.to_datetime_kwargs
        )

    def __str__(self) -> str:
        if self.type == np.dtype("datetime64[ns]"):
            return "datetime64[ns]"
        return str(self.type)


@Engine.register_dtype(
    equivalents=[
        "date",
        datetime.date,
        dtypes.Date,
        dtypes.Date(),
    ]
)
@immutable(init=True)
class Date(_BaseDateTime, dtypes.Date):
    """Semantic representation of a date data type."""

    type = np.dtype("object")

    to_datetime_kwargs: Dict[str, Any] = dataclasses.field(
        default_factory=dict, compare=False, repr=False
    )
    "Any additional kwargs passed to :func:`pandas.to_datetime` for coercion."

    # define __init__ to please mypy
    def __init__(  # pylint:disable=super-init-not-called
        self,
        to_datetime_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        object.__setattr__(
            self, "to_datetime_kwargs", to_datetime_kwargs or {}
        )

    def _coerce(
        self, data_container: PandasObject, pandas_dtype: Any
    ) -> PandasObject:
        to_datetime_fn = self._get_to_datetime_fn(data_container)

        def _to_datetime(col: PandasObject) -> PandasObject:
            col = to_datetime_fn(col, **self.to_datetime_kwargs)
            return col.astype(pandas_dtype).dt.date

        if isinstance(data_container, pd.DataFrame):
            # pd.to_datetime transforms a df input into a series.
            # We actually want to coerce every columns.
            return data_container.transform(_to_datetime)

        return _to_datetime(data_container)

    def coerce(self, data_container: PandasObject) -> PandasObject:
        return self._coerce(data_container, pandas_dtype="datetime64[ns]")

    def coerce_value(self, value: Any) -> Any:
        coerced = self._get_to_datetime_fn(value)(
            value, **self.to_datetime_kwargs
        )
        return coerced.date() if coerced is not None else pd.NaT

    def check(  # type: ignore
        self,
        pandera_dtype: DataType,
        data_container: Optional[pd.Series] = None,
    ) -> Union[bool, Iterable[bool]]:
        if not DataType.check(self, pandera_dtype, data_container):
            if data_container is None:
                return False
            else:
                return np.full_like(data_container, False)
        if data_container is None:
            return True

        def _check_date(value: Any) -> bool:
            return pd.isnull(value) or (
                type(value) is datetime.date  # pylint:disable=C0123
            )

        return data_container.apply(_check_date)

    def __str__(self) -> str:
        return str(dtypes.Date())


Engine.register_dtype(
    numpy_engine.Timedelta64,
    equivalents=[
        "timedelta",
        "timedelta64",
        datetime.timedelta,
        np.timedelta64,
        pd.Timedelta,
        dtypes.Timedelta,
        dtypes.Timedelta(),
    ],
)


@Engine.register_dtype
@immutable(init=True)
class Period(DataType):
    """Representation of pandas :class:`pd.Period`."""

    type: pd.PeriodDtype = dataclasses.field(default=None, init=False)  # type: ignore[assignment]  # noqa
    freq: Union[str, pd.tseries.offsets.DateOffset]

    def __post_init__(self):
        object.__setattr__(self, "type", pd.PeriodDtype(freq=self.freq))

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.PeriodDtype):
        """Convert a :class:`pandas.PeriodDtype` to
        a Pandera :class:`pandera.engines.pandas_engine.Period`."""
        return cls(freq=pd_dtype.freq)  # type: ignore


# ###############################################################################
# # misc
# ###############################################################################


@Engine.register_dtype(equivalents=[pd.SparseDtype])
@immutable(init=True)
class Sparse(DataType):
    """Representation of pandas :class:`pd.SparseDtype`."""

    type: pd.SparseDtype = dataclasses.field(default=None, init=False)  # type: ignore[assignment]  # noqa
    dtype: PandasDataType = np.float64
    fill_value: Any = np.nan

    def __post_init__(self):
        object.__setattr__(
            self,
            "type",
            pd.SparseDtype(dtype=self.dtype, fill_value=self.fill_value),
        )

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.SparseDtype):
        """Convert a :class:`pandas.SparseDtype` to
        a Pandera :class:`pandera.engines.pandas_engine.Sparse`."""
        return cls(  # type: ignore[call-arg]
            dtype=pd_dtype.subtype,  # type: ignore[attr-defined]
            fill_value=pd_dtype.fill_value,
        )


@Engine.register_dtype
@immutable(init=True)
class Interval(DataType):
    """Representation of pandas :class:`pd.IntervalDtype`."""

    type: pd.IntervalDtype = dataclasses.field(default=None, init=False)  # type: ignore[assignment]  # noqa
    subtype: Union[str, np.dtype]

    def __post_init__(self):
        object.__setattr__(
            self, "type", pd.IntervalDtype(subtype=self.subtype)
        )

    @classmethod
    def from_parametrized_dtype(cls, pd_dtype: pd.IntervalDtype):
        """Convert a :class:`pandas.IntervalDtype` to
        a Pandera :class:`pandera.engines.pandas_engine.Interval`."""
        return cls(subtype=pd_dtype.subtype)  # type: ignore


###############################################################################
# pydantic
###############################################################################


@Engine.register_dtype
@dtypes.immutable(init=True)
class PydanticModel(DataType):
    """A pydantic model datatype applying to rows in a dataframe."""

    type: Type[BaseModel] = dataclasses.field(default=None, init=False)  # type: ignore[assignment]
    auto_coerce = True

    # pylint:disable=super-init-not-called
    def __init__(self, model: Type[BaseModel]) -> None:
        object.__setattr__(self, "type", model)

    def coerce(self, data_container: PandasObject) -> PandasObject:
        """Coerce pandas dataframe with pydantic record model."""

        # pylint: disable=import-outside-toplevel
        from pandera.backends.pandas import error_formatters

        def _coerce_row(row):
            """
            Coerce each row using pydantic model, keeping track of failure
            cases.
            """
            try:
                # pylint: disable=no-member
                if PYDANTIC_V2:
                    row = self.type.model_validate(row).model_dump()
                else:
                    row = self.type.parse_obj(row).dict()
                row["failure_cases"] = np.nan
            except ValidationError as exc:
                row["failure_cases"] = {
                    k: row[k] for k in (x["loc"][0] for x in exc.errors())
                }

            return row

        records = data_container.to_dict(orient="records")  # type: ignore
        coerced_df = type(data_container).from_records(  # type: ignore
            [_coerce_row(row) for row in records]
        )

        # raise a ParserError with failure cases where each case is a
        # dictionary containing the failed elements in the pydantic record
        if coerced_df["failure_cases"].any():
            failure_cases = coerced_df["failure_cases"][
                coerced_df["failure_cases"].notna()
            ].astype(str)
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.type}",
                failure_cases=error_formatters.reshape_failure_cases(
                    failure_cases, ignore_na=False
                ),
            )
        return coerced_df.drop(columns="failure_cases")


###############################################################################
# Generic Python types
###############################################################################


@dtypes.immutable(init=True)
class PythonGenericType(DataType):
    """A datatype to support python generics."""

    type: Any = dataclasses.field(default=None, init=False)  # type: ignore
    generic_type: Any = dataclasses.field(default=None, init=False)
    special_type: Any = dataclasses.field(default=None, init=False)
    coercion_model: Type[BaseModel] = dataclasses.field(  # type: ignore
        default=None, init=False
    )
    _pandas_type = object

    def _check_type(self, element: Any) -> bool:
        # if the element is None or pd.NA, this function should return True:
        # the schema should only fail if nullable=False is specifed at the
        # schema/schema component level.
        if element is None or element is pd.NA:
            return True

        try:
            _type = getattr(self, "generic_type") or getattr(
                self, "special_type"
            )
            if (
                engine._is_typeddict(_type)
                and sys.version_info < (3, 12)
                and sys.version_info >= (3, 8)
            ):
                # replace the typing_extensions TypedDict with typing.TypedDict,
                # since pydantic needs typing_extensions.TypedDict but typeguard
                # can only type-check typing.TypedDict

                # pylint: disable=import-outside-toplevel,no-member
                from typing import TypedDict as _TypedDict

                _type = _TypedDict(_type.__name__, _type.__annotations__)  # type: ignore

            if TYPEGUARD_COLLECTION_STRATEGY_AVAILABLE:
                typeguard.check_type(element, _type, **type_types_kwargs)
            else:
                # typeguard <= 3 takes `argname` as the first positional argument
                typeguard.check_type("data_container", element, _type)

            return True
        except TYPEGUARD_ERROR:
            return False

    def _coerce_element(self, element: Any) -> Any:
        try:
            # pylint: disable=not-callable
            if PYDANTIC_V2:
                coerced_element = self.coercion_model(element).root
            else:
                coerced_element = self.coercion_model(
                    __root__=element
                ).__root__
        except ValidationError:
            coerced_element = pd.NA
        return coerced_element

    def check(
        self,
        pandera_dtype: dtypes.DataType,
        data_container: Optional[PandasObject] = None,
    ) -> Union[bool, Iterable[bool]]:
        """Check that data container has the expected type."""
        try:
            pandera_dtype = Engine.dtype(pandera_dtype)
        except TypeError:
            return False

        # the underlying pandas dtype must be an object
        if pandera_dtype != Engine.dtype(self._pandas_type):
            return False

        if data_container is None:
            return True
        elif self.generic_type is None and self.special_type is None:
            return data_container.map(type) == self.type  # type: ignore[operator]
        else:
            return data_container.map(self._check_type)  # type: ignore[operator]

    def coerce(self, data_container: PandasObject) -> PandasObject:
        """Coerce data container to the specified data type."""
        # pylint: disable=import-outside-toplevel
        from pandera.backends.pandas import error_formatters

        orig_isna = data_container.isna()
        coerced_data = data_container.map(self._coerce_element)  # type: ignore[operator]
        failed_selector = coerced_data.isna() & ~orig_isna
        failure_cases = coerced_data[failed_selector]

        if len(failure_cases) > 0:
            raise errors.ParserError(
                f"Could not coerce {type(data_container)} data_container "
                f"into type {self.generic_type or self.type}",
                failure_cases=error_formatters.reshape_failure_cases(
                    failure_cases, ignore_na=False
                ),
            )

        return coerced_data

    def __str__(self) -> str:
        return str(self.generic_type or self.type)


def _create_coercion_model(generic_type: Any):
    if PYDANTIC_V2:
        return create_model(
            "coercion_model",
            __base__=RootModel,
            root=(generic_type, ...),
        )
    return create_model("coercion_model", __root__=(generic_type, ...))


@Engine.register_dtype(equivalents=[dict, "dict"])
@dtypes.immutable(init=True)
class PythonDict(PythonGenericType):
    """A datatype to support python generics."""

    type: Type[dict] = dict

    def __init__(  # pylint:disable=super-init-not-called
        self, generic_type: Optional[Type] = None
    ) -> None:
        if generic_type is not None:
            object.__setattr__(self, "generic_type", generic_type)

            # Use pydantic to coerce data
            object.__setattr__(
                self,
                "coercion_model",
                _create_coercion_model(generic_type),
            )


@Engine.register_dtype(equivalents=[list, "list"])
@dtypes.immutable(init=True)
class PythonList(PythonGenericType):
    """A datatype to support python generics."""

    type: Type[list] = list

    def __init__(  # pylint:disable=super-init-not-called
        self, generic_type: Optional[Type] = None
    ) -> None:
        if generic_type is not None:
            object.__setattr__(self, "generic_type", generic_type)

            # Use pydantic to coerce data
            object.__setattr__(
                self,
                "coercion_model",
                _create_coercion_model(generic_type),
            )


@Engine.register_dtype(equivalents=[tuple, "tuple"])
@dtypes.immutable(init=True)
class PythonTuple(PythonGenericType):
    """A datatype to support python generics."""

    type: Type[list] = list

    def __init__(  # pylint:disable=super-init-not-called
        self, generic_type: Optional[Type] = None
    ) -> None:
        if generic_type is not None:
            object.__setattr__(self, "generic_type", generic_type)

            # Use pydantic to coerce data
            object.__setattr__(
                self,
                "coercion_model",
                _create_coercion_model(generic_type),
            )


@Engine.register_dtype(equivalents=[TypedDict, "TypedDict"])
@dtypes.immutable(init=True)
class PythonTypedDict(PythonGenericType):
    """A datatype to support python generics."""

    type = TypedDict  # type: ignore[assignment]

    def __init__(  # pylint:disable=super-init-not-called
        self,
        special_type: Optional[Type] = None,
    ) -> None:
        if special_type is not None:
            object.__setattr__(self, "special_type", special_type)

            # Use pydantic to coerce data
            object.__setattr__(
                self,
                "coercion_model",
                _create_coercion_model(self.special_type or self.type),  # type: ignore
            )

    def __str__(self) -> str:
        return str(TypedDict.__name__)  # type: ignore[attr-defined]


@Engine.register_dtype(equivalents=[NamedTuple, "NamedTuple"])
@dtypes.immutable(init=True)
class PythonNamedTuple(PythonGenericType):
    """A datatype to support python generics."""

    type = NamedTuple

    def __init__(  # pylint:disable=super-init-not-called
        self,
        special_type: Optional[Type] = None,
    ) -> None:
        if special_type is not None:
            object.__setattr__(self, "special_type", special_type)

            # Use pydantic to coerce data
            object.__setattr__(
                self,
                "coercion_model",
                _create_coercion_model(self.special_type or self.type),  # type: ignore
            )

    def __str__(self) -> str:
        return str(NamedTuple.__name__)


###############################################################################
# pyarrow types
###############################################################################

if PYARROW_INSTALLED and PANDAS_2_0_0_PLUS:

    class ArrowDataType(DataType):
        """Base `DataType` for boxing Pandas Arrow data types."""

        def coerce_value(self, value: Any) -> Any:
            """Coerce a value to a particular type."""
            return pyarrow.scalar(
                value,
                type=(
                    self.type.pyarrow_dtype  # pylint: disable=E1101
                    if self.type
                    else None
                ),
            )

    @Engine.register_dtype(
        equivalents=[
            "bool[pyarrow]",
            pyarrow.bool_,
            pd.ArrowDtype(pyarrow.bool_()),
        ]
    )
    @immutable
    class ArrowBool(ArrowDataType, BOOL):
        """Semantic representation of a :class:`pyarrow.bool_`."""

        type = pd.ArrowDtype(pyarrow.bool_())

    @Engine.register_dtype(
        equivalents=[
            "int64[pyarrow]",
            pyarrow.int64,
            pd.ArrowDtype(pyarrow.int64()),
        ]
    )
    @immutable
    class ArrowInt64(ArrowDataType, dtypes.Int):
        """Semantic representation of a :class:`pyarrow.int64`."""

        type = pd.ArrowDtype(pyarrow.int64())
        bit_width: int = 64

    @Engine.register_dtype(
        equivalents=[
            "int32[pyarrow]",
            pyarrow.int32,
            pd.ArrowDtype(pyarrow.int32()),
        ]
    )
    @immutable
    class ArrowInt32(ArrowInt64):
        """Semantic representation of a :class:`pyarrow.int32`."""

        type = pd.ArrowDtype(pyarrow.int32())
        bit_width: int = 32

    @Engine.register_dtype(
        equivalents=[
            "int16[pyarrow]",
            pyarrow.int16,
            pd.ArrowDtype(pyarrow.int16()),
        ]
    )
    @immutable
    class ArrowInt16(ArrowInt32):
        """Semantic representation of a :class:`pyarrow.int16`."""

        type = pd.ArrowDtype(pyarrow.int16())
        bit_width: int = 16

    @Engine.register_dtype(
        equivalents=[
            "int8[pyarrow]",
            pyarrow.int8,
            pd.ArrowDtype(pyarrow.int8()),
        ]
    )
    @immutable
    class ArrowInt8(ArrowInt16):
        """Semantic representation of a :class:`pyarrow.int8`."""

        type = pd.ArrowDtype(pyarrow.int8())
        bit_width: int = 8

    @Engine.register_dtype(
        equivalents=[
            pyarrow.string,
            pyarrow.utf8,
            pd.ArrowDtype(pyarrow.string()),
            pd.ArrowDtype(pyarrow.utf8()),
        ]
    )
    @immutable
    class ArrowString(ArrowDataType, dtypes.String):
        """Semantic representation of a :class:`pyarrow.string`."""

        type = pd.ArrowDtype(pyarrow.string())

    @Engine.register_dtype(
        equivalents=[
            "uint64[pyarrow]",
            pyarrow.uint64,
            pd.ArrowDtype(pyarrow.uint64()),
        ]
    )
    @immutable
    class ArrowUInt64(ArrowDataType, dtypes.UInt):
        """Semantic representation of a :class:`pyarrow.uint64`."""

        type = pd.ArrowDtype(pyarrow.uint64())
        bit_width: int = 64

    @Engine.register_dtype(
        equivalents=[
            "uint32[pyarrow]",
            pyarrow.uint32,
            pd.ArrowDtype(pyarrow.uint32()),
        ]
    )
    @immutable
    class ArrowUInt32(ArrowUInt64):
        """Semantic representation of a :class:`pyarrow.uint32`."""

        type = pd.ArrowDtype(pyarrow.uint32())
        bit_width: int = 32

    @Engine.register_dtype(
        equivalents=[
            "uint16[pyarrow]",
            pyarrow.uint16,
            pd.ArrowDtype(pyarrow.uint16()),
        ]
    )
    @immutable
    class ArrowUInt16(ArrowUInt32):
        """Semantic representation of a :class:`pyarrow.uint16`."""

        type = pd.ArrowDtype(pyarrow.uint16())
        bit_width: int = 16

    @Engine.register_dtype(
        equivalents=[
            "uint8[pyarrow]",
            pyarrow.uint8,
            pd.ArrowDtype(pyarrow.uint8()),
        ]
    )
    @immutable
    class ArrowUInt8(ArrowUInt16):
        """Semantic representation of a :class:`pyarrow.uint8`."""

        type = pd.ArrowDtype(pyarrow.uint8())
        bit_width: int = 8

    @Engine.register_dtype(
        equivalents=[
            "double[pyarrow]",
            pyarrow.float64,
            pd.ArrowDtype(pyarrow.float64()),
        ]
    )
    @immutable
    class ArrowFloat64(ArrowDataType, dtypes.Float):
        """Semantic representation of a :class:`pyarrow.float64`."""

        type = pd.ArrowDtype(pyarrow.float64())
        bit_width: int = 64

    @Engine.register_dtype(
        equivalents=[
            "float[pyarrow]",
            pyarrow.float32,
            pd.ArrowDtype(pyarrow.float32()),
        ]
    )
    @immutable
    class ArrowFloat32(ArrowFloat64):
        """Semantic representation of a :class:`pyarrow.float32`."""

        type = pd.ArrowDtype(pyarrow.float32())
        bit_width: int = 32

    @Engine.register_dtype(
        equivalents=[
            "halffloat[pyarrow]",
            pyarrow.float16,
            pd.ArrowDtype(pyarrow.float16()),
        ]
    )
    @immutable
    class ArrowFloat16(ArrowFloat32):
        """Semantic representation of a :class:`pyarrow.float16`."""

        type = pd.ArrowDtype(pyarrow.float16())
        bit_width: int = 16

    @Engine.register_dtype(
        equivalents=[pyarrow.decimal128, pyarrow.Decimal128Type]
    )
    @immutable(init=True)
    class ArrowDecimal128(ArrowDataType, dtypes.Decimal):
        """Semantic representation of a :class:`pyarrow.decimal128`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        precision: int = 28
        scale: int = 0

        def __post_init__(self):
            type_ = pd.ArrowDtype(
                pyarrow.decimal128(self.precision, self.scale)
            )
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(
            cls,
            pyarrow_dtype: pyarrow.Decimal128Type,
        ):
            return cls(precision=pyarrow_dtype.precision, scale=pyarrow_dtype.scale)  # type: ignore

    @Engine.register_dtype(
        equivalents=[pyarrow.timestamp, pyarrow.TimestampType]
    )
    @immutable(init=True)
    class ArrowTimestamp(ArrowDataType, dtypes.Timestamp):
        """Semantic representation of a :class:`pyarrow.timestamp`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        unit: Optional[str] = "ns"
        tz: Optional[datetime.tzinfo] = None

        def __post_init__(self):
            type_ = pd.ArrowDtype(pyarrow.timestamp(self.unit, self.tz))
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.TimestampType):
            return cls(unit=pyarrow_dtype.unit, tz=pyarrow_dtype.tz)  # type: ignore

    @Engine.register_dtype(
        equivalents=[pyarrow.dictionary, pyarrow.DictionaryType]
    )
    @immutable(init=True)
    class ArrowDictionary(ArrowDataType):
        """Semantic representation of a :class:`pyarrow.dictionary`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        index_type: Optional[pyarrow.DataType] = pyarrow.int64()
        value_type: Optional[pyarrow.DataType] = pyarrow.int64()
        ordered: bool = False

        def __post_init__(self):
            type_ = pd.ArrowDtype(
                pyarrow.dictionary(
                    self.index_type,
                    self.value_type,
                    self.ordered,
                )
            )
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(
            cls, pyarrow_dtype: pyarrow.DictionaryType
        ):
            return cls(
                index_type=pyarrow_dtype.index_type,  # type: ignore
                value_type=pyarrow_dtype.value_type,  # type: ignore
                ordered=pyarrow_dtype.ordered,  # type: ignore
            )

    @Engine.register_dtype(
        equivalents=[
            pyarrow.list_,
            pyarrow.ListType,
            pyarrow.FixedSizeListType,
        ]
    )
    @immutable(init=True)
    class ArrowList(ArrowDataType):
        """Semantic representation of a :class:`pyarrow.list_`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        value_type: Optional[Union[pyarrow.DataType, pyarrow.Field]] = (
            pyarrow.string()
        )
        list_size: Optional[int] = -1

        def __post_init__(self):
            type_ = pd.ArrowDtype(
                pyarrow.list_(self.value_type, self.list_size)
            )
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(
            cls,
            pyarrow_dtype: Union[pyarrow.ListType, pyarrow.FixedSizeListType],
        ):
            try:
                _dtype = cls(
                    value_type=pyarrow_dtype.value_type,  # type: ignore
                    list_size=pyarrow_dtype.list_size,  # type: ignore
                )
            except AttributeError:
                _dtype = cls(value_type=pyarrow_dtype.value_type)  # type: ignore
            return _dtype

    @Engine.register_dtype(equivalents=[pyarrow.struct, pyarrow.StructType])
    @immutable(init=True)
    class ArrowStruct(ArrowDataType):
        """Semantic representation of a :class:`pyarrow.struct`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        fields: Optional[
            Union[
                Iterable[Union[pyarrow.Field, Tuple[str, pyarrow.DataType]]],
                Dict[str, pyarrow.DataType],
            ]
        ] = tuple()

        def __post_init__(self):
            type_ = pd.ArrowDtype(pyarrow.struct(self.fields))
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.StructType):
            return cls(
                fields=[pyarrow_dtype.field(i) for i in range(pyarrow_dtype.num_fields)]  # type: ignore
            )

    @Engine.register_dtype(
        equivalents=[
            "null[pyarrow]",
            pyarrow.null,
            pd.ArrowDtype(pyarrow.null()),
        ]
    )
    @immutable
    class ArrowNull(ArrowDataType):
        """Semantic representation of a :class:`pyarrow.null`."""

        type = pd.ArrowDtype(pyarrow.null())

    @Engine.register_dtype(
        equivalents=[
            "date32[day][pyarrow]",
            pyarrow.date32,
            pd.ArrowDtype(pyarrow.date32()),
        ]
    )
    @immutable
    class ArrowDate32(ArrowDataType, dtypes.Date):
        """Semantic representation of a :class:`pyarrow.date32`."""

        type = pd.ArrowDtype(pyarrow.date32())

    @Engine.register_dtype(
        equivalents=[
            "date64[ms][pyarrow]",
            pyarrow.date64,
            pd.ArrowDtype(pyarrow.date64()),
        ]
    )
    @immutable
    class ArrowDate64(ArrowDataType, dtypes.Date):
        """Semantic representation of a :class:`pyarrow.date64`."""

        type = pd.ArrowDtype(pyarrow.date64())

    @Engine.register_dtype(
        equivalents=[pyarrow.duration, pyarrow.DurationType]
    )
    @immutable(init=True)
    class ArrowDuration(ArrowDataType):
        """Semantic representation of a :class:`pyarrow.duration`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        unit: Optional[str] = "ns"

        def __post_init__(self):
            type_ = pd.ArrowDtype(pyarrow.duration(self.unit))
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.DurationType):
            return cls(unit=pyarrow_dtype.unit)  # type: ignore

    @Engine.register_dtype(equivalents=[pyarrow.time32, pyarrow.Time32Type])
    @immutable(init=True)
    class ArrowTime32(ArrowDataType):
        """Semantic representation of a :class:`pyarrow.time32`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        unit: Optional[str] = "ms"

        def __post_init__(self):
            type_ = pd.ArrowDtype(pyarrow.time32(self.unit))
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.Time32Type):
            return cls(unit=pyarrow_dtype.unit)  # type: ignore

        def coerce(self, data_container: PandasObject) -> PandasObject:
            if data_container.dtype == self.type:
                return data_container
            else:
                return data_container.astype(
                    pd.ArrowDtype(pyarrow.int32())
                ).astype(self.type)

    @Engine.register_dtype(equivalents=[pyarrow.time64, pyarrow.Time64Type])
    @immutable(init=True)
    class ArrowTime64(ArrowDataType):
        """Semantic representation of a :class:`pyarrow.time64`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        unit: Optional[str] = "ns"

        def __post_init__(self):
            type_ = pd.ArrowDtype(pyarrow.time64(self.unit))
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.Time64Type):
            return cls(unit=pyarrow_dtype.unit)  # type: ignore

        def coerce(self, data_container: PandasObject) -> PandasObject:
            if data_container.dtype == self.type:
                return data_container
            else:
                return data_container.astype(
                    pd.ArrowDtype(pyarrow.int64())
                ).astype(self.type)

    @Engine.register_dtype(equivalents=[pyarrow.map_, pyarrow.MapType])
    @immutable(init=True)
    class ArrowMap(ArrowDataType):
        """Semantic representation of a :class:`pyarrow.map_`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        key_type: Optional[pyarrow.DataType] = pyarrow.int64()
        item_type: Optional[pyarrow.DataType] = pyarrow.int64()
        keys_sorted: bool = False

        def __post_init__(self):
            type_ = pd.ArrowDtype(
                pyarrow.map_(
                    self.key_type,
                    self.item_type,
                    self.keys_sorted,
                )
            )
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(cls, pyarrow_dtype: pyarrow.MapType):
            return cls(
                key_type=pyarrow_dtype.key_type,  # type: ignore
                item_type=pyarrow_dtype.item_type,  # type: ignore
                keys_sorted=pyarrow_dtype.keys_sorted,  # type: ignore
            )

    @Engine.register_dtype(
        equivalents=[
            "binary[pyarrow]",
            pyarrow.binary,
            pyarrow.FixedSizeBinaryType,
            pd.ArrowDtype(pyarrow.binary()),
        ]
    )
    @immutable(init=True)
    class ArrowBinary(ArrowDataType, dtypes.Binary):
        """Semantic representation of a :class:`pyarrow.binary`."""

        type: Optional[pd.ArrowDtype] = dataclasses.field(
            default=None, init=False
        )
        length: Optional[int] = -1

        def __post_init__(self):
            type_ = pd.ArrowDtype(pyarrow.binary(self.length))
            object.__setattr__(self, "type", type_)

        @classmethod
        def from_parametrized_dtype(
            cls,
            pyarrow_dtype: Union[
                pyarrow.DataType, pyarrow.FixedSizeBinaryType
            ],
        ):
            try:
                _dtype = cls(length=pyarrow_dtype.byte_width)  # type: ignore
            except (ValueError, AttributeError):
                _dtype = cls()  # type: ignore
            return _dtype

    @Engine.register_dtype(
        equivalents=[
            "large_binary[pyarrow]",
            pyarrow.large_binary,
            pd.ArrowDtype(pyarrow.large_binary()),
        ]
    )
    @immutable
    class ArrowLargeBinary(ArrowDataType):
        """Semantic representation of a :class:`pyarrow.large_binary`."""

        type = pd.ArrowDtype(pyarrow.large_binary())

    @Engine.register_dtype(
        equivalents=[
            "large_string[pyarrow]",
            pyarrow.large_string,
            pyarrow.large_utf8,
            pd.ArrowDtype(pyarrow.large_string()),
            pd.ArrowDtype(pyarrow.large_utf8()),
        ]
    )
    @immutable
    class ArrowLargeString(ArrowDataType, dtypes.String):
        """Semantic representation of a :class:`pyarrow.large_string`."""

        type = pd.ArrowDtype(pyarrow.large_string())
