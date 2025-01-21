"""Common typing functionality."""

# pylint:disable=abstract-method,too-many-ancestors,invalid-name

import copy
import inspect
from typing import (  # type: ignore[attr-defined]
    TYPE_CHECKING,
    Any,
    Generic,
    Optional,
    Type,
    TypeVar,
    Union,
    _GenericAlias,
)

import pandas as pd
import typing_inspect

from pandera import dtypes, errors
from pandera.engines import numpy_engine, pandas_engine

Bool = dtypes.Bool  #: ``"bool"`` numpy dtype
Date = dtypes.Date  #: ``datetime.date`` object dtype
DateTime = dtypes.DateTime  #: ``"datetime64[ns]"`` numpy dtype
Decimal = dtypes.Decimal  #: ``decimal.Decimal`` object dtype
Timedelta = dtypes.Timedelta  #: ``"timedelta64[ns]"`` numpy dtype
Category = dtypes.Category  #: pandas ``"categorical"`` datatype
Float = dtypes.Float  #: ``"float"`` numpy dtype
Float16 = dtypes.Float16  #: ``"float16"`` numpy dtype
Float32 = dtypes.Float32  #: ``"float32"`` numpy dtype
Float64 = dtypes.Float64  #: ``"float64"`` numpy dtype
Int = dtypes.Int  #: ``"int"`` numpy dtype
Int8 = dtypes.Int8  #: ``"int8"`` numpy dtype
Int16 = dtypes.Int16  #: ``"int16"`` numpy dtype
Int32 = dtypes.Int32  #: ``"int32"`` numpy dtype
Int64 = dtypes.Int64  #: ``"int64"`` numpy dtype
UInt8 = dtypes.UInt8  #: ``"uint8"`` numpy dtype
UInt16 = dtypes.UInt16  #: ``"uint16"`` numpy dtype
UInt32 = dtypes.UInt32  #: ``"uint32"`` numpy dtype
UInt64 = dtypes.UInt64  #: ``"uint64"`` numpy dtype
INT8 = pandas_engine.INT8  #: ``"Int8"`` pandas dtype:: pandas 0.24.0+
INT16 = pandas_engine.INT16  #: ``"Int16"`` pandas dtype: pandas 0.24.0+
INT32 = pandas_engine.INT32  #: ``"Int32"`` pandas dtype: pandas 0.24.0+
INT64 = pandas_engine.INT64  #: ``"Int64"`` pandas dtype: pandas 0.24.0+
UINT8 = pandas_engine.UINT8  #: ``"UInt8"`` pandas dtype:: pandas 0.24.0+
UINT16 = pandas_engine.UINT16  #: ``"UInt16"`` pandas dtype: pandas 0.24.0+
UINT32 = pandas_engine.UINT32  #: ``"UInt32"`` pandas dtype: pandas 0.24.0+
UINT64 = pandas_engine.UINT64  #: ``"UInt64"`` pandas dtype: pandas 0.24.0+
Object = numpy_engine.Object  #: ``"object"`` numpy dtype
String = dtypes.String  #: ``"str"`` numpy dtype
#: ``"string"`` pandas dtypes: pandas 1.0.0+. For <1.0.0, this enum will
#: fall back on the str-as-object-array representation.
STRING = pandas_engine.STRING  #: ``"str"`` numpy dtype
BOOL = pandas_engine.BOOL  #: ``"str"`` numpy dtype


GenericDtype = TypeVar(  # type: ignore
    "GenericDtype",
    bound=Union[
        bool,
        int,
        str,
        float,
        pd.core.dtypes.base.ExtensionDtype,
        Bool,
        Date,
        DateTime,
        Decimal,
        Timedelta,
        Category,
        Float,
        Float16,
        Float32,
        Float64,
        Int,
        Int8,
        Int16,
        Int32,
        Int64,
        UInt8,
        UInt16,
        UInt32,
        UInt64,
        INT8,
        INT16,
        INT32,
        INT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        Object,
        String,
        STRING,
    ],
)

DataFrameModel = TypeVar("DataFrameModel", bound="DataFrameModel")  # type: ignore


# pylint:disable=invalid-name
if TYPE_CHECKING:
    T = TypeVar("T")  # pragma: no cover
else:
    T = DataFrameModel


__orig_generic_alias_call = copy.copy(_GenericAlias.__call__)


def __patched_generic_alias_call(self, *args, **kwargs):
    """
    Patched implementation of _GenericAlias.__call__ so that validation errors
    can be raised when instantiating an instance of pandera DataFrame generics,
    e.g. DataFrame[A](data).
    """
    if DataFrameBase not in self.__origin__.__bases__:
        return __orig_generic_alias_call(self, *args, **kwargs)

    if not self._inst:
        raise TypeError(
            f"Type {self._name} cannot be instantiated; "
            f"use {self.__origin__.__name__}() instead"
        )
    result = self.__origin__(*args, **kwargs)
    try:
        result.__orig_class__ = self
    # Limit the patched behavior to subset of exception types
    except (
        TypeError,
        errors.SchemaError,
        errors.SchemaInitError,
        errors.SchemaDefinitionError,
    ):
        raise
    # In python 3.11.9, all exceptions when setting attributes when defining
    # _GenericAlias subclasses are caught and ignored.
    except Exception:  # pylint: disable=broad-except
        pass
    return result


_GenericAlias.__call__ = __patched_generic_alias_call


class DataFrameBase(Generic[T]):
    # pylint: disable=too-few-public-methods
    """
    Pandera Dataframe base class for validating dataframes on
    initialization.
    """

    default_dtype: Optional[Type] = None

    def __setattr__(self, name: str, value: Any) -> None:
        # pylint: disable=no-member
        object.__setattr__(self, name, value)
        if name == "__orig_class__":
            orig_class = getattr(self, "__orig_class__")
            class_args = getattr(orig_class, "__args__", None)
            if class_args is not None and any(
                x.__name__ == "DataFrameModel"
                for x in inspect.getmro(class_args[0])
            ):
                schema_model = value.__args__[0]
            else:
                raise TypeError("Could not find DataFrameModel in class args")

            # prevent the double validation problem by preventing checks for
            # dataframes with a defined pandera.schema
            pandera_accessor = getattr(self, "pandera", None)

            if (
                pandera_accessor is None
                or pandera_accessor.schema is None
                or pandera_accessor.schema != schema_model.to_schema()
            ):
                self.__dict__ = schema_model.validate(self).__dict__
                if pandera_accessor is None:
                    pandera_accessor = getattr(self, "pandera")
                pandera_accessor.add_schema(schema_model.to_schema())


# pylint:disable=too-few-public-methods
class SeriesBase(Generic[GenericDtype]):
    """Pandera Series base class to use for all pandas-like APIs."""

    default_dtype: Optional[Type] = None

    def __get__(
        self, instance: object, owner: Type
    ) -> str:  # pragma: no cover
        raise AttributeError("Series should resolve to Field-s")


# pylint:disable=too-few-public-methods
class IndexBase(Generic[GenericDtype]):
    """Representation of pandas.Index, only used for type annotation.

    *new in 0.5.0*
    """

    default_dtype: Optional[Type] = None

    def __get__(
        self, instance: object, owner: Type
    ) -> str:  # pragma: no cover
        raise AttributeError("Indexes should resolve to pa.Index-s")


class AnnotationInfo:  # pylint:disable=too-few-public-methods
    """Captures extra information about an annotation.

    Attributes:
        origin: The non-parameterized generic class.
        args: All generic types for accessing as an iterable.
        arg: The first generic type (DataFrameModel does not support more than
            1 argument).
        literal: Whether the annotation is a literal.
        optional: Whether the annotation is optional.
        raw_annotation: The raw annotation.
        metadata: Extra arguments passed to :data:`typing.Annotated`.
    """

    def __init__(self, raw_annotation: Type) -> None:
        self._parse_annotation(raw_annotation)

    @property
    def is_generic_df(self) -> bool:
        """True if the annotation is a DataFrameBase subclass."""
        try:
            if self.origin is None:
                return False
            return issubclass(self.origin, DataFrameBase)
        except TypeError:
            return False

    def _parse_annotation(self, raw_annotation: Type) -> None:
        """Parse key information from annotation.

        :param annotation: A subscripted type.
        :returns: Annotation
        """
        self.raw_annotation = raw_annotation
        self.origin = self.arg = None
        self.is_annotated_type = False

        self.optional = typing_inspect.is_optional_type(raw_annotation)
        if self.optional and typing_inspect.is_union_type(raw_annotation):
            # Annotated with Optional or Union[..., NoneType]
            # get_args -> (pandera.typing.Index[str], <class 'NoneType'>)
            raw_annotation = typing_inspect.get_args(raw_annotation)[0]
            self.raw_annotation = raw_annotation

        self.origin = typing_inspect.get_origin(raw_annotation)
        # Replace empty tuple returned from get_args by None
        args = typing_inspect.get_args(raw_annotation) or None
        self.args = args
        self.arg = args[0] if args else args

        metadata = getattr(raw_annotation, "__metadata__", None)
        if metadata:
            self.is_annotated_type = True
        elif metadata := getattr(self.arg, "__metadata__", None):
            self.arg = typing_inspect.get_args(self.arg)[0]

        self.metadata = metadata
        self.literal = typing_inspect.is_literal_type(self.arg)

        if self.literal:
            self.arg = typing_inspect.get_args(self.arg)[0]
        elif self.origin is None and self.metadata is None:
            if isinstance(raw_annotation, type) and issubclass(
                raw_annotation, SeriesBase
            ):
                # handle case where the provided annotation is just a pandera Series generic.
                self.arg = Any
            else:
                # otherwise assume that the annotation is the data type itself.
                self.arg = raw_annotation
        self.default_dtype = getattr(raw_annotation, "default_dtype", None)
