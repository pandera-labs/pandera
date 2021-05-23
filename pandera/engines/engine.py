"""Data types engine interface."""
# https://github.com/PyCQA/pylint/issues/3268
# pylint:disable=no-value-for-parameter
import functools
import inspect
import warnings
from abc import ABCMeta
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Type,
    TypeVar,
    Union,
    get_type_hints,
)

import typing_inspect

from pandera.dtypes_ import DataType

_DataType = TypeVar("_DataType", bound=DataType)
_Engine = TypeVar("_Engine", bound="Engine")
_EngineType = Type[_Engine]

if TYPE_CHECKING:

    class Dispatch:
        """Only used for type annotation."""

        def __call__(self, data_type: Any, **kwds: Any) -> Any:
            pass

        @staticmethod
        def register(
            data_type: Any, func: Callable[[Any], DataType]
        ) -> Callable[[Any], DataType]:
            """Register a new implementation for the given cls."""


else:
    Dispatch = Callable[[Any], DataType]


@dataclass
class _DtypeRegistry:
    dispatch: Dispatch
    equivalents: Dict[Any, DataType]


class Engine(ABCMeta):
    """Base Engine metaclass.

    Keep a registry of concrete Engines.
    """

    _registry: Dict["Engine", _DtypeRegistry] = {}
    _base_pandera_dtypes: Type[DataType]

    def __new__(cls, name, bases, namespace, **kwargs):

        base_pandera_dtypes = kwargs.pop("base_pandera_dtypes")
        try:  # allow multiple base datatypes
            base_pandera_dtypes = tuple(base_pandera_dtypes)
        except TypeError:
            pass
        namespace["_base_pandera_dtypes"] = base_pandera_dtypes
        engine = super().__new__(cls, name, bases, namespace, **kwargs)

        @functools.singledispatch
        def dtype(data_type: Any) -> DataType:
            raise ValueError(f"Data type '{data_type}' not understood")

        cls._registry[engine] = _DtypeRegistry(dispatch=dtype, equivalents={})
        return engine

    def _check_source_dtype(cls, data_type: Any) -> None:
        if isinstance(data_type, cls._base_pandera_dtypes) or (
            inspect.isclass(data_type)
            and issubclass(data_type, cls._base_pandera_dtypes)
        ):
            raise ValueError(
                f"{cls._base_pandera_dtypes.__name__} subclasses cannot be registered"
                f" with {cls.__name__}."
            )

    def _register_from_parametrized_dtype(
        cls,
        pandera_dtype_cls: Type[DataType],
    ) -> None:
        method = pandera_dtype_cls.__dict__["from_parametrized_dtype"]
        if not isinstance(method, classmethod):
            raise ValueError(
                f"{pandera_dtype_cls.__name__}.from_parametrized_dtype "
                + "must be a classmethod."
            )
        func = method.__func__
        annotations = get_type_hints(func).values()
        dtype = next(iter(annotations))  # get 1st annotation
        # parse typing.Union
        dtypes = typing_inspect.get_args(dtype) or [dtype]

        def _method(*args, **kwargs):
            return func(pandera_dtype_cls, *args, **kwargs)

        for source_dtype in dtypes:
            cls._check_source_dtype(source_dtype)
            cls._registry[cls].dispatch.register(source_dtype, _method)

    def _register_equivalents(
        cls,
        pandera_dtype_cls: Type[DataType],
        *source_dtypes: Any,
    ) -> None:
        pandera_dtype = pandera_dtype_cls()  # type: ignore
        for source_dtype in source_dtypes:
            cls._check_source_dtype(source_dtype)
            cls._registry[cls].equivalents[source_dtype] = pandera_dtype

    def register_dtype(
        cls: _EngineType,
        pandera_dtype_cls: Type[DataType] = None,
        *,
        equivalents: List[Any] = None,
    ):
        """Register a Pandera :class:`DataType`.

        :param pandera_dtype: The DataType to register.
        :param equivalents: Equivalent scalar data type class or
            non-parametrized data type instance.

        .. note::
            The classmethod ``from_parametrized_dtype`` will also be registered.
        """

        def _wrapper(pandera_dtype: Union[DataType, Type[DataType]]):
            if not inspect.isclass(pandera_dtype):
                raise ValueError(
                    f"{cls.__name__}.register_dtype can only decorate a class, "
                    + f"got {pandera_dtype}"
                )

            if equivalents:
                cls._register_equivalents(pandera_dtype, *equivalents)

            if "from_parametrized_dtype" in pandera_dtype.__dict__:
                cls._register_from_parametrized_dtype(pandera_dtype)
            elif not equivalents:
                warnings.warn(
                    f"register_dtype({pandera_dtype}) on a class without a "
                    + "'from_parametrized_dtype' classmethod has no effect."
                )

            return pandera_dtype

        if pandera_dtype_cls:
            return _wrapper(pandera_dtype_cls)

        return _wrapper

    def dtype(cls: _EngineType, data_type: Any) -> _DataType:
        """Convert input into a Pandera :class:`DataType` object."""
        if isinstance(data_type, cls._base_pandera_dtypes):
            return data_type

        if inspect.isclass(data_type) and issubclass(
            data_type, cls._base_pandera_dtypes
        ):
            try:
                return data_type()
            except (TypeError, AttributeError) as err:
                raise TypeError(
                    f"DataType '{data_type.__name__}' cannot be instantiated: "
                    f"{err}\n "
                    + "Usage Tip: Use an instance or a string representation."
                ) from err

        registry = cls._registry[cls]

        equivalent_data_type = registry.equivalents.get(data_type)
        if equivalent_data_type is not None:
            return equivalent_data_type

        try:
            return registry.dispatch(data_type)
        except (KeyError, ValueError):
            raise TypeError(
                f"Data type '{data_type}' not understood by {cls.__name__}."
            ) from None
