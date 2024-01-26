"""Base classes for model api."""

import os
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pandera.api.base.model_components import BaseFieldInfo
from pandera.api.base.model_config import BaseModelConfig
from pandera.api.checks import Check
from pandera.typing import AnnotationInfo

TBaseModel = TypeVar("TBaseModel", bound="BaseModel")


class MetaModel(type):
    """Add string representations, mainly for pydantic."""

    def __repr__(cls):
        return str(cls)

    def __str__(cls):
        return cls.__name__


class BaseModel(metaclass=MetaModel):
    """Base class for a Data Object Model."""

    Config: Type[BaseModelConfig] = BaseModelConfig
    __extras__: Optional[Dict[str, Any]] = None
    __schema__: Optional[Any] = None
    __config__: Optional[Type[BaseModelConfig]] = None

    #: Key according to `FieldInfo.name`
    __fields__: Mapping[str, Tuple[AnnotationInfo, BaseFieldInfo]] = {}
    __checks__: Dict[str, List[Check]] = {}
    __root_checks__: List[Check] = []

    # This is syntactic sugar that delegates to the validate method
    def __new__(cls, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __class_getitem__(
        cls: Type[TBaseModel],
        params: Union[Type[Any], Tuple[Type[Any], ...]],
    ) -> Type[TBaseModel]:
        """
        Parameterize the class's generic arguments with the specified types.

        This allows for using generic types in the Model class definition.
        """
        raise NotImplementedError

    @classmethod
    def to_schema(cls) -> Any:
        """Create a Schema object from this Model class."""
        raise NotImplementedError

    @classmethod
    def to_yaml(cls, stream: Optional[os.PathLike] = None):
        """Convert `Schema` to yaml using `io.to_yaml`."""
        raise NotImplementedError

    @classmethod
    def validate(
        cls: Type[TBaseModel],
        check_obj: Any,
        head: Optional[int] = None,
        tail: Optional[int] = None,
        sample: Optional[int] = None,
        random_state: Optional[int] = None,
        lazy: bool = False,
        inplace: bool = False,
    ) -> Any:
        """Validate a data object."""
        raise NotImplementedError

    @classmethod
    def strategy(cls: Type[TBaseModel], *, size: Optional[int] = None):
        """Create a data synthesis strategy."""
        raise NotImplementedError

    @classmethod
    def example(cls: Type[TBaseModel], *, size: Optional[int] = None) -> Any:
        """Generate an example of this data model specification."""
        raise NotImplementedError

    ####################################
    # Methods for pydantic integration #
    ####################################

    @classmethod
    def __get_validators__(cls):
        """Yield a pydantic-compatible validator."""
        raise NotImplementedError

    @classmethod
    def __modify_schema__(cls, field_schema):
        """Update pydantic field schema."""
        raise NotImplementedError
