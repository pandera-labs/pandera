"""Decorators for integrating pandera into existing data pipelines."""

import functools
import inspect
import sys
import types
import typing
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    NoReturn,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import pandas as pd
from pydantic import validate_arguments

from pandera import errors
from pandera.api.base.error_handler import ErrorHandler
from pandera.api.dataframe.components import ComponentSchema
from pandera.api.dataframe.container import DataFrameSchema
from pandera.api.dataframe.model import DataFrameModel
from pandera.inspection_utils import (
    is_classmethod_from_meta,
    is_decorated_classmethod,
)
from pandera.typing import AnnotationInfo
from pandera.validation_depth import validation_type

Schemas = Union[DataFrameSchema, ComponentSchema]
InputGetter = Union[str, int]
OutputGetter = Union[str, int, Callable]
F = TypeVar("F", bound=Callable)


def _unwrap_fn(fn: Callable) -> Callable:
    if hasattr(fn, "__wrapped__"):
        return _unwrap_fn(fn.__wrapped__)
    return fn


def _get_fn_argnames(fn: Callable) -> List[str]:
    """Get argument names of a function.

    :param fn: get argument names for this function.
    :returns: list of argument names to be matched with the positional
    args passed in the decorator.

    .. note::
       Excludes first positional "self" or "cls" arguments if needed:
       - exclude self:
           - if fn is a method (self being an implicit argument)
       - exclude cls:
           - if fn is a decorated classmethod in Python 3.9+
           - if fn is declared as a regular method on a metaclass

    For functions decorated with ``@classmethod``, cls is excluded only in Python 3.9+
    because that is when Python's handling of classmethods changed and wrapt mirrors it.
    See: https://github.com/GrahamDumpleton/wrapt/issues/182
    """
    if hasattr(fn, "__wrapped__"):
        return _get_fn_argnames(fn.__wrapped__)

    arg_spec_args = inspect.getfullargspec(fn).args

    first_arg_is_self = arg_spec_args[0] == "self"
    first_arg_is_cls = arg_spec_args[0] == "cls"
    is_py_newer_than_39 = sys.version_info[:2] >= (3, 9)
    # Exclusion criteria
    is_regular_method = inspect.ismethod(fn) and first_arg_is_self
    is_decorated_cls_method = (
        is_decorated_classmethod(fn) and is_py_newer_than_39
    )
    is_cls_method_from_meta_method = is_classmethod_from_meta(fn)
    if (
        first_arg_is_self
        or first_arg_is_cls
        or is_regular_method
        or is_decorated_cls_method
        or is_cls_method_from_meta_method
    ):
        # Don't include "self" / "cls" argument
        arg_spec_args = arg_spec_args[1:]
    return arg_spec_args


def _handle_schema_error(
    decorator_name,
    fn: Callable,
    schema: Union[DataFrameSchema, ComponentSchema],
    data_obj: Any,
    schema_error: errors.SchemaError,
) -> NoReturn:
    """Reraise schema validation error with decorator context.

    :param fn: check the DataFrame or Series input of this function.
    :param schema: dataframe/series schema object
    :param arg_df: dataframe/series we are validating.
    :param schema_error: original exception.
    :raises SchemaError: when ``DataFrame`` violates built-in or custom
        checks.
    """
    raise _parse_schema_error(
        decorator_name,
        fn,
        schema,
        data_obj,
        schema_error,
        schema_error.reason_code,
    ) from schema_error


def _parse_schema_error(
    decorator_name,
    fn: Callable,
    schema: Union[DataFrameSchema, ComponentSchema],
    data_obj: Any,
    schema_error: errors.SchemaError,
    reason_code: errors.SchemaErrorReason,
) -> NoReturn:
    """Parse schema validation error with decorator context.

    :param fn: check the DataFrame or Series input of this function.
    :param schema: dataframe/series schema object
    :param arg_df: dataframe/series we are validating.
    :param schema_error: original exception.
    :param reason_code: SchemaErrorReason associated with the error.
    :raises SchemaError: when ``DataFrame`` violates built-in or custom
        checks.
    """
    func_name = fn.__name__
    if isinstance(fn, types.MethodType):
        func_name = fn.__self__.__class__.__name__ + "." + func_name
    msg = f"error in {decorator_name} decorator of function '{func_name}': {schema_error}"
    return errors.SchemaError(  # type: ignore[misc]
        schema,
        data_obj,
        msg,
        failure_cases=schema_error.failure_cases,
        check=schema_error.check,
        check_index=schema_error.check_index,
        reason_code=reason_code,
    )


def check_input(
    schema: Schemas,
    obj_getter: Optional[InputGetter] = None,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable[[F], F]:
    # pylint: disable=duplicate-code
    """Validate function argument when function is called.

    This is a decorator function that validates the schema of a dataframe
    argument in a function.

    :param schema: dataframe/series schema object
    :param obj_getter:  (Default value = None) if int, obj_getter refers to the
        the index of the pandas dataframe/series to be validated in the args
        part of the function signature. If str, obj_getter refers to the
        argument name of the pandas dataframe/series in the function signature.
        This works even if the series/dataframe is passed in as a positional
        argument when the function is called. If None, assumes that the
        dataframe/series is the first argument of the decorated function
    :param head: validate the first n rows. Rows overlapping with `tail` or
        `sample` are de-duplicated.
    :param tail: validate the last n rows. Rows overlapping with `head` or
        `sample` are de-duplicated.
    :param sample: validate a random sample of n rows. Rows overlapping
        with `head` or `tail` are de-duplicated.
    :param random_state: random seed for the ``sample`` argument.
    :param lazy: if True, lazily evaluates dataframe against all validation
        checks and raises a ``SchemaErrors``. Otherwise, raise
        ``SchemaError`` as soon as one occurs.
    :param inplace: if True, applies coercion to the object of validation,
        otherwise creates a copy of the data.
    :returns: wrapped function

    :example:

    Check the input of a decorated function.

    >>> import pandas as pd
    >>> import pandera as pa
    >>>
    >>>
    >>> schema = pa.DataFrameSchema({"column": pa.Column(int)})
    >>>
    >>> @pa.check_input(schema)
    ... def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    ...     df["doubled_column"] = df["column"] * 2
    ...     return df
    >>>
    >>> df = pd.DataFrame({
    ...     "column": range(5),
    ... })
    >>>
    >>> transform_data(df)
       column  doubled_column
    0       0               0
    1       1               2
    2       2               4
    3       3               6
    4       4               8

    See :ref:`here<decorators>` for more usage details.

    """

    def decorator(wrapped):

        @functools.wraps(wrapped)
        def _wrapper(*args, **kwargs):
            """Check pandas DataFrame or Series before calling the function."""
            # pylint: disable=too-many-branches
            args = list(args)
            validate_args = (head, tail, sample, random_state, lazy, inplace)

            sig = inspect.signature(_unwrap_fn(wrapped))
            is_method = [*sig.parameters][0] in ("self", "cls")
            if is_method and len(args) == len(sig.parameters) - 1:
                pos_args = sig.bind_partial(None, *args).arguments
            else:
                pos_args = sig.bind_partial(*args).arguments

            if isinstance(obj_getter, int):
                try:
                    arg_idx = obj_getter + 1 if is_method else obj_getter
                    args[arg_idx] = schema.validate(args[arg_idx])
                except IndexError as exc:
                    raise IndexError(
                        f"error in check_input decorator of function '{wrapped.__name__}': the "
                        f"index '{obj_getter}' was supplied to the check but this "
                        f"function accepts '{len(_get_fn_argnames(wrapped))}' arguments, so the maximum "
                        f"index is 'max(0, len(_get_fn_argnames(fn)) - 1)'. The full error is: '{exc}'"
                    ) from exc
            elif isinstance(obj_getter, str):
                if obj_getter in kwargs:
                    kwargs[obj_getter] = schema.validate(
                        kwargs[obj_getter], *validate_args
                    )
                else:
                    arg_spec_args = _get_fn_argnames(wrapped)
                    pos_args[obj_getter] = schema.validate(
                        pos_args[obj_getter], *validate_args
                    )
                    args = list(pos_args.values())
            elif obj_getter is None:
                try:
                    _fn = _unwrap_fn(wrapped)
                    obj_arg_name, *_ = _get_fn_argnames(wrapped)
                    arg_spec_args = inspect.getfullargspec(_fn).args

                    arg_idx = arg_spec_args.index(obj_arg_name)

                    if obj_arg_name in kwargs:
                        obj = kwargs[obj_arg_name]
                        kwargs[obj_arg_name] = schema.validate(
                            obj, *validate_args
                        )
                    elif obj_arg_name in pos_args:
                        obj = args[arg_idx]
                        args[arg_idx] = schema.validate(obj, *validate_args)
                    else:
                        raise ValueError(
                            f"argument {obj_arg_name} not found in args or kwargs"
                        )
                except errors.SchemaError as e:
                    _handle_schema_error(
                        "check_input", wrapped, schema, obj, e
                    )
            else:
                raise TypeError(
                    f"obj_getter is unrecognized type: {type(obj_getter)}"
                )
            return wrapped(*args, **kwargs)

        return _wrapper

    return decorator


def check_output(
    schema: Schemas,
    obj_getter: Optional[OutputGetter] = None,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable[[F], F]:
    # pylint: disable=duplicate-code
    """Validate function output.

    Similar to input validator, but validates the output of the decorated
    function.

    :param schema: dataframe/series schema object
    :param obj_getter:  (Default value = None) if int, assumes that the output
        of the decorated function is a list-like object, where obj_getter is
        the index of the pandas data dataframe/series to be validated. If str,
        expects that the output is a dict-like object, and obj_getter is the
        key pointing to the dataframe/series to be validated. If a callable is
        supplied, it expects the output of decorated function and should return
        the dataframe/series to be validated.
    :param head: validate the first n rows. Rows overlapping with `tail` or
        `sample` are de-duplicated.
    :param tail: validate the last n rows. Rows overlapping with `head` or
        `sample` are de-duplicated.
    :param sample: validate a random sample of n rows. Rows overlapping
        with `head` or `tail` are de-duplicated.
    :param random_state: random seed for the ``sample`` argument.
    :param lazy: if True, lazily evaluates dataframe against all validation
        checks and raises a ``SchemaErrors``. Otherwise, raise
        ``SchemaError`` as soon as one occurs.
    :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
    :returns: wrapped function

    :example:

    Check the output a decorated function.

    >>> import pandas as pd
    >>> import pandera as pa
    >>>
    >>>
    >>> schema = pa.DataFrameSchema(
    ...     columns={"doubled_column": pa.Column(int)},
    ...     checks=pa.Check(
    ...         lambda df: df["doubled_column"] == df["column"] * 2
    ...     )
    ... )
    >>>
    >>> @pa.check_output(schema)
    ... def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    ...     df["doubled_column"] = df["column"] * 2
    ...     return df
    >>>
    >>> df = pd.DataFrame({"column": range(5)})
    >>>
    >>> transform_data(df)
       column  doubled_column
    0       0               0
    1       1               2
    2       2               4
    3       3               6
    4       4               8

    See :ref:`here<decorators>` for more usage details.
    """

    # make sure that callable obj_getter doesn't work when the schema has
    # any component that requires coercion, since there's no way to re-assign
    # the output to the coerced data.
    # pylint: disable=too-many-boolean-expressions
    if callable(obj_getter) and (
        schema.coerce
        or (schema.index is not None and schema.index.coerce)  # type: ignore[union-attr]
        or (
            isinstance(schema, DataFrameSchema)
            and any(col.coerce for col in schema.columns.values())
        )
    ):
        raise ValueError(
            "Cannot use callable obj_getter when the schema uses coercion."
        )

    def validate(out: Any, fn: Callable) -> None:
        def _try_validate(obj: Any):
            try:
                return schema.validate(
                    obj, head, tail, sample, random_state, lazy, inplace
                )
            except errors.SchemaError as e:
                _handle_schema_error("check_output", fn, schema, obj, e)

        if obj_getter is None:
            return _try_validate(out)
        elif isinstance(obj_getter, (int, str)):
            obj = out[obj_getter]
            validated = _try_validate(obj)
            if isinstance(out, tuple):
                out = list(out)
                out[obj_getter] = validated
                out = tuple(out)
            else:
                out[obj_getter] = validated
            return out
        elif callable(obj_getter):
            obj = obj_getter(out)
            _try_validate(obj)
            return out

        raise TypeError(f"obj_getter is unrecognized type: {type(obj_getter)}")

    def decorator(wrapped):

        @functools.wraps(wrapped)
        def _wrapper(*args, **kwargs):
            """Check pandas DataFrame or Series before calling the function."""
            args = list(args)
            _fn = _unwrap_fn(wrapped)

            if inspect.iscoroutinefunction(_fn):

                async def aio_wrapper():
                    res = await wrapped(*args, **kwargs)
                    validate(res, wrapped)
                    return res

                return aio_wrapper()
            else:
                out = wrapped(*args, **kwargs)
                return validate(out, wrapped)

        return _wrapper

    return decorator


def check_io(
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
    out: Union[
        Schemas,
        Tuple[OutputGetter, Schemas],
        List[Tuple[OutputGetter, Schemas]],
        None,
    ] = None,
    **inputs: Schemas,
) -> Callable[[F], F]:
    """Check schema for multiple inputs and outputs.

    See :ref:`here<decorators>` for more usage details.

    :param wrapped: the function to decorate.
    :param head: validate the first n rows. Rows overlapping with `tail` or
        `sample` are de-duplicated.
    :param tail: validate the last n rows. Rows overlapping with `head` or
        `sample` are de-duplicated.
    :param sample: validate a random sample of n rows. Rows overlapping
        with `head` or `tail` are de-duplicated.
    :param random_state: random seed for the ``sample`` argument.
    :param lazy: if True, lazily evaluates dataframe against all validation
        checks and raises a ``SchemaErrors``. Otherwise, raise
        ``SchemaError`` as soon as one occurs.
    :param inplace: if True, applies coercion to the object of validation,
        otherwise creates a copy of the data.
    :param out: this should be a schema object if the function outputs a single
        dataframe/series. It can be a two-tuple, where the first element is
        a string, integer, or callable that fetches the pandas data structure
        in the output, and the second element is the schema to validate
        against. For multiple outputs, specify a list of two-tuples following
        the above structure.
    :param inputs: kwargs keys should be the argument name in the decorated
        function and values should be the schema used to validate the pandas
        data structure referenced by the argument name.
    :returns: wrapped function
    """

    check_args = (head, tail, sample, random_state, lazy, inplace)

    def decorator(wrapped):

        @functools.wraps(wrapped)
        def _wrapper(*args, **kwargs):
            """Check pandas DataFrame or Series before calling the function.

            :param fn: check the DataFrame or Series output of this function
            :param instance: the object to which the wrapped function was bound
                when it was called. Only applies to methods.
            :param args: the list of positional arguments supplied when the
                decorated function was called.
            :param kwargs: the dictionary of keyword arguments supplied when the
                decorated function was called.
            """
            out_schemas = out
            if isinstance(out, list):
                out_schemas = out
            elif isinstance(out, (DataFrameSchema, ComponentSchema)):
                out_schemas = [(None, out)]  # type: ignore
            elif isinstance(out, tuple):
                out_schemas = [out]
            elif out is None:
                out_schemas = []
            else:
                raise TypeError(
                    f"type of out argument not recognized: {type(out)}"
                )

            wrapped_fn = wrapped
            for input_getter, input_schema in inputs.items():
                # pylint: disable=no-value-for-parameter
                wrapped_fn = check_input(
                    input_schema, input_getter, *check_args  # type: ignore
                )(wrapped_fn)

            # pylint: disable=no-value-for-parameter
            for out_getter, out_schema in out_schemas:  # type: ignore
                wrapped_fn = check_output(out_schema, out_getter, *check_args)(
                    wrapped_fn
                )

            return wrapped_fn(*args, **kwargs)

        return _wrapper

    return decorator


@overload
def check_types(
    wrapped: F,
    *,
    with_pydantic: bool = False,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> F: ...  # pragma: no cover


@overload
def check_types(
    wrapped: None = None,
    *,
    with_pydantic: bool = False,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable[[F], F]: ...  # pragma: no cover


def check_types(
    wrapped=None,
    *,
    with_pydantic: bool = False,
    head: Optional[int] = None,
    tail: Optional[int] = None,
    sample: Optional[int] = None,
    random_state: Optional[int] = None,
    lazy: bool = False,
    inplace: bool = False,
) -> Callable:
    # pylint: disable=too-many-statements
    """Validate function inputs and output based on type annotations.

    See the :ref:`User Guide <dataframe-models>` for more.

    :param wrapped: the function to decorate.
    :param with_pydantic: use ``pydantic.validate_arguments`` to validate
        inputs. This function is still needed to validate function outputs.
    :param head: validate the first n rows. Rows overlapping with `tail` or
        `sample` are de-duplicated.
    :param tail: validate the last n rows. Rows overlapping with `head` or
        `sample` are de-duplicated.
    :param sample: validate a random sample of n rows. Rows overlapping
        with `head` or `tail` are de-duplicated.
    :param random_state: random seed for the ``sample`` argument.
    :param lazy: if True, lazily evaluates dataframe against all validation
        checks and raises a ``SchemaErrors``. Otherwise, raise
        ``SchemaError`` as soon as one occurs.
    :param inplace: if True, applies coercion to the object of validation,
            otherwise creates a copy of the data.
    """
    # pylint: disable=too-many-locals
    if wrapped is None:
        return functools.partial(
            check_types,
            with_pydantic=with_pydantic,
            head=head,
            tail=tail,
            sample=sample,
            random_state=random_state,
            lazy=lazy,
            inplace=inplace,
        )

    # Front-load annotation parsing
    annotated_schema_models: Dict[
        str,
        Iterable[
            Tuple[Union[DataFrameModel, None], Union[AnnotationInfo, None]]
        ],
    ] = {}
    for arg_name_, annotation in typing.get_type_hints(wrapped).items():
        annotation_info = AnnotationInfo(annotation)
        if not annotation_info.is_generic_df:
            # pylint: disable=comparison-with-callable
            if annotation_info.origin == Union:
                annotation_model_pairs = []
                for annot in annotation_info.args:  # type: ignore[union-attr]
                    sub_annotation_info = AnnotationInfo(annot)
                    if not sub_annotation_info.is_generic_df:
                        continue

                    schema_model = cast(
                        DataFrameModel, sub_annotation_info.arg
                    )
                    annotation_model_pairs.append(
                        (schema_model, sub_annotation_info)
                    )
            else:
                continue
        else:
            schema_model = cast(DataFrameModel, annotation_info.arg)
            annotation_model_pairs = [(schema_model, annotation_info)]

        annotated_schema_models[arg_name_] = annotation_model_pairs

    def _check_arg(arg_name: str, arg_value: Any) -> Any:
        """
        Validate function's argument if annoted with a schema, else
        pass-through.
        """
        annotation_model_pairs = annotated_schema_models.get(
            arg_name, [(None, None)]
        )

        if not annotation_model_pairs:
            return arg_value

        error_handler = ErrorHandler(lazy=True)
        for schema_model, annotation_info in annotation_model_pairs:
            if schema_model is None:
                return arg_value

            if (
                annotation_info
                and not (annotation_info.optional and arg_value is None)
                # the pandera.schema attribute should only be available when
                # schema.validate has been called in the DF. There's probably
                # a better way of doing this
            ):
                config = schema_model.__config__
                data_container_type = annotation_info.origin
                schema = schema_model.to_schema()

                if data_container_type and config and config.from_format:
                    arg_value = data_container_type.from_format(
                        arg_value, config
                    )

                # Don't do checks if value is still a built-in type
                if isinstance(
                    arg_value, (int, str, bool, float, dict, list, tuple, set)
                ):
                    return arg_value

                if (
                    not hasattr(arg_value, "pandera")
                    or arg_value.pandera.schema is None
                    # don't re-validate a dataframe that contains the same
                    # exact schema
                    or arg_value.pandera.schema != schema
                ):
                    try:
                        arg_value = schema.validate(
                            arg_value,
                            head,
                            tail,
                            sample,
                            random_state,
                            lazy,
                            inplace,
                        )
                    except errors.SchemaError as e:
                        error_handler.collect_error(
                            validation_type(
                                errors.SchemaErrorReason.INVALID_TYPE
                            ),
                            errors.SchemaErrorReason.INVALID_TYPE,
                            _parse_schema_error(
                                "check_types",
                                wrapped,
                                schema,
                                arg_value,
                                e,
                                errors.SchemaErrorReason.INVALID_TYPE,
                            ),
                        )
                        continue  # pylint: disable=unreachable

                if data_container_type and config and config.to_format:
                    arg_value = data_container_type.to_format(
                        arg_value, config
                    )

                return arg_value

        if error_handler.schema_errors:
            if len(error_handler.schema_errors) == 1:
                raise error_handler.schema_errors[0]

            raise errors.SchemaErrors(
                schema=schema,
                schema_errors=(
                    error_handler.schema_errors
                    if isinstance(arg_value, pd.DataFrame)
                    else error_handler.collect_errors  # type: ignore
                ),
                data=arg_value,
            )

    sig = inspect.signature(wrapped)

    def validate_args(
        named_arguments: Dict[str, Any], arguments: Tuple[Any, ...]
    ) -> List[Any]:
        """
        Validates schemas of both explicit and *args-like function arguments.

        :param named_arguments: Bundled function arguments. Organized as key-value pairs of the
            argument name and value. *args-like arguments are bundled into a single tuple.
            Example: OrderedDict({'arg1': 1, 'arg2': 2, 'star_args': (3, 4, 5)})
        :param arguments: Unpacked function arguments, as written in the function call.
            Example: (1, 2, 3, 4, 5)
        :return: List of validated function arguments.
        """

        # Check for an '*args'-like argument
        if len(arguments) > len(named_arguments):
            (
                star_args_name,
                star_args_values,
            ) = named_arguments.popitem()  # *args is the last item

            star_args_tuple = (
                _check_arg(star_args_name, arg_value)
                for arg_value in star_args_values
            )

            explicit_args_tuple = (
                _check_arg(arg_name, arg_value)
                for arg_name, arg_value in named_arguments.items()
            )

            return list((*explicit_args_tuple, *star_args_tuple))

        else:
            return list(
                _check_arg(arg_name, arg_value)
                for arg_name, arg_value in named_arguments.items()
            )

    def validate_kwargs(
        named_kwargs: Dict[str, Any], kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates schemas of both explicit and **kwargs-like function arguments.

        :param named_kwargs: Bundled function keyword arguments. Organized as key-value pairs of
            the keyword argument name and value. **kwargs-like arguments are bundled into a single
            dictionary.
            Example: OrderedDict({'kwarg1': 1, 'kwarg2': 2, 'star_kwargs': {'kwarg3': 3, 'kwarg4': 4}})
        :param kwargs: Unpacked function keyword arguments, as written in the function call.
            Example: {'kwarg1': 1, 'kwarg2': 2, 'kwarg3': 3, 'kwarg4': 4}
        :return: list of validated function keyword arguments.
        """

        # Check for an '**kwargs'-like argument
        if kwargs.keys() != named_kwargs.keys():
            (
                star_kwargs_name,
                star_kwargs_dict,
            ) = named_kwargs.popitem()  # **kwargs is the last item

            explicit_kwargs_dict = {
                arg_name: _check_arg(arg_name, arg_value)
                for arg_name, arg_value in named_kwargs.items()
            }

            star_kwargs_dict = {
                arg_name: _check_arg(star_kwargs_name, arg_value)
                for arg_name, arg_value in star_kwargs_dict.items()
            }

            return {**explicit_kwargs_dict, **star_kwargs_dict}

        else:
            return {
                arg_name: _check_arg(arg_name, arg_value)
                for arg_name, arg_value in named_kwargs.items()
            }

    def validate_inputs(
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Tuple[List[Any], Dict[str, Any]]:
        validated_pos = validate_args(sig.bind_partial(*args).arguments, args)
        validated_kwd = validate_kwargs(
            sig.bind_partial(**kwargs).arguments, kwargs
        )
        return validated_pos, validated_kwd

    if inspect.iscoroutinefunction(_unwrap_fn(wrapped)):

        @functools.wraps(wrapped)
        async def _wrapper(*args, **kwargs):
            if with_pydantic:
                out = await validate_arguments(wrapped)(*args, **kwargs)
            else:
                validated_pos, validated_kwd = validate_inputs(args, kwargs)
                out = await wrapped(*validated_pos, **validated_kwd)
            return _check_arg("return", out)

    else:

        @functools.wraps(wrapped)
        def _wrapper(*args, **kwargs):
            if with_pydantic:
                out = validate_arguments(wrapped)(*args, **kwargs)
            else:
                validated_pos, validated_kwd = validate_inputs(args, kwargs)
                out = wrapped(*validated_pos, **validated_kwd)
            return _check_arg("return", out)

    return _wrapper
