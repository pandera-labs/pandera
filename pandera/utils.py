"""General utility functions"""

from typing import Any, Callable, TypeVar


F = TypeVar("F", bound=Callable)


def docstring_substitution(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper around pandas.util.Substitution."""

    def decorator(func: F) -> F:
        if args:
            _doc = func.__doc__ % tuple(args)  # type: ignore[operator]
        elif kwargs:
            _doc = func.__doc__ % kwargs  # type: ignore[operator]
        func.__doc__ = _doc
        return func

    return decorator
