"""General utility functions"""

from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable)


def docstring_substitution(*args: Any, **kwargs: Any) -> Callable[[F], F]:
    """Typed wrapper around pandas.util.Substitution."""

    def decorator(func: F) -> F:
        # handle case when pandera is run in optimized mode:
        # https://docs.python.org/3/using/cmdline.html#cmdoption-OO
        if func.__doc__ is None:
            return func

        if args:
            _doc = func.__doc__ % tuple(args)  # type: ignore[operator]
        elif kwargs:
            _doc = func.__doc__ % kwargs  # type: ignore[operator]
        func.__doc__ = _doc  # pylint:disable=possibly-used-before-assignment
        return func

    return decorator


def is_regex(name: str):
    """
    Checks whether a string is a regex pattern, as defined as starting with
    '^' and ending with '$'.
    """
    return name.startswith("^") and name.endswith("$")
