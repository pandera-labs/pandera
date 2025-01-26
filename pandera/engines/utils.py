"""Engine module utilities."""

from typing import Any, Union

import numpy as np
import pandas as pd
from packaging import version

from pandera.engines.type_aliases import PandasObject


def pandas_version():
    """Return the pandas version."""

    return version.parse(pd.__version__)


def numpy_pandas_coercible(series: pd.Series, type_: Any) -> pd.Series:
    """Checks whether a series is coercible with respect to a type.

    Bisects the series until all the failure cases are found.

    NOTE: this does not account for pyspark.pandas .astype behavior, which
    defaults to converting uncastable values to NA values.
    """
    # pylint: disable=import-outside-toplevel,cyclic-import
    from pandera.engines import pandas_engine

    data_type = pandas_engine.Engine.dtype(type_)

    def _coercible(x):
        try:
            data_type.coerce_value(x)
            return True
        except Exception:  # pylint:disable=broad-except
            return False

    return series.map(_coercible).astype(bool)


def numpy_pandas_coerce_failure_cases(
    data_container: Union[PandasObject, np.ndarray], type_: Any
) -> Union[PandasObject, None]:
    """
    Get the failure cases resulting from trying to coerce a pandas/numpy object
    into particular data type.
    """
    # pylint: disable=import-outside-toplevel,cyclic-import
    from pandera.api.checks import Check
    from pandera.api.pandas.types import is_field, is_index, is_table
    from pandera.backends.pandas import error_formatters
    from pandera.backends.pandas.checks import PandasCheckBackend
    from pandera.engines import pandas_engine

    data_type = pandas_engine.Engine.dtype(type_)

    # this is check backend initialized here to use the postprocessing step
    # for formatting the failure cases obtained by the coercible function.
    stub_backend = PandasCheckBackend(Check(lambda _: _, ignore_na=False))

    if isinstance(data_container, np.ndarray):
        if len(data_container.shape) == 1:
            data_container = pd.Series(data_container)
        elif len(data_container.shape) == 2:
            data_container = pd.DataFrame(data_container)
        else:
            raise ValueError(
                "only numpy arrays of 1 or 2 dimensions are supported"
            )

    if is_index(data_container):
        data_container = data_container.to_series()  # type: ignore[union-attr,operator]

    if is_table(data_container):
        check_output = data_container.apply(  # type: ignore[union-attr,call-overload]
            numpy_pandas_coercible,
            args=(data_type,),
        )
        failure_cases = stub_backend.postprocess(
            data_container,  # type: ignore[arg-type]
            check_output,
        ).failure_cases
    elif is_field(data_container):
        check_output = numpy_pandas_coercible(data_container, data_type)  # type: ignore[arg-type]  # noqa
        failure_cases = stub_backend.postprocess(
            data_container,  # type: ignore[arg-type]
            check_output,
        ).failure_cases
    else:
        raise TypeError(
            f"type of data_container {type(data_container)} not understood. "
            "Must be a pandas Series, Index, or DataFrame."
        )
    if failure_cases is None:
        return failure_cases

    failure_cases = error_formatters.reshape_failure_cases(
        failure_cases, ignore_na=False
    )
    if failure_cases.empty:
        return None
    return failure_cases
