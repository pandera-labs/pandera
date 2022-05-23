"""Tests logical dtypes."""

from decimal import Decimal
from types import ModuleType
from typing import Any, Generator, Iterable, List, Optional, cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import pandera as pa
from pandera.engines import pandas_engine
from pandera.errors import ParserError


@pytest.fixture(scope="module")
def datacontainer_lib(request) -> Generator[ModuleType, None, None]:
    """Yield the data container library to test against."""
    local_path = str(request.fspath)
    if "modin" in local_path:
        import modin.pandas as mpd  # pylint: disable=import-outside-toplevel
        import ray  # pylint: disable=import-outside-toplevel

        ray.init()
        yield mpd
        ray.shutdown()

    elif "pyspark" in local_path:
        import pyspark.pandas as ps  # pylint: disable=import-outside-toplevel

        yield ps

    elif "dask" in local_path:
        import dask.dataframe as dd  # pylint: disable=import-outside-toplevel

        yield dd

    elif "core" in local_path or "pandas" in local_path:
        yield pd

    else:
        raise NotImplementedError(f"Not supported test package {local_path}")


@pytest.mark.parametrize(
    "data, expected_datatype, expected_results",
    [
        (
            [
                Decimal("1"),
                Decimal("1.2"),
                Decimal(".3"),
                Decimal("12.3"),
                "foo.bar",
                None,
                pd.NA,
                np.nan,
            ],
            pandas_engine.Decimal(2, 1),
            [True, True, True, False, False, True, True, True],
        ),
    ],
)
def test_logical_datatype_check(
    datacontainer_lib: ModuleType,  # pylint: disable=redefined-outer-name
    data,
    expected_datatype: pandas_engine.DataType,
    expected_results: List[bool],
):
    """Test decimal coerce."""
    data = datacontainer_lib.Series(data, dtype="object")  # type:ignore
    actual_datatype = pandas_engine.Engine.dtype(data.dtype)

    actual_results = expected_datatype.check(actual_datatype, data)
    assert list(expected_results) == list(cast(Iterable, actual_results))


@pytest.mark.parametrize(
    "data, expected_datatype, failure_cases",
    [
        (
            [Decimal("1.2"), Decimal("12.3")],
            pandas_engine.Decimal(2, 1),
            [Decimal("12.3")],
        ),
        (
            [Decimal("1.2"), None, pd.NA, np.nan],
            pandas_engine.Decimal(19, 5),
            [],
        ),
    ],
)
def test_logical_datatype_coerce(
    datacontainer_lib: ModuleType,  # pylint: disable=redefined-outer-name
    data,
    expected_datatype: pandas_engine.DataType,
    failure_cases: List[bool],
):
    """Test decimal coerce."""
    data = datacontainer_lib.Series(data)  # type:ignore
    failure_cases = pd.Series(failure_cases)

    if failure_cases.any():
        with pytest.raises(ParserError) as exc:
            expected_datatype.try_coerce(data)

        actual_failure_cases = pd.Series(
            exc.value.failure_cases["failure_case"].to_numpy()
        )
        assert_series_equal(
            failure_cases, actual_failure_cases, check_names=False
        )

        schema = pa.SeriesSchema(expected_datatype)
        try:
            schema.validate(data, lazy=True)
        except pa.errors.SchemaErrors as err:
            err_failure_cases = pd.Series(
                err.failure_cases["failure_case"].to_numpy()
            )
            assert_series_equal(
                failure_cases, err_failure_cases, check_names=False
            )

    else:
        coerced_data = expected_datatype.coerce(data)
        expected_datatype.check(
            pandas_engine.Engine.dtype(coerced_data.dtype), coerced_data
        )


@pytest.mark.parametrize(
    "data, datatype, expected_value",
    [
        (Decimal("1.2"), pandas_engine.Decimal(2, 1), Decimal("1.2")),
        ("1.2", pandas_engine.Decimal(2, 1), Decimal("1.2")),
        (1.2, pandas_engine.Decimal(2, 1), Decimal("1.2")),
        (1, pandas_engine.Decimal(2, 1), Decimal("1.0")),
        (1, pandas_engine.Decimal(), Decimal("1")),
        (pd.NA, pandas_engine.Decimal(2, 1), pd.NA),
        (None, pandas_engine.Decimal(2, 1), pd.NA),
        (np.nan, pandas_engine.Decimal(2, 1), pd.NA),
    ],
)
def test_logical_datatype_coerce_value(
    data,
    datatype: pandas_engine.DataType,
    expected_value: Any,
):
    """Test decimal coerce."""
    coerced_value = datatype.coerce_value(data)
    if pd.isna(expected_value):
        assert pd.isna(coerced_value)
    else:
        assert coerced_value == expected_value


@pytest.mark.parametrize("precision,scale", [(-1, None), (0, 0), (1, 2)])
def test_invalid_decimal_params(
    precision: Optional[int], scale: Optional[int]
):
    """Test invalid decimal params."""
    with pytest.raises(ValueError):
        pa.Decimal(precision, scale)
