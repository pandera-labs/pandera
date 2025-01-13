"""Check backend for polars."""

from functools import partial
from typing import Optional

import polars as pl
from polars.lazyframe.group_by import LazyGroupBy

from pandera.api.base.checks import CheckResult
from pandera.api.checks import Check
from pandera.api.polars.types import PolarsData
from pandera.api.polars.utils import (
    get_lazyframe_schema,
    get_lazyframe_column_names,
)
from pandera.backends.base import BaseCheckBackend
from pandera.constants import CHECK_OUTPUT_KEY


class PolarsCheckBackend(BaseCheckBackend):
    """Check backend for polars."""

    def __init__(self, check: Check):
        """Initializes a check backend object."""
        super().__init__(check)
        assert check._check_fn is not None, "Check._check_fn must be set."
        self.check = check
        self.check_fn = partial(check._check_fn, **check._check_kwargs)

    def groupby(self, check_obj: pl.LazyFrame) -> LazyGroupBy:
        """Implements groupby behavior for check object."""
        raise NotImplementedError

    def query(self, check_obj: pl.LazyFrame):
        """Implements querying behavior to produce subset of check object."""
        raise NotImplementedError

    def aggregate(self, check_obj: pl.LazyFrame):
        """Implements aggregation behavior for check object."""
        raise NotImplementedError

    def preprocess(self, check_obj: pl.LazyFrame, key: Optional[str]):
        """Preprocesses a check object before applying the check function."""
        return check_obj

    def apply(self, check_obj: PolarsData):
        """Apply the check function to a check object."""
        if self.check.element_wise:
            selector = pl.col(check_obj.key or "*")
            out = check_obj.lazyframe.with_columns(
                selector.map_elements(self.check_fn, return_dtype=pl.Boolean)
            ).select(selector)
        else:
            out = self.check_fn(check_obj)

        if isinstance(out, bool):
            return out

        if len(get_lazyframe_schema(out)) > 1:
            # for checks that return a boolean dataframe, reduce to a single
            # boolean column.
            out = out.select(
                pl.fold(
                    acc=pl.lit(True),
                    function=lambda acc, x: acc & x,
                    exprs=pl.col("*"),
                ).alias(CHECK_OUTPUT_KEY)
            )
        else:
            out = out.rename(
                {get_lazyframe_column_names(out)[0]: CHECK_OUTPUT_KEY}
            )

        return out

    def postprocess(self, check_obj, check_output):
        """Postprocesses the result of applying the check function."""
        if isinstance(check_output, pl.LazyFrame):
            return self.postprocess_lazyframe_output(check_obj, check_output)
        elif isinstance(check_output, bool):
            return self.postprocess_bool_output(check_obj, check_output)
        raise TypeError(  # pragma: no cover
            f"output type of check_fn not recognized: {type(check_output)}"
        )

    def postprocess_lazyframe_output(
        self,
        check_obj: PolarsData,
        check_output: pl.LazyFrame,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        results = pl.LazyFrame(check_output.collect())
        passed = results.select([pl.col(CHECK_OUTPUT_KEY).all()])
        failure_cases = pl.concat(
            [check_obj.lazyframe, results], how="horizontal"
        ).filter(pl.col(CHECK_OUTPUT_KEY).not_())

        if check_obj.key is not None:
            failure_cases = failure_cases.select(check_obj.key)
        return CheckResult(
            check_output=results,
            check_passed=passed,
            checked_object=check_obj,
            failure_cases=failure_cases,
        )

    def postprocess_bool_output(
        self,
        check_obj: PolarsData,
        check_output: bool,
    ) -> CheckResult:
        """Postprocesses the result of applying the check function."""
        ldf_output = pl.LazyFrame({CHECK_OUTPUT_KEY: [check_output]})
        return CheckResult(
            check_output=ldf_output,
            check_passed=ldf_output,
            checked_object=check_obj,
            failure_cases=None,
        )

    def __call__(
        self,
        check_obj: pl.LazyFrame,
        key: Optional[str] = None,
    ) -> CheckResult:
        check_obj = self.preprocess(check_obj, key)
        polars_data = PolarsData(check_obj, key)
        check_output = self.apply(polars_data)
        return self.postprocess(polars_data, check_output)
