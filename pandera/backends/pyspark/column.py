"""Pandera array backends."""

import traceback
from typing import Iterable, NamedTuple, Optional, cast

from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from pandera.api.base.error_handler import ErrorCategory, ErrorHandler
from pandera.backends.pyspark.base import PysparkSchemaBackend
from pandera.backends.pyspark.decorators import validate_scope
from pandera.backends.pyspark.error_formatters import scalar_failure_case
from pandera.engines.pyspark_engine import Engine
from pandera.errors import ParserError, SchemaError, SchemaErrorReason
from pandera.validation_depth import ValidationScope


class CoreCheckResult(NamedTuple):
    """Namedtuple for holding results of core checks."""

    check: str
    reason_code: SchemaErrorReason
    passed: bool
    message: Optional[str] = None
    failure_cases: Optional[Iterable] = None


class ColumnSchemaBackend(PysparkSchemaBackend):
    """Backend for pyspark arrays."""

    def preprocess(self, check_obj, inplace: bool = False):
        return check_obj

    @validate_scope(scope=ValidationScope.SCHEMA)
    def _core_checks(self, check_obj, schema, error_handler):
        """This function runs the core checks"""
        # run the core checks
        for core_check in (
            self.check_name,
            self.check_dtype,
            self.check_nullable,
        ):
            check_result = core_check(check_obj, schema)
            if not check_result.passed:
                error_handler.collect_error(
                    ErrorCategory.SCHEMA,
                    check_result.reason_code,
                    SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=check_result.message,
                        failure_cases=check_result.failure_cases,
                        check=check_result.check,
                        reason_code=check_result.reason_code,
                    ),
                )

    def validate(
        self,
        check_obj,
        schema,
        *,
        head: Optional[int] = None,  # pylint: disable=unused-argument
        tail: Optional[int] = None,  # pylint: disable=unused-argument
        sample: Optional[int] = None,  # pylint: disable=unused-argument
        random_state: Optional[int] = None,  # pylint: disable=unused-argument
        lazy: bool = False,
        inplace: bool = False,
        error_handler: ErrorHandler = None,
    ):
        # pylint: disable=too-many-locals
        check_obj = self.preprocess(check_obj, inplace)

        if schema.coerce:
            try:
                check_obj = (
                    self.coerce_dtype(  # pylint:disable=unexpected-keyword-arg
                        check_obj, schema=schema, error_handler=error_handler
                    )
                )
            except SchemaError as exc:
                assert (
                    error_handler is not None
                ), "The `error_handler` argument must be provided."
                error_handler.collect_error(
                    ErrorCategory.SCHEMA, exc.reason_code, exc
                )

        self._core_checks(check_obj, schema, error_handler)

        self.run_checks(check_obj, schema, error_handler, lazy)

        return check_obj

    @validate_scope(scope=ValidationScope.SCHEMA)
    def coerce_dtype(
        self,
        check_obj,
        *,
        schema=None,
        # pylint: disable=unused-argument
    ):
        """Coerce type of a pyspark.sql.function.col by type specified in dtype.

        :param check_obj: Pyspark DataFrame
        :returns: ``DataFrame`` with coerced data type
        """
        assert schema is not None, "The `schema` argument must be provided."
        if schema.dtype is None or not schema.coerce:
            return check_obj

        try:
            return schema.dtype.try_coerce(check_obj)
        except ParserError as exc:
            raise SchemaError(
                schema=schema,
                data=check_obj,
                message=(
                    f"Error while coercing '{schema.name}' to type "
                    f"{schema.dtype}: {exc}:\n{exc.failure_cases}"
                ),
                failure_cases=exc.failure_cases,
                check=f"coerce_dtype('{schema.dtype}')",
            ) from exc

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_nullable(self, check_obj: DataFrame, schema):
        passed = True

        # Use schema level information to optimize execution of the `nullable` check:
        # ignore this check if Pandera Field's `nullable` property is True
        # (check not necessary) or if df column's `nullable` property is False
        # (PySpark's nullable ensures the presence of values when creating the df)
        if (not schema.nullable) and (check_obj.schema[schema.name].nullable):
            passed = (
                check_obj.filter(col(schema.name).isNull()).limit(1).count()
                == 0
            )

        return CoreCheckResult(
            check="not_nullable",
            reason_code=SchemaErrorReason.SERIES_CONTAINS_NULLS,
            passed=cast(bool, passed),
            message=(f"non-nullable column '{schema.name}' contains null"),
            failure_cases=scalar_failure_case(schema.name),
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_name(self, check_obj: DataFrame, schema):
        column_found = not (
            schema.name is None or schema.name not in check_obj.columns
        )
        return CoreCheckResult(
            check=f"field_name('{schema.name}')",
            reason_code=(
                SchemaErrorReason.WRONG_FIELD_NAME
                if not column_found
                else SchemaErrorReason.NO_ERROR
            ),
            passed=column_found,
            message=(
                f"Expected {type(check_obj)} to have column named: '{schema.name}', "
                f"but found columns '{check_obj.columns}'"
                if not column_found
                else "column check_name validation passed."
            ),
            failure_cases=(
                scalar_failure_case(schema.name) if not column_found else None
            ),
        )

    @validate_scope(scope=ValidationScope.SCHEMA)
    def check_dtype(self, check_obj: DataFrame, schema):
        passed = True
        failure_cases = None
        msg = None

        if schema.dtype is not None:
            dtype_check_results = schema.dtype.check(
                Engine.dtype(
                    check_obj.schema[schema.name].dataType
                ),  # pylint: disable=no-value-for-parameter
            )

            if isinstance(dtype_check_results, bool):
                passed = dtype_check_results
                failure_cases = scalar_failure_case(
                    str(
                        Engine.dtype(check_obj.schema[schema.name].dataType)
                    )  # pylint:disable=no-value-for-parameter
                )
                msg = (
                    f"expected column '{schema.name}' to have type "  # pylint:disable=no-value-for-parameter
                    f"{schema.dtype}, got {Engine.dtype(check_obj.schema[schema.name].dataType)}"  # pylint:disable=no-value-for-parameter
                    if not passed
                    else f"column type matched with expected '{schema.dtype}'"
                )
            reason_code = (
                SchemaErrorReason.WRONG_DATATYPE
                if not dtype_check_results
                else SchemaErrorReason.NO_ERROR
            )

        return CoreCheckResult(
            check=f"dtype('{schema.dtype}')",
            reason_code=reason_code,  # pylint:disable=possibly-used-before-assignment
            passed=passed,
            message=msg,
            failure_cases=failure_cases,
        )

    @validate_scope(scope=ValidationScope.DATA)
    # pylint: disable=unused-argument
    def run_checks(self, check_obj, schema, error_handler, lazy):
        check_results = []
        for check_index, check in enumerate(schema.checks):
            check_args = [schema.name]
            try:
                check_results.append(
                    self.run_check(
                        check_obj,
                        schema,
                        check,
                        check_index,
                        *check_args,
                    )
                )
            except SchemaError as err:
                error_handler.collect_error(
                    ErrorCategory.DATA,
                    SchemaErrorReason.DATAFRAME_CHECK,
                    err,
                )
            except Exception as err:  # pylint: disable=broad-except
                # catch other exceptions that may occur when executing the Check
                err_msg = f'"{err.args[0]}"' if len(err.args) > 0 else ""
                err_str = f"{err.__class__.__name__}({ err_msg})"
                error_handler.collect_error(
                    ErrorCategory.DATA,
                    SchemaErrorReason.CHECK_ERROR,
                    SchemaError(
                        schema=schema,
                        data=check_obj,
                        message=(
                            f"Error while executing check function: {err_str}\n"
                            + traceback.format_exc()
                        ),
                        failure_cases=scalar_failure_case(err_str),
                        check=check,
                        check_index=check_index,
                    ),
                    original_exc=err,
                )

        return check_results
