"""This module is to test the behaviour change based on defined config in pandera"""
# pylint:disable=import-outside-toplevel,abstract-method

from contextlib import nullcontext as does_not_raise
import logging
import pyspark.sql.types as T
from pyspark.sql import DataFrame
import pytest

from pandera.backends.pyspark.decorators import cache_check_obj
from pandera.config import CONFIG
from pandera.pyspark import (
    Check,
    DataFrameSchema,
    Column,
)
from tests.pyspark.conftest import spark_df


class TestPanderaDecorators:
    """Class to test all the different configs types"""

    sample_data = [("Bread", 9), ("Cutter", 15)]

    def test_cache_dataframe_requirements(self, spark, sample_spark_schema):
        """Validates if decorator can only be applied in a proper function."""
        # Set expected properties in Config object
        CONFIG.cache_dataframe = True
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)

        class FakeDataFrameSchemaBackend:
            """Class that simulates DataFrameSchemaBackend class."""

            @cache_check_obj()
            def func_w_check_obj_args(self, check_obj: DataFrame, /):
                """Right function to use this decorator, check_obj as arg."""
                return check_obj.columns

            @cache_check_obj()
            def func_w_check_obj_kwargs(self, *, check_obj: DataFrame = None):
                """Right function to use this decorator, check_obj as kwarg."""
                return check_obj.columns

            @cache_check_obj()
            def func_wo_check_obj(self, message: str):
                """Wrong function to use this decorator."""
                return message

        # Check for a function that does have a `check_obj` as arg
        with does_not_raise():
            instance = FakeDataFrameSchemaBackend()
            _ = instance.func_w_check_obj_args(input_df)

        # Check for a function that does have a `check_obj` as kwarg
        with does_not_raise():
            instance = FakeDataFrameSchemaBackend()
            _ = instance.func_w_check_obj_kwargs(check_obj=input_df)

        # Check for a wrong function, that does not have a `check_obj`
        with pytest.raises(ValueError):
            instance = FakeDataFrameSchemaBackend()
            _ = instance.func_wo_check_obj("wrong")

    @pytest.mark.parametrize(
        "cache_enabled,keep_cache_enabled,"
        "expected_caching_message,expected_unpersisting_message",
        [
            (True, True, True, None),
            (True, False, True, True),
            (False, True, None, None),
            (False, False, None, None),
        ],
        scope="function",
    )

    # pylint:disable=too-many-locals
    def test_cache_dataframe_settings(
        self,
        spark,
        sample_spark_schema,
        cache_enabled,
        keep_cache_enabled,
        expected_caching_message,
        expected_unpersisting_message,
        caplog,
    ):
        """This function validates that caching/unpersisting works as expected."""
        # Set expected properties in Config object
        CONFIG.cache_dataframe = cache_enabled
        CONFIG.keep_cached_dataframe = keep_cache_enabled

        # Prepare test data
        input_df = spark_df(spark, self.sample_data, sample_spark_schema)
        pandera_schema = DataFrameSchema(
            {
                "product": Column(T.StringType(), Check.str_startswith("B")),
                "price_val": Column(T.IntegerType()),
            }
        )

        # Capture log message
        with caplog.at_level(logging.DEBUG, logger="pandera"):
            df_out = pandera_schema.validate(input_df)

        # Assertions
        assert isinstance(df_out, DataFrame)

        CACHE_MESSAGE = "Caching dataframe..."
        UNPERSIST_MESSAGE = "Unpersisting dataframe..."
        if expected_caching_message:
            assert (
                CACHE_MESSAGE in caplog.text
            ), "Debugging info has no information about caching the dataframe."
        else:
            assert (
                CACHE_MESSAGE not in caplog.text
            ), "Debugging info has information about caching. It shouldn't."

        if expected_unpersisting_message:
            assert (
                UNPERSIST_MESSAGE in caplog.text
            ), "Debugging info has no information about unpersisting the dataframe."
        else:
            assert (
                UNPERSIST_MESSAGE not in caplog.text
            ), "Debugging info has information about unpersisting. It shouldn't."
