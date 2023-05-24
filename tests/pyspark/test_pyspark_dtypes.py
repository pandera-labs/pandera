"""Unit tests for pyspark container."""

from typing import Any
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from pandera.pyspark import DataFrameSchema, Column
from tests.pyspark.conftest import spark_df
from pandera.backends.pyspark.utils import ConfigParams
from pandera.backends.pyspark.decorators import validate_scope


class BaseClass:
    """Base class for all the dtypes"""

    params: Any = ConfigParams()

    def validate_datatype(self, df, pandera_schema):
        """
        This function validates the dataframe schema and pandera defined schema to ensure both work
        """
        df_out = pandera_schema(df)

        assert df.pandera.schema == pandera_schema
        assert isinstance(pandera_schema.validate(df), DataFrame)
        assert isinstance(pandera_schema(df), DataFrame)
        return df_out

    def pytest_generate_tests(self, metafunc):
        """This function runs for each test class and maps the input parameters for each test function in a class"""
        # called once per each test function
        funcarglist = metafunc.cls.params[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def validate_data(
        self, df, pandera_equivalent, column_name, return_error=False
    ):
        """
        This function runs the actual validation of object on the dataframe
        """
        pandera_schema = DataFrameSchema(
            columns={
                column_name: Column(pandera_equivalent),
            },
        )
        df_out = self.validate_datatype(df, pandera_schema)
        if df_out.pandera.errors:
            if return_error:
                return df_out.pandera.errors
            else:
                print(df_out.pandera.errors)
                raise Exception


class TestAllNumericTypes(BaseClass):
    """This class is to test all the numeric types"""

    # a map specifying multiple argument sets for a test method
    params = {
        "test_pyspark_all_float_types": [
            {"pandera_equivalent": float},
            {"pandera_equivalent": "FloatType()"},
            {"pandera_equivalent": T.FloatType()},
            {"pandera_equivalent": T.FloatType},
            {"pandera_equivalent": "float"},
        ],
        "test_pyspark_decimal_default_types": [
            {"pandera_equivalent": "decimal"},
            {"pandera_equivalent": "DecimalType()"},
            {"pandera_equivalent": T.DecimalType},
            {"pandera_equivalent": T.DecimalType()},
        ],
        "test_pyspark_decimal_parameterized_types": [
            {
                "pandera_equivalent": {
                    "parameter_match": T.DecimalType(20, 5),
                    "parameter_mismatch": T.DecimalType(20, 3),
                }
            }
        ],
        "test_pyspark_all_double_types": [
            {"pandera_equivalent": T.DoubleType()},
            {"pandera_equivalent": T.DoubleType},
            {"pandera_equivalent": "double"},
            {"pandera_equivalent": "DoubleType()"},
        ],
        "test_pyspark_all_int_types": [
            {"pandera_equivalent": int},
            {"pandera_equivalent": "int"},
            {"pandera_equivalent": "IntegerType()"},
            {"pandera_equivalent": T.IntegerType()},
            {"pandera_equivalent": T.IntegerType},
        ],
        "test_pyspark_all_longint_types": [
            {"pandera_equivalent": "bigint"},
            {"pandera_equivalent": "long"},
            {"pandera_equivalent": T.LongType},
            {"pandera_equivalent": T.LongType()},
            {"pandera_equivalent": "LongType()"},
        ],
        "test_pyspark_all_shortint_types": [
            {"pandera_equivalent": "ShortType()"},
            {"pandera_equivalent": T.ShortType},
            {"pandera_equivalent": T.ShortType()},
            {"pandera_equivalent": "short"},
            {"pandera_equivalent": "smallint"},
        ],
        "test_pyspark_all_bytetint_types": [
            {"pandera_equivalent": "ByteType()"},
            {"pandera_equivalent": T.ByteType},
            {"pandera_equivalent": T.ByteType()},
            {"pandera_equivalent": "bytes"},
            {"pandera_equivalent": "tinyint"},
        ],
    }

    def create_schema(self, column_name, datatype):
        """Create schmea for a column and datatype"""
        spark_schema = T.StructType(
            [
                T.StructField(column_name, datatype, False),
            ],
        )
        return spark_schema

    def test_pyspark_all_float_types(
        self, spark, sample_data, pandera_equivalent
    ):
        """
        Test float dtype column
        """
        column_name = "price"
        spark_schema = self.create_schema(column_name, T.FloatType())
        df = spark_df(spark, sample_data, spark_schema)
        self.validate_data(df, pandera_equivalent, column_name)

    def test_pyspark_all_double_types(
        self, spark, sample_data, pandera_equivalent
    ):
        """
        Test double dtype column
        """
        column_name = "price"
        spark_schema = self.create_schema(column_name, T.DoubleType())
        df = spark_df(spark, sample_data, spark_schema)
        self.validate_data(df, pandera_equivalent, column_name)

    def test_pyspark_decimal_default_types(
        self, spark, sample_data, pandera_equivalent
    ):
        """
        Test decimal dtype column with default values
        """
        column_name = "price"
        spark_schema = self.create_schema(column_name, T.DecimalType())
        df = spark_df(spark, sample_data, spark_schema)
        self.validate_data(df, pandera_equivalent, column_name)

    @validate_scope(params=BaseClass.params, scope="SCHEMA")
    def test_pyspark_decimal_parameterized_types(
        self, spark, sample_data, pandera_equivalent
    ):
        """
        Test decimal dtype column with parameterized inputs
        """
        column_name = "price"
        spark_schema = self.create_schema(column_name, T.DecimalType(20, 5))
        df = spark_df(spark, sample_data, spark_schema)
        self.validate_data(
            df, pandera_equivalent["parameter_match"], column_name
        )
        errors = self.validate_data(
            df, pandera_equivalent["parameter_mismatch"], column_name, True
        )
        assert dict(errors["SCHEMA"]) == {
            "WRONG_DATATYPE": [
                {
                    "schema": None,
                    "column": "price",
                    "check": "dtype('DecimalType(20,3)')",
                    "error": "expected column 'price' to have type DecimalType(20,3), "
                    "got DecimalType(20,5)",
                }
            ]
        }

    def test_pyspark_all_int_types(
        self, spark, sample_data, pandera_equivalent
    ):
        """
        Test int dtype column
        """
        column_name = "price"
        spark_schema = self.create_schema(column_name, T.IntegerType())
        df = spark_df(spark, sample_data, spark_schema)
        self.validate_data(df, pandera_equivalent, column_name)

    def test_pyspark_all_longint_types(
        self, spark, sample_data, pandera_equivalent
    ):
        """
        Test long dtype column
        """
        column_name = "price"
        spark_schema = self.create_schema(column_name, T.LongType())
        df = spark_df(spark, sample_data, spark_schema)
        self.validate_data(df, pandera_equivalent, column_name)

    def test_pyspark_all_shortint_types(
        self, spark, sample_data, pandera_equivalent
    ):
        """
        Test short int dtype column
        """
        column_name = "price"
        spark_schema = self.create_schema(column_name, T.ShortType())
        df = spark_df(spark, sample_data, spark_schema)
        self.validate_data(df, pandera_equivalent, column_name)

    def test_pyspark_all_bytetint_types(
        self, spark, sample_data, pandera_equivalent
    ):
        """
        Test byte int dtype column
        """
        column_name = "price"
        spark_schema = self.create_schema(column_name, T.ByteType())
        df = spark_df(spark, sample_data, spark_schema)
        self.validate_data(df, pandera_equivalent, column_name)


class TestAllDatetimeTestClass(BaseClass):
    """This class is to test all the datetime types"""

    # a map specifying multiple argument sets for a test method
    params = {
        "test_pyspark_all_date_types": [
            {"pandera_equivalent": T.DateType},
            {"pandera_equivalent": "DateType()"},
            {"pandera_equivalent": T.DateType()},
            {"pandera_equivalent": "date"},
        ],
        "test_pyspark_all_datetime_types": [
            {"pandera_equivalent": T.TimestampType},
            {"pandera_equivalent": "TimestampType()"},
            {"pandera_equivalent": T.TimestampType()},
            {"pandera_equivalent": "datetime"},
            {"pandera_equivalent": "timestamp"},
        ],
        "test_pyspark_all_daytimeinterval_types": [
            {"pandera_equivalent": T.DayTimeIntervalType},
            {"pandera_equivalent": "timedelta"},
            {"pandera_equivalent": T.DayTimeIntervalType()},
            {"pandera_equivalent": "DayTimeIntervalType()"},
        ],
        "test_pyspark_daytimeinterval_param_mismatch": [
            {"pandera_equivalent": T.DayTimeIntervalType(1, 3)},
        ],
    }

    def test_pyspark_all_date_types(
        self, pandera_equivalent, sample_date_object
    ):
        """
        Test date dtype column
        """
        column_name = "purchase_date"
        df = sample_date_object.select(column_name)
        self.validate_data(df, pandera_equivalent, column_name)

    def test_pyspark_all_datetime_types(
        self, pandera_equivalent, sample_date_object
    ):
        """
        Test datetime dtype column
        """
        column_name = "purchase_datetime"
        df = sample_date_object.select(column_name)
        self.validate_data(df, pandera_equivalent, column_name)

    def test_pyspark_all_daytimeinterval_types(
        self, pandera_equivalent, sample_date_object
    ):
        """
        Test daytimeinterval dtype column
        """
        column_name = "expiry_time"
        df = sample_date_object.select(column_name)
        self.validate_data(df, pandera_equivalent, column_name)

    @validate_scope(params=BaseClass.params, scope="SCHEMA")
    def test_pyspark_daytimeinterval_param_mismatch(
        self, pandera_equivalent, sample_date_object
    ):
        """
        Test daytimeinterval parameter mismatch
        """
        column_name = "expected_time"
        df = sample_date_object.select(column_name)
        errors = self.validate_data(df, pandera_equivalent, column_name, True)
        assert dict(errors["SCHEMA"]) == {
            "WRONG_DATATYPE": [
                {
                    "schema": None,
                    "column": "expected_time",
                    "check": "dtype('DayTimeIntervalType(1, 3)')",
                    "error": "expected column 'expected_time' to have type DayTimeIntervalType(1, 3), "
                    "got DayTimeIntervalType(2, 3)",
                }
            ]
        }


class TestBinaryStringTypes(BaseClass):
    """Test the binary type data types"""

    # a map specifying multiple argument sets for a test method
    params = {
        "test_pyspark_all_binary_types": [
            {"pandera_equivalent": "binary"},
            {"pandera_equivalent": "BinaryType()"},
            {"pandera_equivalent": T.BinaryType()},
            {"pandera_equivalent": T.BinaryType},
        ],
        "test_pyspark_all_string_types": [
            {"pandera_equivalent": str},
            {"pandera_equivalent": "string"},
            {"pandera_equivalent": "StringType()"},
            {"pandera_equivalent": T.StringType()},
            {"pandera_equivalent": T.StringType},
        ],
    }

    def test_pyspark_all_binary_types(
        self, pandera_equivalent, sample_string_binary_object
    ):
        """
        Test binary dytpe column
        """
        column_name = "purchase_info"
        df = sample_string_binary_object.select(column_name)
        self.validate_data(df, pandera_equivalent, column_name)

    def test_pyspark_all_string_types(
        self, pandera_equivalent, sample_string_binary_object
    ):
        """
        Test string dtype column
        """
        column_name = "product"
        df = sample_string_binary_object.select(column_name)
        self.validate_data(df, pandera_equivalent, column_name)


class TestComplexType(BaseClass):
    """This class is to test all the complex types"""

    params = {
        "test_pyspark_array_type": [
            {
                "pandera_equivalent": {
                    "schema_match": T.ArrayType(T.ArrayType(T.StringType())),
                    "schema_mismatch": T.ArrayType(
                        T.ArrayType(T.IntegerType())
                    ),
                }
            }
        ],
        "test_pyspark_map_type": [
            {
                "pandera_equivalent": {
                    "schema_match": T.MapType(T.StringType(), T.StringType()),
                    "schema_mismatch": T.MapType(
                        T.StringType(), T.IntegerType()
                    ),
                }
            }
        ],
    }

    @validate_scope(params=BaseClass.params, scope="SCHEMA")
    def test_pyspark_array_type(self, sample_complex_data, pandera_equivalent):
        """
        Test array dtype column
        """
        column_name = "customer_details"
        df = sample_complex_data.select(column_name)
        self.validate_data(df, pandera_equivalent["schema_match"], column_name)
        errors = self.validate_data(
            df, pandera_equivalent["schema_mismatch"], column_name, True
        )
        assert dict(errors["SCHEMA"]) == {
            "WRONG_DATATYPE": [
                {
                    "schema": None,
                    "column": "customer_details",
                    "check": "dtype('ArrayType(ArrayType(IntegerType(), True), True)')",
                    "error": "expected column 'customer_details' to have type ArrayType(ArrayType(IntegerType(), True), True), got ArrayType(ArrayType(StringType(), True), True)",
                }
            ]
        }

    @validate_scope(params=BaseClass.params, scope="SCHEMA")
    def test_pyspark_map_type(self, sample_complex_data, pandera_equivalent):
        """
        Test map dtype column
        """
        column_name = "product_details"
        df = sample_complex_data.select(column_name)
        self.validate_data(df, pandera_equivalent["schema_match"], column_name)
        errors = self.validate_data(
            df, pandera_equivalent["schema_mismatch"], column_name, True
        )
        assert dict(errors["SCHEMA"]) == {
            "WRONG_DATATYPE": [
                {
                    "schema": None,
                    "column": "product_details",
                    "check": "dtype('MapType(StringType(), IntegerType(), True)')",
                    "error": "expected column 'product_details' to have type MapType(StringType(), IntegerType(), True), got MapType(StringType(), StringType(), True)",
                }
            ]
        }
