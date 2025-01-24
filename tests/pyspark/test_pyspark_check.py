"""Unit tests for pyspark container."""

# pylint:disable=abstract-method
import datetime
import decimal
from unittest import mock

import pytest
from pyspark.sql.functions import col
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    MapType,
    ShortType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

import pandera.extensions
import pandera.pyspark as pa
from pandera.backends.pyspark.decorators import validate_scope
from pandera.errors import PysparkSchemaError
from pandera.pyspark import Column, DataFrameModel, DataFrameSchema, Field
from pandera.validation_depth import ValidationScope

pytestmark = pytest.mark.parametrize(
    "spark_session", ["spark", "spark_connect"]
)


@pytest.fixture(scope="function")
def extra_registered_checks():
    """temporarily registers custom checks onto the Check class"""

    # pylint: disable=unused-variable
    with mock.patch(
        "pandera.Check.REGISTERED_CUSTOM_CHECKS", new_callable=dict
    ):

        @pandera.extensions.register_check_method
        def new_pyspark_check(pyspark_obj, *, max_value) -> bool:
            """Ensure values of a series are strictly below a maximum value.
            :param data: PysparkDataframeColumnObject column object which is a contains dataframe and column name to do the check
            :param max_value: Upper bound not to be exceeded. Must be
                a type comparable to the dtype of the column datatype of pyspark
            """
            # test case exists but not detected by pytest so no cover added
            cond = col(pyspark_obj.column_name) <= max_value
            return pyspark_obj.dataframe.filter(~cond).limit(1).count() == 0

        yield


class TestDecorator:
    """This class is used to test the decorator to check datatype mismatches and unacceptable datatype"""

    @validate_scope(scope=ValidationScope.DATA)
    def test_datatype_check_decorator(self, spark_session, request):
        """
        Test to validate the decorator to check datatype mismatches and unacceptable datatype
        """
        spark = request.getfixturevalue(spark_session)
        schema = DataFrameSchema(
            {
                "product": Column(StringType()),
                "code": Column(StringType(), pa.Check.str_startswith("B")),
            }
        )

        spark_schema = StructType(
            [
                StructField("product", StringType(), False),
                StructField("code", StringType(), False),
            ],
        )
        pass_case_data = [["foo", "B1"], ["bar", "B2"]]
        df = spark.createDataFrame(data=pass_case_data, schema=spark_schema)
        df_out = schema.validate(df)
        if df_out.pandera.errors:
            print(df_out.pandera.errors)
            raise PysparkSchemaError

        fail_schema = DataFrameSchema(
            {
                "product": Column(StringType()),
                "code": Column(IntegerType(), pa.Check.str_startswith("B")),
            }
        )

        spark_schema = StructType(
            [
                StructField("product", StringType(), False),
                StructField("code", IntegerType(), False),
            ],
        )
        fail_case_data = [["foo", 1], ["bar", 2]]
        df = spark.createDataFrame(data=fail_case_data, schema=spark_schema)
        df_out = schema.validate(df)
        expected = {
            "DATA": {
                "DATAFRAME_CHECK": [
                    {
                        "check": None,
                        "column": None,
                        "error": 'The check with name "str_startswith" '
                        "was expected to be run for string but got integer "
                        "instead from the input. This error is usually caused "
                        "by schema mismatch the value is different from schema "
                        "defined in pandera schema and one in the dataframe",
                        "schema": None,
                    }
                ]
            },
            "SCHEMA": {
                "WRONG_DATATYPE": [
                    {
                        "check": f"dtype('{str(StringType())}')",
                        "column": "code",
                        "error": "expected "
                        "column "
                        "'code' to "
                        "have type "
                        f"{str(StringType())}, "
                        "got "
                        f"{str(IntegerType())}",
                        "schema": None,
                    }
                ]
            },
        }
        assert dict(df_out.pandera.errors["DATA"]) == expected["DATA"]
        assert dict(df_out.pandera.errors["SCHEMA"]) == expected["SCHEMA"]

        df = spark.createDataFrame(data=fail_case_data, schema=spark_schema)
        try:
            df_out = fail_schema.validate(df)
        except TypeError as err:
            assert (
                err.__str__()
                == 'The check with name "str_startswith" only supports the following datatypes [\'string\'] and not the given "integer" datatype'
            )


class BaseClass:
    """This is the base class for the all the test cases class"""

    def __int__(self, params=None):
        pass

    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_expression": "a",
    }

    sample_array_data = {
        "test_pass_data": [("foo", ["a"]), ("bar", ["a"])],
        "test_expression": "a",
    }

    sample_map_data = {
        "test_pass_data": [("foo", {"a": "a"}), ("bar", {"b": "b"})],
        "test_expression": "b",
    }

    sample_bolean_data = {
        "test_pass_data": [("foo", True), ("bar", True)],
        "test_expression": False,
    }

    def pytest_generate(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        raise NotImplementedError

    @staticmethod
    def convert_value(sample_data, conversion_datatype):
        """
        Convert the sample data to other formats excluding dates and does not
        support complex datatypes such as array and map as of now
        """
        data_dict = {}
        for key, value in sample_data.items():
            if key == "test_expression":
                if not isinstance(value, list):
                    data_dict[key] = conversion_datatype(value)
                else:
                    data_dict[key] = [conversion_datatype(i) for i in value]

            else:
                if not isinstance(value[0][1], list):
                    data_dict[key] = [
                        (i[0], conversion_datatype(i[1])) for i in value
                    ]
                else:
                    final_val = []
                    for row in value:
                        data_val = []
                        for column in row[1]:
                            data_val.append(conversion_datatype(column))
                        final_val.append((row[0], data_val))
                    data_dict[key] = final_val
        return data_dict

    @staticmethod
    def convert_numeric_data(sample_data, convert_type):
        """
        Convert the numeric data to required format
        """
        if convert_type in ("double", "float"):
            data_dict = BaseClass.convert_value(sample_data, float)

        if convert_type == "decimal":
            data_dict = BaseClass.convert_value(sample_data, decimal.Decimal)

        return data_dict  # pylint:disable=possibly-used-before-assignment

    @staticmethod
    def convert_timestamp_to_date(sample_data):
        """
        Convert the sample data of timestamp type to date type
        """
        data_dict = {}
        for key, value in sample_data.items():
            if key == "test_expression":
                if not isinstance(value, list):
                    data_dict[key] = value.date()
                else:
                    data_dict[key] = [i.date() for i in value]

            else:
                if not isinstance(value[0][1], list):
                    data_dict[key] = [(i[0], i[1].date()) for i in value]
                else:
                    final_val = []
                    for row in value:
                        data_val = []
                        for column in row[1]:
                            data_val.append(column.date())
                        final_val.append((row[0], data_val))
                    data_dict[key] = final_val
        return data_dict

    @staticmethod
    def check_function(
        spark,
        check_fn,
        pass_case_data,
        fail_case_data,
        data_types,
        function_args,
    ):
        """
        This function does performs the actual validation
        """
        schema = DataFrameSchema(
            {
                "product": Column(StringType()),
                "code": (
                    Column(data_types, check_fn(*function_args))
                    if isinstance(function_args, tuple)
                    else Column(data_types, check_fn(function_args))
                ),
            }
        )
        spark_schema = StructType(
            [
                StructField("product", StringType(), False),
                StructField("code", data_types, False),
            ],
        )
        df = spark.createDataFrame(data=pass_case_data, schema=spark_schema)
        df_out = schema.validate(df)
        if df_out.pandera.errors:
            print(df_out.pandera.errors)
            raise PysparkSchemaError

        with pytest.raises(PysparkSchemaError):
            df_fail = spark.createDataFrame(
                data=fail_case_data, schema=spark_schema
            )
            df_out = schema.validate(df_fail)
            if df_out.pandera.errors:
                raise PysparkSchemaError


class TestEqualToCheck(BaseClass):
    """This class is used to test the equal to check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 30), ("bar", 30)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }

    sample_timestamp_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 1, 10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 2, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 10, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 1, 10, 0),
    }

    sample_string_data = {
        "test_pass_data": [("foo", "a"), ("bar", "a")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": "a",
    }

    sample_bolean_data = {
        "test_pass_data": [("foo", True), ("bar", True)],
        "test_fail_data": [("foo", False), ("bar", False)],
        "test_expression": True,
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        return {
            "test_equal_to_check": [
                {"datatype": LongType(), "data": self.sample_numeric_data},
                {"datatype": IntegerType(), "data": self.sample_numeric_data},
                {"datatype": ByteType(), "data": self.sample_numeric_data},
                {"datatype": ShortType(), "data": self.sample_numeric_data},
                {"datatype": StringType(), "data": self.sample_string_data},
                {
                    "datatype": DoubleType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "double"
                    ),
                },
                {
                    "datatype": TimestampType(),
                    "data": self.sample_timestamp_data,
                },
                {
                    "datatype": DateType(),
                    "data": self.convert_timestamp_to_date(
                        self.sample_timestamp_data
                    ),
                },
                {
                    "datatype": DecimalType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "decimal"
                    ),
                },
                {
                    "datatype": FloatType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "float"
                    ),
                },
                {"datatype": BooleanType(), "data": self.sample_bolean_data},
            ],
            "test_failed_unaccepted_datatypes": [
                {
                    "datatype": ArrayType(StringType()),
                    "data": self.sample_array_data,
                },
                {
                    "datatype": MapType(StringType(), StringType()),
                    "data": self.sample_map_data,
                },
            ],
        }

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize("check_fn", [pa.Check.equal_to, pa.Check.eq])
    def test_equal_to_check(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if all the values are equal to defined value"""
        spark = request.getfixturevalue(spark_session)
        self.check_function(
            spark,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize("check_fn", [pa.Check.equal_to, pa.Check.eq])
    def test_failed_unaccepted_datatypes(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(TypeError):
            self.check_function(
                spark,
                check_fn,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )


class TestNotEqualToCheck(BaseClass):
    """This class is used to test the not equal to check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }

    sample_timestamp_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 3, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 10, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 3, 10, 0),
    }

    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": "a",
    }

    sample_bolean_data = {
        "test_pass_data": [("foo", True), ("bar", True)],
        "test_fail_data": [("foo", False), ("bar", True)],
        "test_expression": False,
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        return {
            "test_not_equal_to_check": [
                {"datatype": LongType(), "data": self.sample_numeric_data},
                {"datatype": IntegerType(), "data": self.sample_numeric_data},
                {"datatype": ByteType(), "data": self.sample_numeric_data},
                {"datatype": ShortType(), "data": self.sample_numeric_data},
                {"datatype": StringType(), "data": self.sample_string_data},
                {
                    "datatype": DoubleType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "double"
                    ),
                },
                {
                    "datatype": TimestampType(),
                    "data": self.sample_timestamp_data,
                },
                {
                    "datatype": DateType(),
                    "data": self.convert_timestamp_to_date(
                        self.sample_timestamp_data
                    ),
                },
                {
                    "datatype": DecimalType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "decimal"
                    ),
                },
                {
                    "datatype": FloatType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "float"
                    ),
                },
                {"datatype": BooleanType(), "data": self.sample_bolean_data},
            ],
            "test_failed_unaccepted_datatypes": [
                {
                    "datatype": ArrayType(StringType()),
                    "data": self.sample_array_data,
                },
                {
                    "datatype": MapType(StringType(), StringType()),
                    "data": self.sample_map_data,
                },
            ],
        }

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize("check_fn", [pa.Check.not_equal_to, pa.Check.ne])
    def test_not_equal_to_check(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if all the values are equal to defined value"""
        spark = request.getfixturevalue(spark_session)
        self.check_function(
            spark,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize("check_fn", [pa.Check.not_equal_to, pa.Check.ne])
    def test_failed_unaccepted_datatypes(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(TypeError):
            self.check_function(
                spark,
                check_fn,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )


class TestGreaterThanCheck(BaseClass):
    """This class is used to test the greater than check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 30,
    }

    sample_timestamp_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 1, 10, 0),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 3)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 1)),
        ],
        "test_expression": datetime.date(2020, 10, 1),
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        return {
            "test_greater_than_check": [
                {"datatype": LongType(), "data": self.sample_numeric_data},
                {"datatype": IntegerType(), "data": self.sample_numeric_data},
                {"datatype": ByteType(), "data": self.sample_numeric_data},
                {"datatype": ShortType(), "data": self.sample_numeric_data},
                {
                    "datatype": DoubleType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "double"
                    ),
                },
                {
                    "datatype": TimestampType(),
                    "data": self.sample_timestamp_data,
                },
                {"datatype": DateType(), "data": self.sample_date_data},
                {
                    "datatype": DecimalType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "decimal"
                    ),
                },
                {
                    "datatype": FloatType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "float"
                    ),
                },
            ],
            "test_failed_unaccepted_datatypes": [
                {"datatype": StringType(), "data": self.sample_string_data},
                {"datatype": BooleanType(), "data": self.sample_bolean_data},
                {
                    "datatype": ArrayType(StringType()),
                    "data": self.sample_array_data,
                },
                {
                    "datatype": MapType(StringType(), StringType()),
                    "data": self.sample_map_data,
                },
            ],
        }

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize("check_fn", [pa.Check.greater_than, pa.Check.gt])
    def test_greater_than_check(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if all the values are equal to defined value"""
        spark = request.getfixturevalue(spark_session)
        self.check_function(
            spark,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize("check_fn", [pa.Check.greater_than, pa.Check.gt])
    def test_failed_unaccepted_datatypes(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(TypeError):
            self.check_function(
                spark,
                check_fn,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )


class TestGreaterThanEqualToCheck(BaseClass):
    """This class is used to test the greater than equal to check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": 31,
    }

    sample_timestamp_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 1, 11, 0),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 3)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 1)),
        ],
        "test_expression": datetime.date(2020, 10, 2),
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        return {
            "test_greater_than_or_equal_to_check": [
                {"datatype": LongType(), "data": self.sample_numeric_data},
                {"datatype": IntegerType(), "data": self.sample_numeric_data},
                {"datatype": ByteType(), "data": self.sample_numeric_data},
                {"datatype": ShortType(), "data": self.sample_numeric_data},
                {
                    "datatype": DoubleType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "double"
                    ),
                },
                {
                    "datatype": TimestampType(),
                    "data": self.sample_timestamp_data,
                },
                {"datatype": DateType(), "data": self.sample_date_data},
                {
                    "datatype": DecimalType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "decimal"
                    ),
                },
                {
                    "datatype": FloatType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "float"
                    ),
                },
            ],
            "test_failed_unaccepted_datatypes": [
                {"datatype": StringType(), "data": self.sample_string_data},
                {"datatype": BooleanType(), "data": self.sample_bolean_data},
                {
                    "datatype": ArrayType(StringType()),
                    "data": self.sample_array_data,
                },
                {
                    "datatype": MapType(StringType(), StringType()),
                    "data": self.sample_map_data,
                },
            ],
        }

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize(
        "check_fn", [pa.Check.greater_than_or_equal_to, pa.Check.ge]
    )
    def test_greater_than_or_equal_to_check(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if all the values are equal to defined value"""
        spark = request.getfixturevalue(spark_session)
        self.check_function(
            spark,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize(
        "check_fn", [pa.Check.greater_than_or_equal_to, pa.Check.ge]
    )
    def test_failed_unaccepted_datatypes(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(TypeError):
            self.check_function(
                spark,
                check_fn,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )


class TestLessThanCheck(BaseClass):
    """This class is used to test the less than check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 33), ("bar", 31)],
        "test_expression": 33,
    }

    sample_timestamp_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 12, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 2, 12, 0),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 3)),
        ],
        "test_expression": datetime.date(2020, 10, 3),
    }

    sample_test_none_expression_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 33), ("bar", 31)],
        "test_expression": None,
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        return {
            "test_less_than_check": [
                {"datatype": LongType(), "data": self.sample_numeric_data},
                {"datatype": IntegerType(), "data": self.sample_numeric_data},
                {"datatype": ByteType(), "data": self.sample_numeric_data},
                {"datatype": ShortType(), "data": self.sample_numeric_data},
                {
                    "datatype": DoubleType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "double"
                    ),
                },
                {
                    "datatype": TimestampType(),
                    "data": self.sample_timestamp_data,
                },
                {"datatype": DateType(), "data": self.sample_date_data},
                {
                    "datatype": DecimalType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "decimal"
                    ),
                },
                {
                    "datatype": FloatType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "float"
                    ),
                },
            ],
            "test_failed_unaccepted_datatypes": [
                {"datatype": StringType(), "data": self.sample_string_data},
                {"datatype": BooleanType(), "data": self.sample_bolean_data},
                {
                    "datatype": ArrayType(StringType()),
                    "data": self.sample_array_data,
                },
                {
                    "datatype": MapType(StringType(), StringType()),
                    "data": self.sample_map_data,
                },
            ],
            "test_failed_none_expression": [
                {
                    "datatype": IntegerType(),
                    "data": self.sample_test_none_expression_data,
                },
            ],
        }

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize("check_fn", [pa.Check.less_than, pa.Check.lt])
    def test_less_than_check(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if all the values are equal to defined value"""
        spark = request.getfixturevalue(spark_session)
        self.check_function(
            spark,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize("check_fn", [pa.Check.less_than, pa.Check.lt])
    def test_failed_unaccepted_datatypes(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(TypeError):
            self.check_function(
                spark,
                check_fn,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize("check_fn", [pa.Check.less_than, pa.Check.lt])
    def test_failed_none_expression(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(ValueError):
            self.check_function(
                spark,
                check_fn,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )


class TestLessThanOrEqualToCheck(BaseClass):
    """This class is used to test the less than equal to check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 33)],
        "test_fail_data": [("foo", 34), ("bar", 31)],
        "test_expression": 33,
    }

    sample_timestamp_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 12, 0)),
        ],
        "test_expression": datetime.datetime(2020, 10, 2, 11, 0),
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 1)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 3)),
        ],
        "test_expression": datetime.date(2020, 10, 2),
    }

    sample_test_none_expression_data = {
        "test_pass_data": [("foo", 31), ("bar", 33)],
        "test_fail_data": [("foo", 34), ("bar", 31)],
        "test_expression": None,
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        return {
            "test_less_than_or_equal_to_check": [
                {"datatype": LongType(), "data": self.sample_numeric_data},
                {"datatype": IntegerType(), "data": self.sample_numeric_data},
                {"datatype": ByteType(), "data": self.sample_numeric_data},
                {"datatype": ShortType(), "data": self.sample_numeric_data},
                {
                    "datatype": DoubleType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "double"
                    ),
                },
                {
                    "datatype": TimestampType(),
                    "data": self.sample_timestamp_data,
                },
                {"datatype": DateType(), "data": self.sample_date_data},
                {
                    "datatype": DecimalType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "decimal"
                    ),
                },
                {
                    "datatype": FloatType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "float"
                    ),
                },
            ],
            "test_failed_unaccepted_datatypes": [
                {"datatype": StringType(), "data": self.sample_string_data},
                {"datatype": BooleanType(), "data": self.sample_bolean_data},
                {
                    "datatype": ArrayType(StringType()),
                    "data": self.sample_array_data,
                },
                {
                    "datatype": MapType(StringType(), StringType()),
                    "data": self.sample_map_data,
                },
            ],
            "test_failed_none_expression": [
                {
                    "datatype": IntegerType(),
                    "data": self.sample_test_none_expression_data,
                },
            ],
        }

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize(
        "check_fn", [pa.Check.less_than_or_equal_to, pa.Check.le]
    )
    def test_less_than_or_equal_to_check(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if all the values are equal to defined value"""
        spark = request.getfixturevalue(spark_session)
        self.check_function(
            spark,
            check_fn,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize(
        "check_fn", [pa.Check.less_than_or_equal_to, pa.Check.le]
    )
    def test_failed_unaccepted_datatypes(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(TypeError):
            self.check_function(
                spark,
                check_fn,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )

    @validate_scope(scope=ValidationScope.DATA)
    @pytest.mark.parametrize(
        "check_fn", [pa.Check.less_than_or_equal_to, pa.Check.le]
    )
    def test_failed_none_expression(
        self, spark_session, check_fn, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(ValueError):
            self.check_function(
                spark,
                check_fn,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )


class TestIsInCheck(BaseClass):
    """This class is used to test the isin check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": [31, 32],
    }

    sample_timestamp_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 3, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 10, 0)),
        ],
        "test_expression": [
            datetime.datetime(2020, 10, 1, 10, 0),
            datetime.datetime(2020, 10, 2, 10, 0),
        ],
    }
    sample_date_data = {
        "test_pass_data": [
            ("foo", datetime.date(2020, 10, 1)),
            ("bar", datetime.date(2020, 10, 2)),
        ],
        "test_fail_data": [
            ("foo", datetime.date(2020, 10, 2)),
            ("bar", datetime.date(2020, 10, 3)),
        ],
        "test_expression": [
            datetime.date(2020, 10, 1),
            datetime.date(2020, 10, 2),
        ],
    }

    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": ["b", "c"],
    }
    sample_bolean_data = {
        "test_pass_data": [("foo", [True]), ("bar", [True])],
        "test_expression": [False],
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        return {
            "test_isin_check": [
                {"datatype": LongType(), "data": self.sample_numeric_data},
                {"datatype": IntegerType(), "data": self.sample_numeric_data},
                {"datatype": ByteType(), "data": self.sample_numeric_data},
                {"datatype": ShortType(), "data": self.sample_numeric_data},
                {
                    "datatype": DoubleType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "double"
                    ),
                },
                {
                    "datatype": TimestampType(),
                    "data": self.sample_timestamp_data,
                },
                {"datatype": DateType(), "data": self.sample_date_data},
                {
                    "datatype": DecimalType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "decimal"
                    ),
                },
                {
                    "datatype": FloatType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "float"
                    ),
                },
                {"datatype": StringType(), "data": self.sample_string_data},
            ],
            "test_failed_unaccepted_datatypes": [
                {"datatype": BooleanType(), "data": self.sample_bolean_data},
                {
                    "datatype": ArrayType(StringType()),
                    "data": self.sample_array_data,
                },
                {
                    "datatype": MapType(StringType(), StringType()),
                    "data": self.sample_map_data,
                },
            ],
        }

    @validate_scope(scope=ValidationScope.DATA)
    def test_isin_check(self, spark_session, datatype, data, request) -> None:
        """Test the Check to see if all the values are is in the defined value"""
        spark = request.getfixturevalue(spark_session)
        self.check_function(
            spark,
            pa.Check.isin,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )

    @validate_scope(scope=ValidationScope.DATA)
    def test_failed_unaccepted_datatypes(
        self, spark_session, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(TypeError):
            self.check_function(
                spark,
                pa.Check.isin,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )


class TestNotInCheck(BaseClass):
    """This class is used to test the notin check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 32)],
        "test_fail_data": [("foo", 30), ("bar", 31)],
        "test_expression": [30, 33],
    }

    sample_timestamp_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 10, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 3, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 10, 0)),
        ],
        "test_expression": [
            datetime.datetime(2020, 10, 3, 10, 0),
            datetime.datetime(2020, 10, 4, 10, 0),
        ],
    }

    sample_string_data = {
        "test_pass_data": [("foo", "b"), ("bar", "c")],
        "test_fail_data": [("foo", "a"), ("bar", "b")],
        "test_expression": ["a", "d"],
    }

    sample_bolean_data = {
        "test_pass_data": [("foo", [True]), ("bar", [True])],
        "test_expression": [False],
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        return {
            "test_notin_check": [
                {"datatype": LongType(), "data": self.sample_numeric_data},
                {"datatype": IntegerType(), "data": self.sample_numeric_data},
                {"datatype": ByteType(), "data": self.sample_numeric_data},
                {"datatype": ShortType(), "data": self.sample_numeric_data},
                {
                    "datatype": DoubleType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "double"
                    ),
                },
                {
                    "datatype": TimestampType(),
                    "data": self.sample_timestamp_data,
                },
                {
                    "datatype": DateType(),
                    "data": self.convert_timestamp_to_date(
                        self.sample_timestamp_data
                    ),
                },
                {
                    "datatype": DecimalType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "decimal"
                    ),
                },
                {
                    "datatype": FloatType(),
                    "data": self.convert_numeric_data(
                        self.sample_numeric_data, "float"
                    ),
                },
                {"datatype": StringType(), "data": self.sample_string_data},
            ],
            "test_failed_unaccepted_datatypes": [
                {"datatype": BooleanType(), "data": self.sample_bolean_data},
                {
                    "datatype": ArrayType(StringType()),
                    "data": self.sample_array_data,
                },
                {
                    "datatype": MapType(StringType(), StringType()),
                    "data": self.sample_map_data,
                },
            ],
        }

    @validate_scope(scope=ValidationScope.DATA)
    def test_notin_check(self, spark_session, datatype, data, request) -> None:
        """Test the Check to see if all the values are equal to defined value"""
        spark = request.getfixturevalue(spark_session)
        self.check_function(
            spark,
            pa.Check.notin,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            data["test_expression"],
        )

    @validate_scope(scope=ValidationScope.DATA)
    def test_failed_unaccepted_datatypes(
        self, spark_session, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(TypeError):
            self.check_function(
                spark,
                pa.Check.notin,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )


class TestStringType(BaseClass):
    """This class is used to test the string types checks"""

    @validate_scope(scope=ValidationScope.DATA)
    def test_str_startswith_check(self, spark_session, request) -> None:
        """Test the Check to see if any value is not in the specified value"""
        spark = request.getfixturevalue(spark_session)
        check_func = pa.Check.str_startswith
        check_value = "B"

        pass_data = [("Bal", "Bread"), ("Bal", "Butter")]
        fail_data = [("Bal", "Test"), ("Bal", "Butter")]
        BaseClass.check_function(
            spark, check_func, pass_data, fail_data, StringType(), check_value
        )

    @validate_scope(scope=ValidationScope.DATA)
    def test_str_endswith_check(self, spark_session, request) -> None:
        """Test the Check to see if any value is not in the specified value"""
        spark = request.getfixturevalue(spark_session)
        check_func = pa.Check.str_endswith
        check_value = "d"

        pass_data = [("Bal", "Bread"), ("Bal", "Bad")]
        fail_data = [("Bal", "Test"), ("Bal", "Bad")]
        BaseClass.check_function(
            spark, check_func, pass_data, fail_data, StringType(), check_value
        )

    @validate_scope(scope=ValidationScope.DATA)
    def test_str_contains_check(self, spark_session, request) -> None:
        """Test the Check to see if any value is not in the specified value"""
        spark = request.getfixturevalue(spark_session)
        check_func = pa.Check.str_contains
        check_value = "Ba"

        pass_data = [("Bal", "Bat!"), ("Bal", "Bat78")]
        fail_data = [("Bal", "Cs"), ("Bal", "Jam!")]
        BaseClass.check_function(
            spark, check_func, pass_data, fail_data, StringType(), check_value
        )


class TestInRangeCheck(BaseClass):
    """This class is used to test the value in range check"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 33)],
        "test_fail_data": [("foo", 35), ("bar", 31)],
    }

    sample_timestamp_data = {
        "test_pass_data": [
            ("foo", datetime.datetime(2020, 10, 1, 11, 0)),
            ("bar", datetime.datetime(2020, 10, 2, 11, 0)),
        ],
        "test_fail_data": [
            ("foo", datetime.datetime(2020, 10, 1, 10, 0)),
            ("bar", datetime.datetime(2020, 10, 5, 12, 0)),
        ],
    }

    sample_bolean_data = {
        "test_pass_data": [("foo", [True]), ("bar", [True])],
        "test_expression": [False],
    }

    def pytest_generate_tests(self, metafunc):
        """This function passes the parameter for each function based on parameter form get_data_param function"""
        # called once per each test function
        funcarglist = self.get_data_param()[metafunc.function.__name__]
        argnames = sorted(funcarglist[0])
        metafunc.parametrize(
            argnames,
            [
                [funcargs[name] for name in argnames]
                for funcargs in funcarglist
            ],
        )

    def create_min_max(self, data_dictionary):
        """This function create the min and max value from the data dictionary to be used for in range test"""
        value_dict = [value[1] for value in data_dictionary["test_pass_data"]]
        min_val = min(value_dict)
        max_val = max(value_dict)
        if isinstance(min_val, datetime.datetime):
            add_value = datetime.timedelta(1)
        elif isinstance(min_val, datetime.date):
            add_value = datetime.timedelta(1)
        else:
            add_value = 1
        return min_val, max_val, add_value

    def get_data_param(self):
        """Generate the params which will be used to test this function. All the accpetable
        data types would be tested"""
        param_vals = [
            {"datatype": LongType(), "data": self.sample_numeric_data},
            {"datatype": IntegerType(), "data": self.sample_numeric_data},
            {"datatype": ByteType(), "data": self.sample_numeric_data},
            {"datatype": ShortType(), "data": self.sample_numeric_data},
            {
                "datatype": DoubleType(),
                "data": self.convert_numeric_data(
                    self.sample_numeric_data, "double"
                ),
            },
            {"datatype": TimestampType(), "data": self.sample_timestamp_data},
            {
                "datatype": DateType(),
                "data": self.convert_timestamp_to_date(
                    self.sample_timestamp_data
                ),
            },
            {
                "datatype": DecimalType(),
                "data": self.convert_numeric_data(
                    self.sample_numeric_data, "decimal"
                ),
            },
            {
                "datatype": FloatType(),
                "data": self.convert_numeric_data(
                    self.sample_numeric_data, "float"
                ),
            },
        ]

        return {
            "test_inrange_exclude_min_max_check": param_vals,
            "test_inrange_exclude_min_only_check": param_vals,
            "test_inrange_exclude_max_only_check": param_vals,
            "test_inrange_include_min_max_check": param_vals,
            "test_failed_unaccepted_datatypes": [
                {"datatype": StringType(), "data": self.sample_string_data},
                {"datatype": BooleanType(), "data": self.sample_bolean_data},
                {
                    "datatype": ArrayType(StringType()),
                    "data": self.sample_array_data,
                },
                {
                    "datatype": MapType(StringType(), StringType()),
                    "data": self.sample_map_data,
                },
            ],
        }

    @validate_scope(scope=ValidationScope.DATA)
    def test_inrange_exclude_min_max_check(
        self, spark_session, datatype, data, request
    ) -> None:
        """Test the Check to see if any value is not in the specified value"""
        spark = request.getfixturevalue(spark_session)
        min_val, max_val, add_value = self.create_min_max(data)
        self.check_function(
            spark,
            pa.Check.in_range,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            (min_val - add_value, max_val + add_value, False, False),
        )

    @validate_scope(scope=ValidationScope.DATA)
    def test_inrange_exclude_min_only_check(
        self, spark_session, datatype, data, request
    ) -> None:
        """Test the Check to see if any value is not in the specified value"""
        spark = request.getfixturevalue(spark_session)
        min_val, max_val, add_value = self.create_min_max(data)
        self.check_function(
            spark,
            pa.Check.in_range,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            (min_val, max_val + add_value, True, False),
        )

    @validate_scope(scope=ValidationScope.DATA)
    def test_inrange_exclude_max_only_check(
        self, spark_session, datatype, data, request
    ) -> None:
        """Test the Check to see if any value is not in the specified value"""
        spark = request.getfixturevalue(spark_session)
        min_val, max_val, add_value = self.create_min_max(data)
        self.check_function(
            spark,
            pa.Check.in_range,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            (min_val - add_value, max_val, False, True),
        )

    @validate_scope(scope=ValidationScope.DATA)
    def test_inrange_include_min_max_check(
        self, spark_session, datatype, data, request
    ) -> None:
        """Test the Check to see if any value is not in the specified value"""
        spark = request.getfixturevalue(spark_session)
        (
            min_val,
            max_val,
            add_value,  # pylint:disable=unused-variable
        ) = self.create_min_max(data)
        self.check_function(
            spark,
            pa.Check.in_range,
            data["test_pass_data"],
            data["test_fail_data"],
            datatype,
            (min_val, max_val, True, True),
        )

    @validate_scope(scope=ValidationScope.DATA)
    def test_failed_unaccepted_datatypes(
        self, spark_session, datatype, data, request
    ) -> None:
        """Test the Check to see if error is raised for datatypes which are not accepted for this function"""
        spark = request.getfixturevalue(spark_session)
        with pytest.raises(TypeError):
            self.check_function(
                spark,
                pa.Check.in_range,
                data["test_pass_data"],
                None,
                datatype,
                data["test_expression"],
            )


class TestCustomCheck(BaseClass):
    """This test validates the functionality of custom checks"""

    sample_numeric_data = {
        "test_pass_data": [("foo", 31), ("bar", 33)],
        "test_fail_data": [("foo", 34), ("bar", 31)],
        "test_expression": 33,
    }

    @staticmethod
    def _check_extension(
        spark, schema, pass_case_data, fail_case_data, data_types
    ):
        """
        This function does performs the actual validation
        """
        spark_schema = StructType(
            [
                StructField("product", StringType(), False),
                StructField("code", data_types, False),
            ],
        )
        df = spark.createDataFrame(data=pass_case_data, schema=spark_schema)
        df_out = schema.validate(df)
        if df_out.pandera.errors:
            print(df_out.pandera.errors)
            raise PysparkSchemaError

        with pytest.raises(PysparkSchemaError):
            df_fail = spark.createDataFrame(
                data=fail_case_data, schema=spark_schema
            )
            df_out = schema.validate(df_fail)
            if df_out.pandera.errors:
                raise PysparkSchemaError

    def test_extension(
        self, spark_session, extra_registered_checks, request
    ):  # pylint: disable=unused-argument
        """Test custom extension with DataFrameSchema way of defining schema"""
        spark = request.getfixturevalue(spark_session)
        schema = DataFrameSchema(
            {
                "product": Column(StringType()),
                "code": Column(
                    IntegerType(),
                    pa.Check.new_pyspark_check(
                        max_value=self.sample_numeric_data["test_expression"]
                    ),
                ),
            }
        )
        self._check_extension(
            spark,
            schema,
            self.sample_numeric_data["test_pass_data"],
            self.sample_numeric_data["test_fail_data"],
            IntegerType(),
        )

    def test_extension_pydantic(
        self, spark_session, extra_registered_checks, request
    ):  # pylint: disable=unused-argument
        """Test custom extension with DataFrameModel way of defining schema"""
        spark = request.getfixturevalue(spark_session)

        class Schema(DataFrameModel):
            """Test Schema"""

            product: StringType
            code: IntegerType = Field(
                new_pyspark_check={
                    "max_value": self.sample_numeric_data["test_expression"]
                }
            )

        self._check_extension(
            spark,
            Schema,
            self.sample_numeric_data["test_pass_data"],
            self.sample_numeric_data["test_fail_data"],
            IntegerType(),
        )
