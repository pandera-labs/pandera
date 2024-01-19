from pandera.api.checks import Check
from pandera.api.pandas.components import Column
from pandera.api.pandas.container import DataFrameSchema
from pandera.backends.pandas.error_formatters import summarize_failure_cases
import pytest
from pandera.errors import SchemaError, SchemaErrorReason
import pandas as pd

check = Check.isin(["coke", "7up", "mountain_dew"])
mock_schema = DataFrameSchema({"flavour": Column(str, checks=[check])})

dataframe = pd.DataFrame({"flavour": ["pepsi", "coke", "fanta"]})


@pytest.mark.parametrize(
    "schema_name, schema_errors, failure_cases",
    [
        (
            "MySchema",
            [
                SchemaError(
                    schema=mock_schema,
                    data=dataframe,
                    message=None,
                    failure_cases=pd.DataFrame(
                        {"index": [0, 2], "failure_case": ["pepsi", "fanta"]}
                    ),
                    check=check,
                    check_index=0,
                    check_output=pd.Series(
                        name="Flavour", data=[False, True, False]
                    ),
                    reason_code=SchemaErrorReason.DATAFRAME_CHECK,
                )
            ],
            pd.DataFrame(
                {
                    "index": [0, 1],
                    "schema_context": ["Column", "Column"],
                    "column": ["flavour", "flavour"],
                    "check": [
                        "isin(['coke', '7up', 'mountain_dew'])",
                        "isin(['coke', '7up', 'mountain_dew'])",
                    ],
                    "check_number": [0, 0],
                    "failure_case": ["pepsi", "fanta"],
                }
            ),
        )
    ],
)
def test_summarize_failure_cases(schema_name, schema_errors, failure_cases):
    summary = summarize_failure_cases(
        schema_name=schema_name,
        schema_errors=schema_errors,
        failure_cases=failure_cases,
    )

    assert summary == "foo"


if __name__ == "__main__":
    schema_name = None
    schema_errors = [
        SchemaError(
            schema=mock_schema,
            data=dataframe,
            message=None,
            failure_cases=pd.DataFrame(
                {"index": [0, 2], "failure_case": ["pepsi", "fanta"]}
            ),
            check=check,
            check_index=0,
            check_output=pd.Series(name="Flavour", data=[False, True, False]),
            reason_code=SchemaErrorReason.DATAFRAME_CHECK,
        )
    ]
    failure_cases = pd.DataFrame(
        {
            "index": [0, 1],
            "schema_context": ["Column", "Column"],
            "column": ["flavour", "flavour"],
            "check": [
                "isin(['coke', '7up', 'mountain_dew'])",
                "isin(['coke', '7up', 'mountain_dew'])",
            ],
            "check_number": [0, 0],
            "failure_case": ["pepsi", "fanta"],
        }
    )
    summary = summarize_failure_cases(
        schema_name=schema_name,
        schema_errors=schema_errors,
        failure_cases=failure_cases,
    )
