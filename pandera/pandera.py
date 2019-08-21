"""Validate Pandas Data Structures."""

import inspect
import warnings
import pandas as pd
import wrapt
from typing import Union, Optional, List, Dict

from collections import OrderedDict
from enum import Enum
from functools import partial
from scipy import stats


class SchemaInitError(Exception):
    pass


class SchemaDefinitionError(Exception):
    pass


class SchemaError(Exception):
    pass


class PandasDtype(Enum):
    Bool = "bool"
    DateTime = "datetime64[ns]"
    Category = "category"
    Float = "float64"
    Int = "int64"
    Object = "object"
    String = "object"
    Timedelta = "timedelta64[ns]"


Bool = PandasDtype.Bool
DateTime = PandasDtype.DateTime
Category = PandasDtype.Category
Float = PandasDtype.Float
Int = PandasDtype.Int
Object = PandasDtype.Object
String = PandasDtype.String
Timedelta = PandasDtype.Timedelta

N_FAILURE_CASES = 10


class Check(object):

    def __init__(
            self,
            fn: callable,
            groups: Union[str, List[str], None] = None,
            groupby: Union[str, List[str], callable, None] = None,
            element_wise: Union[bool, List[bool]] = False,
            error: Optional[str] = None,
            n_failure_cases: Optional[int] = N_FAILURE_CASES):
        """Check object applies function element-wise or series-wise

        :param fn: A function to check series schema. If element_wise
            is True, then callable signature should be: x -> bool where x is a
            scalar element in the column. Otherwise, signature is expected
            to be: pd.Series -> bool|pd.Series[bool].
        :param groups: The dict input to the `fn` callable will be constrained
            to the groups specified by `groups`.
        :param groupby: Only applies to Column Checks. If a string or list of
            strings is provided, then these columns are used to group the
            Column Series by `groupby`. If a callable is passed, the expected
            signature is DataFrame -> DataFrameGroupby. The function has access
            to the entire dataframe, but the Column.name is selected from this
            DataFrameGroupby object so that a SeriesGroupBy object is passed
            into `fn`.

            Specifying this argument changes the `fn` signature to:

            dict[str|tuple[str], Series] -> bool|pd.Series[bool]

            Where specific groups can be obtained from the input dict.
        :param element_wise: Whether or not to apply validator in an
            element-wise fashion. If bool, assumes that all checks should be
            applied to the column element-wise. If list, should be the same
            number of elements as checks.
        :param error: custom error message if series fails validation check.
        :param n_failure_cases: report the top n failure cases. If None, then
            report all failure cases.
        """
        if element_wise and groupby is not None:
            raise SchemaInitError("Cannot use groupby when element_wise=True.")
        self.fn = fn
        self.element_wise = element_wise
        self.error = error
        self.n_failure_cases = n_failure_cases

        if groupby is None and groups is not None:
            raise ValueError(
                "`groupby` argument needs to be provided when `groups` "
                "argument is defined")

        if isinstance(groupby, str):
            groupby = [groupby]
        self.groupby = groupby
        if isinstance(groups, str):
            groups = [groups]
        self.groups = groups

    @property
    def error_message(self) -> str:

        if self.error:
            return "%s: %s" % (self.fn.__name__, self.error)
        return "%s" % self.fn.__name__

    def vectorized_error_message(
            self,
            parent_schema,
            check_index: int,
            failure_cases: Union[pd.DataFrame, pd.Series]) -> str:
        """Constructs an error message when an element-wise validator fails.

        :param DataFrameSchema parent_schema: The schema object that is being
            checked and that was inherited from the parent class.
        :param check_index: The validator that failed.
        :param failure_cases: The failure cases encountered by the element-wise
            validator.

        """
        return (
                "%s failed element-wise validator %d:\n"
                "%s\nfailure cases:\n%s" %
                (parent_schema,
                 check_index,
                 self.error_message,
                 self._format_failure_cases(failure_cases)))

    def generic_error_message(self,
                              parent_schema,
                              check_index: int) -> str:
        """Constructs an error message when a check validator fails.

        :param DataFrameSchema parent_schema: The schema object that is being
            checked and that was inherited from the parent class.
        :param check_index: The validator that failed.

        """
        return "%s failed series validator %d: %s" % \
               (parent_schema, check_index, self.error_message)

    def _format_failure_cases(self,
                              failure_cases: Union[pd.DataFrame, pd.Series]
                              ) -> pd.DataFrame:
        """Constructs readable error messages for vectorized_error_message.

        :param failure_cases: The failure cases encountered by the
            element-wise validator.

        """
        if isinstance(failure_cases.index, pd.MultiIndex):
            failure_cases = (
                failure_cases
                .rename("failure_case")
                .reset_index()
                .assign(
                    index=lambda df: (
                        df.apply(tuple, axis=1).astype(str)
                    )
                )
            )
        elif isinstance(failure_cases, pd.DataFrame):
            failure_cases = (
                failure_cases
                .pipe(lambda df: pd.Series(
                    df.itertuples()).map(lambda x: x.__repr__()))
                .rename("failure_case")
                .reset_index()
            )
        else:
            failure_cases = (
                failure_cases
                .rename("failure_case")
                .reset_index()
            )

        failure_cases = (
            failure_cases
            .groupby("failure_case").index.agg([list, len])
            .rename(columns={"list": "index", "len": "count"})
            .sort_values("count", ascending=False)
        )

        self.failure_cases = failure_cases
        return failure_cases.head(self.n_failure_cases)

    def _format_input(self,
                      groupby_obj,
                      groups
                      ) -> Union[Dict[str, Union[pd.Series, pd.DataFrame]]]:
        """

        :param groupby_obj:
        :param groups:
        :return: check_obj
        """
        if groups is None:
            return {group_key: group for group_key, group in groupby_obj}
        group_keys = set(group_key for group_key, _ in groupby_obj)
        invalid_groups = [g for g in groups if g not in group_keys]
        if invalid_groups:
            raise KeyError(
                "groups %s provided in `groups` argument not a valid group "
                "key. Valid group keys: %s" % (invalid_groups, group_keys))
        return {
            group_key: group for group_key, group in groupby_obj
            if group_key in groups
        }

    def prepare_series_input(self,
                             series: pd.Series,
                             dataframe: pd.DataFrame) -> Dict[str, pd.Series]:
        """Prepare input for Column check.

        :param series: One-dimensional ndarray with axis labels
            (including time series).
        :param dataframe: Two-dimensional size-mutable,
            potentially heterogeneous tabular data structure with labeled axes
            (rows and columns)
        :return: a check_obj dictionary of pd.Series to be used by `_check_fn`
            and `_vectorized_series_check`

        """
        if dataframe is None or self.groupby is None:
            return series
        elif isinstance(self.groupby, list):
            groupby_obj = (
                pd.concat([series, dataframe[self.groupby]], axis=1)
                .groupby(self.groupby)[series.name]
            )
        elif callable(self.groupby):
            groupby_obj = self.groupby(
                pd.concat([series, dataframe], axis=1))[series.name]
        else:
            raise TypeError("Type %s not recognized for `groupby` argument.")

        return self._format_input(groupby_obj, self.groups)

    def prepare_dataframe_input(self,
                                dataframe: pd.DataFrame):
        """Prepare input for DataFrameSchema check."""
        if self.groupby is None:
            return dataframe
        else:
            groupby_obj = dataframe.groupby(self.groupby)
        return self._format_input(groupby_obj, self.groups)

    def _vectorized_check(self,
                          parent_schema,
                          check_index: int,
                          check_obj: Dict[str, pd.Series]):
        """Perform a vectorized check on a series.

        :param DataFrameSchema parent_schema: The schema object that is
            being checked and that was inherited from the parent class.
        :param check_index: The validator to check the series for
        :param check_obj: a dictionary of pd.Series to be used by
            `_check_fn` and `_vectorized_series_check`

        """
        val_result = self.fn(check_obj)
        if isinstance(val_result, pd.Series):
            if not val_result.dtype == PandasDtype.Bool.value:
                raise TypeError(
                    "validator %d: %s must return bool or Series of type "
                    "bool, found %s" %
                    (check_index, self.fn.__name__, val_result.dtype))
            if val_result.all():
                return True
            elif isinstance(check_obj, dict) or \
                    check_obj.shape[0] != val_result.shape[0] or \
                    (check_obj.index != val_result.index).all():
                raise SchemaError(
                    self.generic_error_message(parent_schema, check_index))
            else:
                raise SchemaError(self.vectorized_error_message(
                    parent_schema, check_index, check_obj[~val_result]))
        else:
            if val_result:
                return True
            raise SchemaError(
                self.generic_error_message(parent_schema, check_index))

    def __call__(self,
                 parent_schema,
                 check_index: int,
                 check_obj: Dict[str, pd.Series]):
        _vcheck = partial(
            self._vectorized_check, parent_schema, check_index)
        if self.element_wise:
            val_result = check_obj.apply(self.fn, axis=1) if \
                isinstance(check_obj, pd.DataFrame) else check_obj.map(self.fn)
            if val_result.all():
                return True
            raise SchemaError(self.vectorized_error_message(
                parent_schema, check_index, check_obj[~val_result]))
        elif isinstance(check_obj, (pd.Series, dict, pd.DataFrame)):
            return _vcheck(check_obj)
        else:
            raise ValueError(
                "check_obj type %s not supported. Must be a "
                "Series, a dictionary of Series, or DataFrame" % check_obj)


DEFAULT_ALPHA = 0.01


class Hypothesis(Check):
    """Extends Check to perform a hypothesis test on a Column or DataFrame."""

    RELATIONSHIPS = {
        "greater_than": (lambda stat, pvalue, alpha=DEFAULT_ALPHA:
                         stat > 0 and pvalue / 2 < alpha),
        "less_than": (lambda stat, pvalue, alpha=DEFAULT_ALPHA:
                      stat < 0 and pvalue / 2 < alpha),
        "not_equal": (lambda stat, pvalue, alpha=DEFAULT_ALPHA:
                      pvalue < alpha),
        "equal": (lambda stat, pvalue, alpha=DEFAULT_ALPHA: pvalue >= alpha),
    }

    def __init__(self,
                 test: callable,
                 samples: Union[str, List[str], None],
                 groupby: Union[str, List[str], callable, None] = None,
                 relationship: Union[str, callable] = "equal",
                 test_kwargs: Dict = None,
                 relationship_kwargs: Dict = None,
                 error: Optional[str] = None):
        """Initialises hypothesis to perform a hypothesis test on a Column.

            Can function on a single column or be grouped by another column.

        :param test: A function to check a series schema.
        :param samples: for `Column` or `SeriesSchema` hypotheses, this refers
            to the group keys in the `groupby` column(s) used to group the
            `Series` into a dict of `Series`. The `samples` column(s) are
            passed into the `test` function as positional arguments.

            For `DataFrame`-level hypotheses, `samples` refers to a column or
            multiple columns to pass into the `test` function. The `samples`
            column(s) are passed into the `test`  function as positional
            arguments.
        :param groupby: If a string or list of strings is provided, then these
            columns are used to group the Column Series by `groupby`. If a
            callable is passed, the expected signature is
            DataFrame -> DataFrameGroupby. The function has access to the
            entire dataframe, but the Column.name is selected from this
            DataFrameGroupby object so that a SeriesGroupBy object is passed
            into the `hypothesis_check` function.

            Specifying this argument changes the `fn` signature to:
            dict[str|tuple[str], Series] -> bool|pd.Series[bool]

            Where specific groups can be obtained from the input dict.
        :param relationship: Represents what relationship conditions are
            imposed on the hypothesis test. A function or lambda function can
            be supplied.

            If a string is provided, a lambda function will be returned from
            Hypothesis.relationships. Available relationships are:
            "greater_than", "less_than", "not_equal" or "equal".

            If callable, the input function signature should have the signature
            `(stat: float, pvalue: float, **kwargs)` where `stat` is the
            hypothesis test statistic, `pvalue` assesses statistical
            significance, and `**kwargs` are other arguments supplied by the
            `**relationship_kwargs` argument.

            Default is "equal" for the null hypothesis.

        :param test_kwargs: Key Word arguments to be supplied to the test.
        :param relationship_kwargs: Key Word arguments to be supplied to
            the relationship function. e.g. `alpha` could be used to specify a
            threshold in a t-test.
        :param error: error message to show
        """
        self.test = partial(test, **{} if test_kwargs is None else test_kwargs)
        self.relationship = partial(self.relationships(relationship),
                                    **relationship_kwargs)
        if isinstance(samples, str):
            samples = [samples]
        self.samples = samples
        super(Hypothesis, self).__init__(
            self.hypothesis_check,
            groupby=groupby,
            element_wise=False,
            error=error)

    @property
    def is_one_sample_test(self):
        return len(self.samples) == 1

    def prepare_series_input(self,
                             series: pd.Series,
                             dataframe: pd.DataFrame):
        """Prepare input for Hypothesis check.

        :param series: One-dimensional ndarray with axis labels
            (including time series).
        :param dataframe: Two-dimensional size-mutable,
            potentially heterogeneous tabular data structure with labeled axes
            (rows and columns)
        :return: a check_obj dictionary of pd.Series to be used by `_check_fn`
            and `_vectorized_series_check`

        """
        self.groups = self.samples
        return super(Hypothesis, self).prepare_series_input(series, dataframe)

    def prepare_dataframe_input(self,
                                dataframe: pd.DataFrame):
        """Prepare input for DataFrameSchema Hypothesis check."""
        if self.groupby is not None:
            raise SchemaDefinitionError(
                "`groupby` cannot be used for DataFrameSchema checks, must "
                "be used in Column checks.")
        if self.is_one_sample_test:
            return dataframe[self.samples[0]]
        check_obj = [(sample, dataframe[sample]) for sample in self.samples]
        return self._format_input(check_obj, self.samples)

    def relationships(self,
                      relationship: Union[str, callable]):
        """Impose a relationship on a supplied Test function.

        :param relationship: represents what relationship conditions are
            imposed on the hypothesis test. A function or lambda function can
            be supplied. If a string is provided, a lambda function will be
            returned from Hypothesis.relationships. Available relationships
            are: "greater_than", "less_than", "not_equal"

        """
        if isinstance(relationship, str):
            if relationship not in self.RELATIONSHIPS:
                raise SchemaError(
                    "The relationship %s isn't a built in method"
                    % relationship)
            else:
                relationship = self.RELATIONSHIPS[relationship]
        elif not callable(relationship):
            raise ValueError(
                "expected relationship to be str or callable, found %s" % type(
                    relationship)
            )
        return relationship

    def hypothesis_check(self,
                         check_obj: Dict[str, pd.Series]):
        """Create a function fn which is checked via the Check parent class.

        :param check_obj: a dictionary of pd.Series to be used by
            `hypothesis_check` and `_vectorized_series_check`

        """
        if self.is_one_sample_test:
            # one-sample case where no groupby argument supplied, apply to
            # entire column
            return self.relationship(*self.test(check_obj))
        else:
            return self.relationship(
                *self.test(*[check_obj.get(s) for s in self.samples]))

    @classmethod
    def two_sample_ttest(
            cls,
            sample1: str,
            sample2: str,
            groupby: Union[str, List[str], callable, None] = None,
            relationship: str = "equal",
            alpha=DEFAULT_ALPHA,
            equal_var=True,
            nan_policy="propagate"):
        """Calculate a t-test for the means of two columns.

        This reuses the scipy.stats.ttest_ind to perfom a two-sided test for
        the null hypothesis that 2 independent samples have identical average
        (expected) values. This test assumes that the populations have
        identical variances by default.

        :param sample1: The first sample group to test. For `Column` and
            `SeriesSchema` hypotheses, refers to the level in the `groupby`
            column. For `DataFrameSchema` hypotheses, refers to column in
            the `DataFrame`.
        :param sample2: The second sample group to test. For `Column` and
            `SeriesSchema` hypotheses, refers to the level in the `groupby`
            column. For `DataFrameSchema` hypotheses, refers to column in
            the `DataFrame`.
        :param groupby: If a string or list of strings is provided, then these
            columns are used to group the Column Series by `groupby`. If a
            callable is passed, the expected signature is
            DataFrame -> DataFrameGroupby. The function has access to the
            entire dataframe, but the Column.name is selected from this
            DataFrameGroupby object so that a SeriesGroupBy object is passed
            into `fn`.

            Specifying this argument changes the `fn` signature to:
            dict[str|tuple[str], Series] -> bool|pd.Series[bool]

            Where specific groups can be obtained from the input dict.
        :param relationship: Represents what relationship conditions are
            imposed on the hypothesis test. Available relationships
            are: "greater_than", "less_than", "not_equal", and "equal".
            For example, `group1 greater_than group2` specifies an alternative
            hypothesis that the mean of group1 is greater than group 2 relative
            to a null hypothesis that they are equal.
        :param alpha: (Default value = 0.01) The significance level; the
            probability of rejecting the null hypothesis when it is true. For
            example, a significance level of 0.01 indicates a 1% risk of
            concluding that a difference exists when there is no actual
            difference.
        :param equal_var: (Default value = True) If True (default), perform a
            standard independent 2 sample test that assumes equal population
            variances. If False, perform Welch's t-test, which does not
            assume equal population variance
        :param nan_policy: Defines how to handle when input returns nan, one of
            {'propagate', 'raise', 'omit'}, (Default value = 'propagate').
            For more details see:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html  # noqa E53
        """
        if relationship not in cls.RELATIONSHIPS:
            raise SchemaError(
                "relationship must be one of %s" % set(cls.RELATIONSHIPS))
        return cls(
            test=stats.ttest_ind,
            samples=[sample1, sample2],
            groupby=groupby,
            relationship=relationship,
            test_kwargs={"equal_var": equal_var, "nan_policy": nan_policy},
            relationship_kwargs={"alpha": alpha},
            error="failed two sample ttest between '%s' and '%s'" % (
                sample1, sample2),
        )

    @classmethod
    def one_sample_ttest(
            cls,
            sample: str,
            popmean: float,
            relationship: str,
            alpha: float = DEFAULT_ALPHA):
        """Calculate a t-test for the mean of one column.

        :param sample: The sample group to test. For `Column` and
            `SeriesSchema` hypotheses, refers to the `groupby` level in the
            `Column`. For `DataFrameSchema` hypotheses, refers to column in
            the `DataFrame`.
        :param popmean: population mean to compare `sample` to.
        :param relationship: Represents what relationship conditions are
            imposed on the hypothesis test. Available relationships
            are: "greater_than", "less_than", "not_equal" and "equal". For
            example, `group1 greater_than group2` specifies an alternative
            hypothesis that the mean of group1 is greater than group 2 relative
            to a null hypothesis that they are equal.
        :param alpha: (Default value = 0.01) The significance level; the
            probability of rejecting the null hypothesis when it is true. For
            example, a significance level of 0.01 indicates a 1% risk of
            concluding that a difference exists when there is no actual
            difference.
        """
        if relationship not in cls.RELATIONSHIPS:
            raise SchemaError(
                "relationship must be one of %s" % set(cls.RELATIONSHIPS))
        return cls(
            test=stats.ttest_ind,
            samples=sample,
            relationship=relationship,
            test_kwargs={"popmean": popmean},
            relationship_kwargs={"alpha": alpha},
            error="failed one sample ttest between for column '%s'" % (
                sample),
        )


class DataFrameSchema(object):
    """A light-weight pandas DataFrame validator."""

    def __init__(
            self,
            columns,
            checks: callable = None,
            index=None,
            transformer: callable = None,
            coerce: bool = False,
            strict=False
            ):
        """A light-weight pandas DataFrame validator.

        :param columns: a dict where keys are column names and values are
            Column objects specifying the datatypes and properties of a
            particular column.
        :param checks: dataframe-wide checks.
        :param index: specify the datatypes and properties of the index.
        :param transformer: a callable with signature:
            pandas.DataFrame -> pandas.DataFrame. If specified, calling
            `validate` will verify properties of the columns and return the
            transformed dataframe object.
        :param coerce: whether or not to coerce all of the columns on
            validation.
        :param strict: whether or not to accept columns in the dataframe that
            aren't in the DataFrameSchema.
        """
        if checks is None:
            checks = []
        if isinstance(checks, Check):
            checks = [checks]
        self._checks = checks
        self.index = index
        self.columns = columns
        self.transformer = transformer
        self.coerce = coerce
        self.strict = strict
        self._validate_schema()

    def __call__(
            self,
            dataframe: pd.DataFrame,
            head: int = None,
            tail: int = None,
            sample: int = None,
            random_state: Optional[int] = None
            ):
        """Delegate to `validate` method.

        :param dataframe: the dataframe to be validated.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        """
        return self.validate(dataframe)

    def _validate_schema(self):

        for column_name, column in self.columns.items():
            for check in column._checks:
                if check.groupby is None or callable(check.groupby):
                    continue
                nonexistent_groupby_columns = [
                    c for c in check.groupby if c not in self.columns]
                if nonexistent_groupby_columns:
                    raise SchemaInitError(
                        "groupby argument %s in Check for Column %s not "
                        "specified in the DataFrameSchema." %
                        (nonexistent_groupby_columns, column_name))

    @staticmethod
    def _dataframe_to_validate(dataframe: pd.DataFrame,
                               head: int,
                               tail: int,
                               sample: int,
                               random_state: Optional[int]) -> pd.DataFrame:
        dataframe_subsample = []
        if head is not None:
            dataframe_subsample.append(dataframe.head(head))
        if tail is not None:
            dataframe_subsample.append(dataframe.tail(tail))
        if sample is not None:
            dataframe_subsample.append(
                dataframe.sample(sample, random_state=random_state))
        return dataframe if len(dataframe_subsample) == 0 else \
            pd.concat(dataframe_subsample).drop_duplicates()

    def _check_dataframe(self,
                         dataframe: pd.DataFrame):
        return all(
            check(self, check_index, check.prepare_dataframe_input(dataframe))
            for check_index, check in enumerate(self._checks))

    def validate(
            self,
            dataframe: pd.DataFrame,
            head: int = None,
            tail: int = None,
            sample: int = None,
            random_state: Optional[int] = None) -> pd.DataFrame:
        """Check if all columns in a dataframe have a column in the Schema.

        :param dataframe: the dataframe to be validated.
        :param head: validate the first n rows. Rows overlapping with `tail` or
            `sample` are de-duplicated.
        :param tail: validate the last n rows. Rows overlapping with `head` or
            `sample` are de-duplicated.
        :param sample: validate a random sample of n rows. Rows overlapping
            with `head` or `tail` are de-duplicated.
        """
        if self.strict:
            for column in dataframe:
                if column not in self.columns:
                    raise SchemaError(
                        "column '%s' not in DataFrameSchema %s" %
                        (column, self.columns)
                    )

        for colname, col in self.columns.items():
            if colname not in dataframe and col.required:
                raise SchemaError(
                    "column '%s' not in dataframe\n%s" %
                    (colname, dataframe.head()))

            if col.coerce or self.coerce:
                dataframe[colname] = col.coerce_dtype(dataframe[colname])

        schema_elements = [
            col.set_name(col_name) for col_name, col in self.columns.items()
            if col.required or col_name in dataframe
        ]
        if self.index is not None:
            schema_elements += [self.index]

        dataframe_to_validate = self._dataframe_to_validate(
            dataframe, head, tail, sample, random_state)
        assert (
            all(s(dataframe_to_validate) for s in schema_elements) and
            self._check_dataframe(dataframe))
        if self.transformer is not None:
            dataframe = self.transformer(dataframe)
        return dataframe


class SeriesSchemaBase(object):
    """Base series validator object."""

    def __init__(self,
                 pandas_dtype,
                 checks: callable = None,
                 nullable: bool = False,
                 allow_duplicates: bool = True,
                 name: str = None):
        """Initialize series schema base object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: If element_wise is True, then callable signature should
            be:
            x -> x where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        :param nullable: Whether or not column can contain null values.
        :param allow_duplicates:
        """
        self._pandas_dtype = pandas_dtype
        self._nullable = nullable
        self._allow_duplicates = allow_duplicates
        if checks is None:
            checks = []
        if isinstance(checks, Check):
            checks = [checks]
        self._checks = checks
        self._name = name

        for check in self._checks:
            if check.groupby is not None and not isinstance(self, Column):
                raise SchemaInitError(
                    "Can only use `groupby` with a pandera.Column, found %s" %
                    type(self))

    def __call__(self,
                 series: pd.Series,
                 dataframe: pd.DataFrame = None):
        """Validate a series."""
        if series.name != self._name:
            raise SchemaError(
                "Expected %s to have name '%s', found '%s'" %
                (type(self), self._name, series.name))
        expected_dtype = _dtype = self._pandas_dtype if \
            isinstance(self._pandas_dtype, str) else self._pandas_dtype.value
        if self._nullable:
            series = series.dropna()
            if dataframe is not None:
                dataframe = dataframe.loc[series.index]
            if _dtype in ["int_", "int8", "int16", "int32", "int64", "uint8",
                          "uint16", "uint32", "uint64"]:
                _series = series.astype(_dtype)
                if (_series != series).any():
                    # in case where dtype is meant to be int, make sure that
                    # casting to int results in the same values.
                    raise SchemaError(
                        "after dropping null values, expected values in "
                        "series '%s' to be int, found: %s" %
                        (series.name, set(series)))
                series = _series
        else:
            nulls = series.isnull()
            if nulls.sum() > 0:
                if series.dtype != _dtype:
                    raise SchemaError(
                        "expected series '%s' to have type %s, got %s and "
                        "non-nullable series contains null values: %s" %
                        (series.name, self._pandas_dtype.value, series.dtype,
                         series[nulls].head(N_FAILURE_CASES).to_dict()))
                else:
                    raise SchemaError(
                        "non-nullable series '%s' contains null values: %s" %
                        (series.name,
                         series[nulls].head(N_FAILURE_CASES).to_dict()))

        # Check if the series contains duplicate values
        if not self._allow_duplicates:
            duplicates = series.duplicated()
            if any(duplicates):
                raise SchemaError(
                    "series '%s' contains duplicate values: %s" %
                    (series.name,
                     series[duplicates].head(N_FAILURE_CASES).to_dict()))

        if series.dtype != _dtype:
            raise SchemaError(
                "expected series '%s' to have type %s, got %s" %
                (series.name, expected_dtype, series.dtype))

        return all(
            check(
                self,
                check_index,
                check.prepare_series_input(series, dataframe))
            for check_index, check in enumerate(self._checks))


class SeriesSchema(SeriesSchemaBase):

    def __init__(self,
                 pandas_dtype,
                 checks: callable = None,
                 nullable: bool = False,
                 allow_duplicates: bool = True,
                 name: str = None):
        """Initialize series schema object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: If element_wise is True, then callable signature should
            be:
            x -> x where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        :param nullable: Whether or not column can contain null values.
        :param allow_duplicates:
        """
        super(SeriesSchema, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates, name)

    def validate(self,
                 series: pd.Series) -> pd.Series:
        """Check if all values in a series have a corresponding column in the
            DataFrameSchema

        :param series: One-dimensional ndarray with axis labels
            (including time series).

        """
        if not isinstance(series, pd.Series):
            raise TypeError("expected %s, got %s" % (pd.Series, type(series)))
        if super(SeriesSchema, self).__call__(series):
            return series
        raise SchemaError()


class Index(SeriesSchemaBase):

    def __init__(self,
                 pandas_dtype,
                 checks: callable = None,
                 nullable: bool = False,
                 allow_duplicates: bool = True,
                 name: str = None):
        super(Index, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates, name)

    def __call__(self, df):
        return super(Index, self).__call__(pd.Series(df.index))

    def __repr__(self):
        if self._name is None:
            return "<Schema Index>"
        return "<Schema Index: '%s'>" % self._name


class MultiIndex(DataFrameSchema):

    def __init__(self, indexes, coerce: bool = False, strict=False):
        super(MultiIndex, self).__init__(
            columns={
                i if index._name is None else index._name: Column(
                    index._pandas_dtype,
                    checks=index._checks,
                    nullable=index._nullable,
                    allow_duplicates=index._allow_duplicates,
                )
                for i, index in enumerate(indexes)
            },
            coerce=coerce,
            strict=strict,
        )

    def __call__(self, df):
        return isinstance(
            super(MultiIndex, self).__call__(df.index.to_frame()),
            pd.DataFrame
        )

    def __repr__(self):
        return "<Schema MultiIndex: '%s'>" % [c for c in self.columns]


class Column(SeriesSchemaBase):

    def __init__(
            self,
            pandas_dtype,
            checks: callable = None,
            nullable: bool = False,
            allow_duplicates: bool = True,
            coerce: bool = False,
            required: bool = True):
        """Initialize column validator object.

        :param pandas_dtype: datatype of the column. If a string is specified,
            then assumes one of the valid pandas string values:
            http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
        :param checks: if element_wise is True, then callable signature should
            be: x -> bool where x is a scalar element in the column. Otherwise,
            x is assumed to be a pandas.Series object.
        :param nullable: Whether or not column can contain null values.
        :param allow_duplicates: Whether or not to coerce the column to the
            specified pandas_dtype before validation
        :param coerce: If True, when schema.validate is called the column will
            be coerced into the specified dtype.
        :param required: Whether or not column is allowed to be missing
        """
        super(Column, self).__init__(
            pandas_dtype, checks, nullable, allow_duplicates)
        self.coerce = coerce
        self.required = required

    def set_name(self,
                 name: str):
        """Used to set or modify the name of a column object.

        :param str name: the name of the column object

        """
        self._name = name
        return self

    def coerce_dtype(self,
                     series: pd.Series) -> pd.Series:
        """Coerce the type of a pd.Series by the type specified in the Column
            object's self._pandas_dtype

        :param series: One-dimensional ndarray with axis labels
            (including time series).

        """
        _dtype = str if self._pandas_dtype is String \
            else self._pandas_dtype.value
        return series.astype(_dtype)

    def __call__(self,
                 df: pd.DataFrame):
        if self._name is None:
            raise RuntimeError(
                "need to `set_name` of column before calling it.")
        return super(Column, self).__call__(
            df[self._name], df.drop(self._name, axis=1))

    def __repr__(self):
        if isinstance(self._pandas_dtype, PandasDtype):
            dtype = self._pandas_dtype.value
        else:
            dtype = self._pandas_dtype
        return "<Schema Column: '%s' type=%s>" % (self._name, dtype)


def _get_fn_argnames(fn: callable):
    arg_spec_args = inspect.getfullargspec(fn).args

    if inspect.ismethod(fn) and arg_spec_args[0] == "self":
        # don't include "self" argument
        arg_spec_args = arg_spec_args[1:]
    return arg_spec_args


def check_input(
        schema: DataFrameSchema,
        obj_getter: Union[int, str, None] = None,
        head: int = None,
        tail: int = None,
        sample: int = None,
        random_state: Optional[int] = None):
    """Validate function argument when function is called.

    This is a decorator function that validates the schema of a dataframe
    argument in a function. Note that if a transformer is specified by the
    schema, the decorator will return the transformed dataframe, which will be
    passed into the decorated function.

    :param schema: dataframe/series schema object
    :param obj_getter:  (Default value = None) if int, obj_getter refers to the
        the index of the pandas dataframe/series to be validated in the args
        part of the function signature. If str, obj_getter refers to the
        argument name of the pandas dataframe/series in the function signature.
        This works even if the series/dataframe is passed in as a positional
        argument when the function is called. If None, assumes that the
        dataframe/series is the first argument of the decorated function
    :param head: validate the first n rows. Rows overlapping with `tail` or
        `sample` are de-duplicated.
    :param tail: validate the last n rows. Rows overlapping with `head` or
        `sample` are de-duplicated.
    :param sample: validate a random sample of n rows. Rows overlapping
        with `head` or `tail` are de-duplicated.
    """

    @wrapt.decorator
    def _wrapper(fn, instance, args, kwargs):
        args = list(args)
        if isinstance(obj_getter, int):
            try:
                args[obj_getter] = schema.validate(args[obj_getter])
            except IndexError as e:
                raise SchemaError(
                        "error in check_input decorator of function '%s': the "
                        "index '%s' was supplied to the check but this "
                        "function accepts '%s' arguments, so the maximum "
                        "index is '%s'. The full error is: '%s'" %
                        (fn.__name__,
                         obj_getter,
                         len(_get_fn_argnames(fn)),
                         max(0, len(_get_fn_argnames(fn))-1),
                         e
                         )
                        )
        elif isinstance(obj_getter, str):
            if obj_getter in kwargs:
                kwargs[obj_getter] = schema.validate(kwargs[obj_getter])
            else:
                arg_spec_args = _get_fn_argnames(fn)
                args_dict = OrderedDict(
                    zip(arg_spec_args, args))
                args_dict[obj_getter] = schema.validate(args_dict[obj_getter])
                args = list(args_dict.values())
        elif obj_getter is None:
            try:
                args[0] = schema.validate(
                    args[0], head, tail, sample, random_state)
            except SchemaError as e:
                raise SchemaError(
                    "error in check_input decorator of function '%s': %s" %
                    (fn.__name__, e))
        else:
            raise ValueError(
                "obj_getter is unrecognized type: %s" % type(obj_getter))
        return fn(*args, **kwargs)

    return _wrapper


def check_output(
        schema: DataFrameSchema,
        obj_getter: Union[int, str, callable, None] = None,
        head: int = None,
        tail: int = None,
        sample: int = None,
        random_state: Optional[int] = None):
    """Validate function output.

    Similar to input validator, but validates the output of the decorated
    function. Note that the `transformer` function supplied to the
    DataFrameSchema will not have an effect in the check_output schema
    validator.

    :param schema: dataframe/series schema object
    :param obj_getter:  (Default value = None) if int, assumes that the output
        of the decorated function is a list-like object, where obj_getter is
        the index of the pandas data dataframe/series to be validated. If str,
        expects that the output is a dict-like object, and obj_getter is the
        key pointing to the dataframe/series to be validated. If a callable is
        supplied, it expects the output of decorated function and should return
        the dataframe/series to be validated.
    :param head: validate the first n rows. Rows overlapping with `tail` or
        `sample` are de-duplicated.
    :param tail: validate the last n rows. Rows overlapping with `head` or
        `sample` are de-duplicated.
    :param sample: validate a random sample of n rows. Rows overlapping
        with `head` or `tail` are de-duplicated.
    """

    @wrapt.decorator
    def _wrapper(fn, instance, args, kwargs):
        if schema.transformer is not None:
            warnings.warn(
                "The schema transformer function has no effect in a "
                "check_output decorator. Please perform the necessary "
                "transformations in the '%s' function instead." % fn.__name__)
        out = fn(*args, **kwargs)
        if obj_getter is None:
            obj = out
        elif isinstance(obj_getter, (int, str)):
            obj = out[obj_getter]
        elif callable(obj_getter):
            obj = obj_getter(out)
        else:
            raise ValueError(
                "obj_getter is unrecognized type: %s" % type(obj_getter))
        try:
            schema.validate(obj, head, tail, sample, random_state)
        except SchemaError as e:
            raise SchemaError(
                "error in check_output decorator of function '%s': %s" %
                (fn.__name__, e))

        return out

    return _wrapper
