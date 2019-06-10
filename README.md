<div align="center"><img src="docs/source/_static/pandera-banner.svg" width="450"></div>

---

### A light-weight and flexible validation package for [pandas](http://pandas.pydata.org) data structures.

<br>

[![Build Status](https://travis-ci.org/cosmicBboy/pandera.svg?branch=master)](https://travis-ci.org/cosmicBboy/pandera)
[![PyPI version shields.io](https://img.shields.io/pypi/v/pandera.svg)](https://pypi.org/project/pandera/)
[![PyPI license](https://img.shields.io/pypi/l/pandera.svg)](https://pypi.python.org/pypi/pandera/)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Documentation Status](https://readthedocs.org/projects/pandera/badge/?version=latest)](https://pandera.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/cosmicBboy/pandera/branch/master/graph/badge.svg)](https://codecov.io/gh/cosmicBboy/pandera)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pandera.svg)](https://pypi.python.org/pypi/pandera/

## Why?

Because pandas data structures hide a lot of information, and explicitly
validating them in production-critical or reproducible research settings is
a good idea.

And it also makes it easier to review pandas code :)


## Documentation

The official documentation is hosted on ReadTheDocs: https://pandera.readthedocs.io

## Install

```
pip install pandera
```

## Example Usage

### `DataFrameSchema`

```python
import pandas as pd

from pandera import Column, DataFrameSchema, Float, Int, String, Check


# validate columns
schema = DataFrameSchema({
    # the check function expects a series argument and should output a boolean
    # or a boolean Series.
    "column1": Column(Int, Check(lambda s: s <= 10)),
    "column2": Column(Float, Check(lambda s: s < -1.2)),
    # you can provide a list of validators
    "column3": Column(String, [
        Check(lambda s: s.str.startswith("value_")),
        Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
    ]),
})

# alternatively, you can pass strings representing the legal pandas datatypes:
# http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes
schema = DataFrameSchema({
    "column1": Column("int64", Check(lambda s: s <= 10)),
    ...
})

df = pd.DataFrame({
    "column1": [1, 4, 0, 10, 9],
    "column2": [-1.3, -1.4, -2.9, -10.1, -20.4],
    "column3": ["value_1", "value_2", "value_3", "value_2", "value_1"]
})

validated_df = schema.validate(df)
print(validated_df)

#     column1  column2  column3
#  0        1     -1.3  value_1
#  1        4     -1.4  value_2
#  2        0     -2.9  value_3
#  3       10    -10.1  value_2
#  4        9    -20.4  value_1
```

## Tests

```
pip install pytest
pytest tests
```

## Contributing to pandera [![GitHub contributors](https://img.shields.io/github/contributors/cosmicBboy/pandera.svg)](https://github.com/cosmicBboy/pandera/graphs/contributors)
All contributions, bug reports, bug fixes, documentation improvements, enhancements and ideas are welcome.

A detailed overview on how to contribute can be found in the [contributing guide](https://github.com/cosmicBboy/pandera/blob/master/.github/CONTRIBUTING.md) on GitHub.

## Issues

Go [here](https://github.com/cosmicBboy/pandera/issues) to submit feature
requests or bugfixes.
