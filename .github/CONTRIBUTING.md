# Contributing

Whether you are a novice or experienced software developer, all contributions
and suggestions are welcome!

## Getting Started

If you are looking to contribute to the _pandera_ codebase, the best place to
start is the [GitHub "issues" tab](https://github.com/pandera-dev/pandera/issues).
This is also a great place for filing bug reports and making suggestions for
ways in which we can improve the code and documentation.

## Contributing to the Codebase

The code is hosted on [GitHub](https://github.com/pandera-dev/pandera/issues),
so you will need to use [Git](http://git-scm.com/) to clone the project and make
changes to the codebase.

First create your own fork of pandera, then clone it:

```bash
# replace <my-username> with your github username
git clone https://github.com/<my-username>/pandera.git
```

Once you've obtained a copy of the code, create a development environment that's
separate from your existing Python environment so that you can make and test
changes without compromising your own work environment.

An excellent guide on setting up python environments can be found
[here](https://pandas.pydata.org/docs/development/contributing.html#creating-a-python-environment).
Pandera offers a `environment.yml` to set up a conda-based environment and
`requirements-dev.txt` for a virtualenv.

### Environment Setup

#### Option 1: `miniconda` Setup

Install [miniconda](https://docs.conda.io/en/latest/miniconda.html), then run:

```bash
conda create -n pandera-dev python=3.12  # or any python version 3.8+
conda env update -n pandera-dev -f environment.yml
conda activate pandera-dev
pip install -e .
```

#### Option 2: `virtualenv` Setup

```bash
pip install virtualenv
virtualenv .venv/pandera-dev
source .venv/pandera-dev/bin/activate
pip install --upgrade pip
pip install -r dev/requirements-3.12.txt  # or any python version 3.8+
pip install -e .
```

#### Run Tests

```bash
pytest tests
```

#### Build Documentation Locally

```bash
make docs
```

#### Adding New Dependencies

This repo uses [mamba](https://github.com/mamba-org/mamba), which is a faster
implementation of [miniconda](https://docs.conda.io/en/latest/miniconda.html),
to run the `nox` test suite. Simply install it via conda-forge:

```bash
conda install -c conda-forge mamba
```

To add new dependencies to the project, first alter the `environment.yml` file. Then to sync the dependencies from the `environment.yml` file to the `requirements.in`, run the following command:

```bash
make nox-requirements
```

This will:

- Invoke `python scripts/generate_pip_deps_from_conda.py` to convert `environment.yml`
  to a `requirements.in` file.
- Use `pip-compile` via the `uv` package to create requirements files in the
  `ci` and `dev` directories. The `ci` requirements files are used by github
   actions, while those in the `dev` directory should be used to create local
   development enviornments.

You can use the resulting `requirements-{3.x}.txt` file to install your dependencies
with `pip`:

```bash
pip install -r dev/requirements-{3.x}.txt  # replace {3.x} with desired python version
```

Moreover to add new extra dependencies in setup.py, it is necessary to add it to
the **_extras_require** dictionary.


#### Set up `pre-commit`

This project uses [pre-commit](https://pre-commit.com/) to ensure that code
standard checks pass locally before pushing to the remote project repo. Follow
the [installation instructions](https://pre-commit.com/#installation), then
set up hooks with `pre-commit install`. After, `black`, `pylint` and `mypy`
checks should be run with every commit.

Make sure everything is working correctly by running

```bash
pre-commit run --all
```

### Making Changes

Before making changes to the codebase or documentation, create a new branch with:

```bash
git checkout -b <my-branch>
```

We recommend following the branch-naming convention described in [Making Pull Requests](#making-pull-requests).

### DCO-signing Commits

This project enforces the [DCO](https://developercertificate.org/) standard for
contributions, which requires authors to sign off on their commits. This can be
done with the `-s` or `--signoff` flag:

```bash
git commit -s -m 'my commit'
```

Refer to [this guide](https://github.com/src-d/guide/blob/master/developer-community/fix-DCO.md#dco-is-missing)
to add sign-offs retroactivately.

### Run the Full Test Suite Locally

Before submitting your changes for review, make sure to check that your changes
do not break any tests by running:

```bash
make nox-tests
```

### Run a Specific Test Suite Locally

The above command will run the tests in mamba virtual environments for all of
the supported pandera extras packages, versions of python, pandas, etc. To run
a test for a specific set of versions, first run:

```bash
nox --list
```

You should see an output like this:

```bash
...
* tests(extra='core', pydantic='1.10.11', python='3.8', pandas='1.5.3') -> Run the test suite.
* tests(extra='strategies', pydantic='1.10.11', python='3.8', pandas='1.5.3') -> Run the test suite.
* tests(extra='hypotheses', pydantic='1.10.11', python='3.8', pandas='1.5.3') -> Run the test suite.
...
```

Then run a specific test condition with:

```bash
nox -db mamba --envdir .nox-mamba -s "tests(extra='core', pydantic='1.10.11', python='3.8', pandas='1.5.3')"
```

### Project Releases

Releases are organized under [milestones](https://github.com/pandera-dev/pandera/milestones),
which are be associated with a corresponding branch. This project uses
[semantic versioning](https://semver.org/), and we recommend prioritizing issues
associated with the next release.

### Contributing Documentation

Maybe the easiest, fastest, and most useful way to contribute to this project
(and any other project) is to contribute documentation. If you find an API
within the project that doesn't have an example or description, or could be
clearer in its explanation, contribute yours!

You can also find issues for improving documentation under the
[docs](https://github.com/pandera-dev/pandera/labels/docs) label. If you have
ideas for documentation improvements, you can create a new issue [here](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=docs&template=documentation-improvement.md&title=)

This project uses Sphinx for auto-documentation and RST syntax for docstrings.
Once you have the code downloaded and you find something that is in need of some
TLD, take a look at the [Sphinx](https://www.sphinx-doc.org/en/1.0/rest.html)
documentation or well-documented
[examples](https://pandera.readthedocs.io/en/stable/_modules/pandera/schemas.html#DataFrameSchema)
within the codebase for guidance on contributing.

You can build the html documentation by running `nox -s docs`. The built
documentation can be found in `docs/_build`.

### Contributing Bugfixes

Bugs are reported under the [bug](https://github.com/pandera-dev/pandera/labels/bug)
label, so if you find a bug create a new issue [here](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=bug&template=bug_report.md&title=).

### Contributing Enhancements

New feature issues can be found under the
[enhancements](https://github.com/pandera-dev/pandera/labels/enhancement) label.
You can request a feature by creating a new issue [here](https://github.com/pandera-dev/pandera/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=).

### Making Pull Requests

Once your changes are ready to be submitted, make sure to push your changes to
your fork of the GitHub repo before creating a pull request. Depending on the
type of issue the pull request is resolving, your pull request should merge
onto the appropriate branch:

#### Bugfixes

- branch naming convention: `bugfix/<issue number>` or `bugfix/<bugfix-name>`
- pull request to: `main`

#### Documentation

- branch naming convention: `docs/<issue number>` or `docs/<doc-name>`
- pull request to: `release/x.x.x` branch if specified in the issue milestone, otherwise `main`

#### Enhancements

- branch naming convention: `feature/<issue number>` or `feature/<enhancement-name>`
- pull request to: `release/x.x.x` branch if specified in the issue milestone, otherwise `main`

We will review your changes, and might ask you to make additional changes
before it is finally ready to merge. However, once it's ready, we will merge
it, and you will have successfully contributed to the codebase!

### Questions, Ideas, General Discussion

Head on over to the [discussion](https://github.com/pandera-dev/pandera/discussions)
section if you have questions or ideas, want to show off something that you
did with `pandera`, or want to discuss a topic related to the project.

### Dataframe Schema Style Guides

We have guidelines regarding dataframe and schema styles that are encouraged
for each pull request.

If specifying a single column DataFrame, this can be expressed as a one-liner:

```python
DataFrameSchema({"col1": Column(...)})
```

If specifying one column with multiple lines, or multiple columns:

```python
DataFrameSchema(
    {
        "col1": Column(
            int,
            checks=[
                Check(...),
                Check(...),
            ]
        ),
    }
)
```

If specifying columns with additional arguments that fit in one line:

```python
DataFrameSchema(
    {"a": Column(int, nullable=True)},
    strict=True
)
```

If specifying columns with additional arguments that don't fit in one line:

```python
DataFrameSchema(
    {
        "a": Column(
            int,
            nullable=True,
            coerce=True,
            ...
        ),
        "b": Column(
            ...,
        )
    },
    strict=True)
```

## Deprecation policy

This project adopts a rolling policy regarding the minimum supported version of its dependencies, based on [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html):

- **Python**: 42 months
- **NumPy**: 24 months
- **Pandas**: 18 months

This means the latest minor (X.Y) version from N months prior. Patch versions (x.y.Z) are not pinned, and only the latest available at the moment of publishing the release is guaranteed to work.
