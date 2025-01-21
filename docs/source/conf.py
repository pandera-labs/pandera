# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import datetime
import doctest
import inspect
import logging as pylogging

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import subprocess
import sys

import sphinx.application
from sphinx.util import logging

import pandera

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

_year = datetime.datetime.now().year
project = "pandera"
author = "Pandera developers"
copyright = f"{_year}, {author}"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
    "sphinx.ext.linkcode",  # link to github, see linkcode_resolve() below
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

doctest_global_setup = """
import platform
import sys
import pandas as pd
import numpy as np
from packaging import version
pd.options.display.max_columns = None # For Travis on macOS
pd.options.display.max_rows = None # For Travis on macOS

try:
    import hypothesis
except ImportError:
    SKIP_STRATEGY = True
else:
    SKIP_STRATEGY = False


SKIP = sys.version_info < (3, 6)
PY36 = sys.version_info < (3, 7)
PANDAS_LT_V2 = version.parse(pd.__version__).release < (1, 0)
PANDAS_GT_V2 = version.parse(pd.__version__).release >= (2, 0)
SKIP_PANDAS_LT_V1 = PANDAS_LT_V2 or PY36
SKIP_PANDAS_LT_V1_OR_GT_V2 = PANDAS_LT_V2 or PANDAS_GT_V2 or PY36
SKIP_SCALING = True
SKIP_SCHEMA_MODEL = SKIP_PANDAS_LT_V1_OR_GT_V2
SKIP_MODIN = True

"""

doctest_default_flags = (
    0
    | doctest.DONT_ACCEPT_TRUE_FOR_1
    | doctest.ELLIPSIS
    | doctest.IGNORE_EXCEPTION_DETAIL
    | doctest.NORMALIZE_WHITESPACE
)

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

# copy CONTRIBUTING.md docs into source directory
root_dir = os.path.dirname(__file__)
shutil.copyfile(
    os.path.join(root_dir, "..", "..", ".github", "CONTRIBUTING.md"),
    os.path.join(root_dir, "CONTRIBUTING.md"),
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [".ipynb_checkpoints/*", "notebooks/try_pandera.ipynb"]

autoclass_content = "both"

autodoc_default_options = {
    "undoc-members": False,
}

# sphinx-autodoc-typehints options
set_type_checking_flag = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

announcement = """
📢 Pandera 0.19.0 now supports <a href="polars.html">Polars</a> 🎉.
If you like this project, <a href='https://github.com/unionai-oss/pandera' target='_blank'>give us a star ⭐️! </a>
"""

html_logo = "_static/pandera-banner.png"
html_favicon = "_static/pandera-favicon.png"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#78ac1b",
        "color-brand-content": "#78ac1b",
        "color-api-highlight-on-target": "#e5fff5",
        "color-announcement-background": "#FEE7B8",
        "color-announcement-text": "#535353",
    },
    "dark_css_variables": {
        "color-brand-primary": "#78ac1b",
        "color-brand-content": "#78ac1b",
        "color-api-highlight-on-target": "#e5fff5",
        "color-announcement-background": "#493100",
    },
    "source_repository": "https://github.com/pandera-dev/pandera",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "announcement": announcement,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css",
    "https://cdn.jsdelivr.net/npm/@docsearch/css@3",
    "default.css",
]
html_js_files = [
    "custom.js",
    "https://cdn.jsdelivr.net/npm/@docsearch/js@3",
    "docsearch_config.js",
]

autosummary_generate = True
autosummary_generate_overwrite = False
autosummary_filename_map = {
    "pandera.Check": "pandera.Check",
    "pandera.check": "pandera.check_decorator",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "pyspark": ("https://spark.apache.org/docs/latest/api/python/", None),
    "modin": ("https://modin.readthedocs.io/en/latest/", None),
    "polars": ("https://docs.pola.rs/py-polars/html/", None),
    "typeguard": ("https://typeguard.readthedocs.io/en/stable/", None),
}

# strip prompts
copybutton_prompt_text = (
    r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
)
copybutton_prompt_is_regexp = True


# this is a workaround to filter out forward reference issue in
# sphinx_autodoc_typehints
class FilterTypeAnnotationWarnings(pylogging.Filter):
    def filter(self, record: pylogging.LogRecord) -> bool:
        # You probably should make this check more specific by checking
        # that dataclass name is in the message, so that you don't filter out
        # other meaningful warnings
        return not (
            # NOTE: forward reference false positive needs to be handled
            # correctly
            record.getMessage().startswith(
                (
                    "Cannot resolve forward reference in type annotations",
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.DataFrame"',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.DataFrame',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.Index',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.Series',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.geopandas.GeoDataFrame',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.geopandas.GeoSeries',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.pyspark.DataFrame',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.pyspark.Series',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.pyspark.Index',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.api.pandas.container.DataFrameSchema',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.DataFrame.style"',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.api.polars.container.DataFrameSchema',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.api.pyspark.container.DataFrameSchema',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.Series"',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.modin.DataFrame',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.modin.Series',
                    "Cannot resolve forward reference in type annotations of "
                    '"pandera.typing.modin.Index',
                )
            )
        )


logging.getLogger("sphinx_autodoc_typehints").logger.addFilter(
    FilterTypeAnnotationWarnings()
)


# based on pandas/doc/source/conf.py
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != "py":
        return None

    submod = sys.modules.get(info["module"])
    if submod is None:
        return None

    obj = submod
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None

    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))
    except TypeError:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        lineno = None

    fn = os.path.relpath(fn, start=os.path.dirname(pandera.__file__))

    linespec = f"#L{lineno}-L{lineno + len(source) - 1}" if lineno else ""

    try:  # check if we are building "latest" version on reathedocs
        tag = (
            subprocess.check_output(["git", "branch", "--show-current"])
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError:
        tag = None

    if tag != "main":
        tag = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )

    return f"https://github.com/pandera-dev/pandera/blob/{tag}/pandera/{fn}{linespec}"


# myst-nb configuration
myst_enable_extensions = [
    "colon_fence",
]
myst_heading_anchors = 3

nb_execution_mode = os.getenv("PANDERA_DOCS_NB_EXECUTION_MODE", "auto")
nb_execution_timeout = 60
nb_execution_excludepatterns = ["_contents/try_pandera.ipynb"]

# docsearch configuration
docsearch_container = "#docsearch"
docsearch_app_id = os.getenv("DOCSEARCH_SEARCH_APP_ID", "GA9NROLUXR")
docsearch_api_key = os.getenv("DOCSEARCH_SEARCH_API_KEY")
docsearch_index_name = os.getenv("DOCSEARCH_INDEX_NAME", "pandera")
docsearch_search_parameters = {
    "facetFilters": [f"version:{os.getenv('READTHEDOCS_VERSION', 'stable')}"]
}


class CustomWarningSuppressor(pylogging.Filter):
    """Filter logs by `suppress_warnings`."""

    def __init__(self, app: sphinx.application.Sphinx) -> None:
        self.app = app
        super().__init__()

    def filter(self, record: pylogging.LogRecord) -> bool:
        msg = record.getMessage()

        # TODO: These are all warnings that should be fixed as follow-ups to the
        # monodocs build project.
        filter_out = (
            "Definition list ends without a blank line; unexpected unindent",
            "Unexpected indentation",
            "Block quote ends without a blank line; unexpected unindent",
        )

        if msg.strip().startswith(filter_out):
            return False

        if (
            msg.strip().startswith("document isn't included in any toctree")
            and record.location == "_tags/tagsindex"
        ):
            # ignore this warning, since we don't want the side nav to be
            # cluttered with the tags index page.
            return False

        return True


def add_warning_suppressor(app: sphinx.application.Sphinx) -> None:
    logger = pylogging.getLogger("sphinx")
    warning_handler, *_ = (
        h
        for h in logger.handlers
        if isinstance(h, logging.WarningStreamHandler)
    )
    warning_handler.filters.insert(0, CustomWarningSuppressor(app))


def add_docsearch_config(app: sphinx.application.Sphinx) -> None:
    app.add_config_value(
        "docsearch_app_id", default="", rebuild="html", types=[str]
    )
    app.add_config_value(
        "docsearch_api_key", default="", rebuild="html", types=[str]
    )
    app.add_config_value(
        "docsearch_index_name", default="", rebuild="html", types=[str]
    )
    app.add_config_value(
        "docsearch_container",
        default="#docsearch",
        rebuild="html",
        types=[str],
    )
    app.add_config_value(
        "docsearch_search_parameters", default="", rebuild="html", types=[dict]
    )


def add_docsearch_assets(
    app: sphinx.application.Sphinx, config: sphinx.application.Config
):
    app.add_js_file("docsearch_config.js", loading_method="defer")

    # Update global context
    config.html_context.update(
        {
            "docsearch_app_id": config.docsearch_app_id,
            "docsearch_api_key": app.config.docsearch_api_key,
            "docsearch_index_name": app.config.docsearch_index_name,
            "docsearch_container": app.config.docsearch_container,
            "docsearch_search_parameters": app.config.docsearch_search_parameters,
        }
    )


def setup(app: sphinx.application.Sphinx) -> None:
    add_warning_suppressor(app)
    add_docsearch_config(app)

    app.connect("config-inited", add_docsearch_assets)
