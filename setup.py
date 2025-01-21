from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

version = {}
with open("pandera/version.py") as fp:
    exec(fp.read(), version)

_extras_require = {
    "strategies": ["hypothesis >= 6.92.7"],
    "hypotheses": ["scipy"],
    "io": ["pyyaml >= 5.1", "black", "frictionless <= 4.40.8"],
    "pyspark": ["pyspark[connect] >= 3.2.0"],
    "modin": ["modin", "ray", "dask[dataframe]"],
    "modin-ray": ["modin", "ray"],
    "modin-dask": ["modin", "dask[dataframe]"],
    "dask": ["dask[dataframe]"],
    "mypy": ["pandas-stubs"],
    "fastapi": ["fastapi"],
    "geopandas": ["geopandas", "shapely"],
    "polars": ["polars >= 0.20.0"],
}

extras_require = {
    **_extras_require,
    "all": list(set(x for y in _extras_require.values() for x in y)),
}

setup(
    name="pandera",
    version=version["__version__"],
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    description="A light-weight and flexible data validation and testing tool for statistical data objects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pandera-dev/pandera",
    project_urls={
        "Documentation": "https://pandera.readthedocs.io",
        "Issue Tracker": "https://github.com/pandera-dev/pandera/issues",
    },
    keywords=["pandas", "validation", "data-structures"],
    license="MIT",
    data_files=[("", ["LICENSE.txt"])],
    packages=find_packages(include=["pandera*"]),
    package_data={"pandera": ["py.typed"]},
    install_requires=[
        "numpy >= 1.19.0",
        "packaging >= 20.0",
        "pandas >= 1.2.0",
        "pydantic",
        "typeguard",
        "typing_extensions >= 3.7.4.3 ; python_version<'3.8'",
        "typing_inspect >= 0.6.0",
    ],
    extras_require=extras_require,
    python_requires=">=3.7",
    platforms="any",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
    ],
)
