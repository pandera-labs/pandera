.. pandera package index documentation toctree

.. currentmodule:: pandera

API
===

The ``io`` module and built-in ``Hypothesis`` checks require a pandera
installation with the corresponding extension, see the
:ref:`installation<installation>` instructions for more details.

Schemas
-------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.schemas.DataFrameSchema
   pandera.schemas.SeriesSchema


Schema Components
-----------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.schema_components.Column
   pandera.schema_components.Index
   pandera.schema_components.MultiIndex


Schema Models
-------------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.model.SchemaModel

**Model Components**

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.model_components.Field
   pandera.model_components.check
   pandera.model_components.dataframe_check

**Typing**

.. autosummary::
   :toctree: generated
   :template: typing_module.rst
   :nosignatures:

   pandera.typing

**Config**

.. autosummary::
   :toctree: generated
   :template: model_component_class.rst
   :nosignatures:

   pandera.model.BaseConfig


Checks
------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.checks.Check
   pandera.hypotheses.Hypothesis


Pandas Data Types
-----------------

.. autosummary::
   :toctree: generated
   :template: pandas_dtype_class.rst
   :nosignatures:

   pandera.dtypes.DataType


Decorators
----------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.decorators.check_input
   pandera.decorators.check_output
   pandera.decorators.check_io
   pandera.decorators.check_types


Schema Inference
----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.schema_inference.infer_schema


IO Utils
--------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pandera.io.from_yaml
   pandera.io.to_yaml
   pandera.io.to_script


Data Synthesis Strategies
-------------------------

.. autosummary::
   :toctree: generated
   :template: strategies_module.rst
   :nosignatures:

   pandera.strategies


Extensions
----------

.. autosummary::
   :toctree: generated
   :template: module.rst
   :nosignatures:

   pandera.extensions


Errors
------

.. autosummary::
   :toctree: generated
   :template: class.rst
   :nosignatures:

   pandera.errors.SchemaError
   pandera.errors.SchemaErrors
   pandera.errors.SchemaInitError
   pandera.errors.SchemaDefinitionError
