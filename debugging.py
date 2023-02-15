import pandera as pa

# TODO:
# - ✅ refactor pandas_strategies so that all uses of check.strategy are replaced
#   the STRATEGY_DISPATCHER to fetch the correct strategy function based on the
#   check name and datatype
# - Define built-in checks as actual pandera.core.checks.Check methods. Then
#   use the @register_check for two purposes: (i) as a decorator for
#   type-specific implementations of the built-in checks, or (ii) to register
#   new, user-defined custom checks.

# schema = pa.DataFrameSchema({"columns": pa.Column(int, pa.Check.gt(0))}).example()
# print(schema)
schema = pa.DataFrameSchema(
    {"col1": pa.Column(int)}, checks=[pa.Check.gt(0)]
).example(size=3)
print(schema)


check = getattr(pa.Check, "ge")(0)
print(check)
print(check._check_fn)
print(check._check_fn.__module__)
print(check._check_fn.__qualname__)
