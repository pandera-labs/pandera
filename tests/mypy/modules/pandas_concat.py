# pylint: skip-file
import pandas as pd

df = pd.DataFrame([[1]])
sr = pd.Series([1])


df_concat = pd.concat([df, df])
sr_concat = pd.concat([sr, sr])
sr_axis1_concat = pd.concat([sr, sr], axis=1)

# mypy error without pandera plugin
df_generator_concat: pd.DataFrame = pd.concat(df for _ in range(3))

# mypy error without pandera plugin
sr_generator_concat: pd.Series = pd.concat(sr for _ in range(3))
