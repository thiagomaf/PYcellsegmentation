import pandas as pd
df_inspect = pd.read_parquet("5B_morphology_focus.ome_s0.transcripts.parquet")
print(df_inspect.columns)
print(df_inspect.head())