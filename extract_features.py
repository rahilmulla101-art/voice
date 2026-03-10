import pandas as pd

df = pd.read_csv("features.csv")
print("\n==== COLUMNS ====")
print(df.columns)

print("\n==== FIRST ROWS ====")
print(df.head())
