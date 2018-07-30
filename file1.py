import pandas as pd
import numpy as np

print("this should print now")
file_loc = "C:/Users/bpbla/OneDrive/Documents/Cambridge/BiotechVC_CJBS/payme1.csv"
df = pd.read_csv(file_loc)

df=df[(~df['this one'].str.contains('me'))]
# df=df[df['test'].str.contains('abc') & ]
print(df)
