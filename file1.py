import pandas as pd
import numpy as np
file_loc = "C:/Users/pc/Desktop/payme1.csv"
df = pd.read_csv(file_loc)

df=df[(~df['this one'].str.contains('me'))]
#df=df[df['test'].str.contains('abc') & ]
print(df)
