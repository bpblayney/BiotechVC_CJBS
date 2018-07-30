import pandas as pd
import numpy as np
import sys, os

print("this should print now")
# file_loc = "C:/Users/bpbla/OneDrive/Documents/Cambridge/BiotechVC_CJBS/payme1.csv"
sys.path.append(os.path.realpath('..'))
#dirname = os.path.dirname(__file__)
dirname = sys.path.append(os.path.realpath('..'))
#file_loc = os.path.join(dirname, 'payme1.csv')
#df = pd.read_csv(file_loc)
df = pd.read_csv('payme1.csv')

df=df[(~df['this one'].str.contains('me'))]
# df=df[df['test'].str.contains('abc') & ]
print(df)
