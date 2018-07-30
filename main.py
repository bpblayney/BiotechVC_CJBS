# This is the main file!

import pandas as pd
import numpy as np
import sys, os

print("this should print now")
sys.path.append(os.path.realpath('..'))
dirname = sys.path.append(os.path.realpath('..'))
df = pd.read_excel('Biotech companies.xls', skiprows=7)

print(df.head())