import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import sys, os

#Import data
#file_loc = "C:/Users/pc/Desktop/Btech.csv"
#file_loc2 = "C:/Users/pc/Desktop/Btech2.csv"
#df1 = pd.read_csv(file_loc, header=2,encoding='latin-1')
#df2 = pd.read_csv(file_loc2, header=2,encoding='latin-1')
#df=pd.concat([df1, df2])

sys.path.append(os.path.realpath('..'))
dirname = sys.path.append(os.path.realpath('..'))
dfB1 = pd.read_excel('Biotech companies.xls', skiprows=7)
dfB2 = pd.read_excel('Biotech companies-2.xls', skiprows=7)
dfB = pd.concat([dfB1, dfB2])
df = dfB

# Keyword search function
df=df[~df['Business Description'].str.contains('LLC')]
df=df[~df['Business Description'].str.contains('Institute', case=False)]
df=df[~df['Business Description'].str.contains('Services')] # Too many false positves if case insenesitive
df=df[~df['Business Description'].str.contains('Cannabis', case=False)]
df=df[~df['Business Description'].str.contains('Consulting', case=False)] # check case sensitivity


# Get first word of company name to match to parent company
first_word = df['Company Name'].str.split().str[0]
# Combine dataframes
result = pd.concat([df, first_word], axis=1)
# Change final column name
result.columns.values[-1] = "For Match"

# Find matches
result['Success']=[x[0] in x[1] for x in zip(result['For Match'], result['Parent Company'])]
result['Success'] = result['Success'].apply(str)

# Remove matches (remove where both first word matches and is operating subsidiary)
result=result[~(result['Company Status'].str.contains('Operating Subsidiary') & result['Success'].str.contains('True'))]

# Plot Biotech companies
for_plot_T1 = result['Year Founded']
for_plot_T1.groupby(for_plot_T1).count().plot(kind="bar") #Counts number per year
sfont = {'fontname':'Arial'}
plt.xlabel("Year", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
plt.tight_layout()
#plt.show()


# Split Public and Private
for_plot_T2 = result[['Year Founded', 'Company Type']]

for_plot_T2pr = for_plot_T2[for_plot_T2['Company Type'].str.contains('Pr')]
for_plot_T2pu = for_plot_T2[for_plot_T2['Company Type'].str.contains('Pu')]

a=for_plot_T2pr.groupby(for_plot_T2pr['Year Founded']).count()
b=for_plot_T2pu.groupby(for_plot_T2pu['Year Founded']).count()

T2=pd.concat([a, b], axis=1)

# Plot for Task 2
fig, ax = subplots()
T2.plot(kind='bar', stacked=True, ax=ax)
plt.xlabel("Year", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
ax.legend(["Private Companies", "Public Companies"])
plt.tight_layout()
#plt.show()

# Start Task 3

mAbs_HitList = ["mAbs", "monoclonal", "antibodies"]
RDNA_HitList = ["RDNA", "R-DNA", "Recombinant"]
Antisense_HitList = ["Antisense"]
GeneTherapy_HitList = ["Gene Therapy"]
Chemicals_HitList = ["Chemicals"]

df["All Description"] = df["Business Description"] + df["Long Business Description"] + df["Product Description"] # Possibly need spaces between

is_mAbs = [any(x in str for x in mAbs_HitList) for str in df['All Description']]
is_RDNA = [any(x in str for x in RDNA_HitList) for str in df['All Description']]
is_Antisense = [any(x in str for x in Antisense_HitList) for str in df['All Description']]
is_GeneTherapy = [any(x in str for x in GeneTherapy_HitList) for str in df['All Description']]
is_Chemicals = [any(x in str for x in Chemicals_HitList) for str in df['All Description']]
print(is_mAbs)
print(sum(is_mAbs))
print(sum(is_RDNA))
print(sum(is_Antisense))
print(sum(is_GeneTherapy))
print(sum(is_Chemicals))