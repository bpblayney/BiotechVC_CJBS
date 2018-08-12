import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import sys, os
import seaborn as sns
import nltk

sys.path.append(os.path.realpath('..'))
#dirname = sys.path.append(os.path.realpath('..'))
dfB1 = pd.read_excel('Biotech companies.xls', skiprows=7)
dfB2 = pd.read_excel('Biotech companies-2.xls', skiprows=7)
dfEd = pd.read_excel('Educational Background.xls')
df = pd.concat([dfB1, dfB2])

 # ----- Top-Level Filtering -----
filterOut_caseSensitive = ['LLC', 'Services']
filterOut = ['Institute', 'Cannabis', 'Consulting']
df = df[[not(any(x in s for x in filterOut_caseSensitive)) for s in df['Business Description']]]
df = df[[not(any(x.lower() in s.lower() for x in filterOut)) for s in df['Business Description']]]

 # ----- Remove Subsidiaries -----
def Remove_Subsidiaries(df):
    # Get first word of company name to match to parent company
    first_word = df['Company Name'].str.split().str[0] # Note: some company names have commas after - still works but cleaner to remove punctuation
    result = pd.concat([df, first_word], axis=1) # Combine dataframes
    result.columns.values[-1] = "For Match" # Change final column name
    
    # Find matches
    result['Success']=[x[0] in x[1] for x in zip(result['For Match'], result['Parent Company'])]
    result['Success'] = result['Success'].apply(str)
    
    # Remove matches (remove where both first word matches and is operating subsidiary)
    result = result[~(result['Company Status'].str.contains('Operating Subsidiary') & result['Success'].str.contains('True'))]
    return result
    df = Remove_Subsidiaries(df)
 
 # ----- Task 1: Companies by year founded -----
for_plot_T1 = df['Year Founded']
for_plot_T1.groupby(for_plot_T1).count().plot(kind="bar", color='#ffb81c') # Counts number per year
sfont = {'fontname':'Arial'}
plt.xlabel("Year Founded", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
#plt.title("Biotech Industry", **sfont, fontsize=20)
plt.tight_layout()

T1=for_plot_T1.groupby(for_plot_T1).count() #For output file
T1=T1.to_frame()
T1.rename(columns={T1.columns[0]: "Number of Companies"}, inplace=True)

 # ----- Task 2: Public and Private Companies -----
for_plot_T2pr = for_plot_T2[for_plot_T2['Company Type'].str.contains('Pr')]
for_plot_T2pu = for_plot_T2[for_plot_T2['Company Type'].str.contains('Pu')]

a=for_plot_T2pr.groupby(for_plot_T2pr['Year Founded']).count()
b=for_plot_T2pu.groupby(for_plot_T2pu['Year Founded']).count()
a.rename(columns={a.columns[0]: "Private Companies"}, inplace=True)
b.rename(columns={b.columns[0]: "Public Companies"}, inplace=True)
T2=pd.concat([a, b], axis=1)
 # Plot for Task 2
fig, ax = subplots()
T2.plot(kind='bar', stacked=True, ax=ax, color=['#ffb81c', '#544c41'])
plt.xlabel("Year", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
ax.legend(["Private Companies", "Public Companies"])
plt.tight_layout()

 # ----- Task 3: Types of Drug -----
mAbs_HitList = ["mAbs", "monoclonal", "mabs", "monoclonal antibodies"]
RDNA_HitList = ["RDNA", "R-DNA", "Recombinant", "rDNA", "r-DNA", "recombinant"]
Antisense_HitList = ["Antisense", "antisense", "3GA"]
GeneTherapy_HitList = ["Gene Therapy", "gene therapies", "gene therapy", "Gene therapies"]
Chemicals_HitList = ["Chemicals", "chemicals"]
df["All Description"] = df["Business Description"] + df["Long Business Description"] + df["Product Description"] # Possibly need spaces between
AllDesc_OneString = ' '.join(df['All Description'])
df["is_mAbs"] = [any(x in str for x in mAbs_HitList) for str in df['All Description']]
df["is_RDNA"] = [any(x in str for x in RDNA_HitList) for str in df['All Description']]
df["is_Antisense"] = [any(x in str for x in Antisense_HitList) for str in df['All Description']]
df["is_GeneTherapy"] = [any(x in str for x in GeneTherapy_HitList) for str in df['All Description']]
df["is_Chemicals"] = [any(x in str for x in Chemicals_HitList) for str in df['All Description']]
#foo = df[df["is_mAbs"]]
plotT3_mAbs = df[['Year Founded', "is_mAbs"]]
plotT3_mAbs = plotT3_mAbs.groupby(plotT3_mAbs['Year Founded']).sum()
plotT3_RDNA = df[['Year Founded', "is_RDNA"]]
plotT3_RDNA = plotT3_RDNA.groupby(plotT3_RDNA['Year Founded']).sum()
plotT3_Antisense = df[['Year Founded', "is_Antisense"]]
plotT3_Antisense = plotT3_Antisense.groupby(plotT3_Antisense['Year Founded']).sum()
plotT3_GeneTherapy = df[['Year Founded', "is_GeneTherapy"]]
plotT3_GeneTherapy = plotT3_GeneTherapy.groupby(plotT3_GeneTherapy['Year Founded']).sum()
plotT3_Chemicals = df[['Year Founded', "is_Chemicals"]]
plotT3_Chemicals = plotT3_Chemicals.groupby(plotT3_Chemicals['Year Founded']).sum()
T3 = pd.concat([plotT3_mAbs, plotT3_RDNA, plotT3_Antisense, plotT3_GeneTherapy, plotT3_Chemicals], axis=1)
 # Plot for Task 3
sns.set()
fig, ax = subplots()
T3.plot(kind='bar', stacked=True, ax=ax)
plt.xlabel("Year", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
ax.legend(["mAbs", "RDNA", "Antisense", "Gene Therapy", "Chemicals"])
plt.tight_layout()
N=4
T3n = pd.DataFrame(data = np.convolve(T3["is_mAbs"], np.ones((N,))/N, mode='same'),
                   index = T3.index.values,
                   columns = ["is_mAbs"])
T3n["is_RDNA"] = np.convolve(T3["is_RDNA"], np.ones((N,))/N, mode='same')
T3n["is_Antisense"] = np.convolve(T3["is_Antisense"], np.ones((N,))/N, mode='same')
T3n["is_GeneTherapy"] = np.convolve(T3["is_GeneTherapy"], np.ones((N,))/N, mode='same')
T3n["is_Chemicals"] = np.convolve(T3["is_Chemicals"], np.ones((N,))/N, mode='same')
fig, ax = subplots()
T3n.plot(kind='line', stacked=False, ax=ax)
plt.xlabel("Year", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
ax.legend(["mAbs", "RDNA", "Antisense", "Gene Therapy", "Chemicals"])
plt.tight_layout()
 #sns.relplot(x="Year", y="Number of companies", hue=")
 # Task 4: Clinical development stage
phase0_HitList = ["phase 0"]
phase1_HitList = ["phase I", "phase 1"]
phase2_HitList = ["phase II", "phase 2"]
phase3_HitList = ["phase III", "phase 3"]
phase4_HitList = ["phase IV", "phase 4"]
preclinical_HitList = ["preclinical"]
df["is_phase0"] = [any(x in str for x in phase0_HitList) for str in df['All Description']]
df["is_phase1"] = [any(x in str for x in phase1_HitList) for str in df['All Description']]
df["is_phase2"] = [any(x in str for x in phase2_HitList) for str in df['All Description']]
df["is_phase3"] = [any(x in str for x in phase3_HitList) for str in df['All Description']]
df["is_phase4"] = [any(x in str for x in phase4_HitList) for str in df['All Description']]
df["is_preclinical"] = [any(x in str for x in preclinical_HitList) for str in df['All Description']]

plotT4_phase0 = df[['Year Founded', "is_phase0"]]
plotT4_phase0 = plotT4_phase0.groupby(plotT4_phase0['Year Founded']).sum()
plotT4_phase1 = df[['Year Founded', "is_phase1"]]
plotT4_phase1 = plotT4_phase1.groupby(plotT4_phase1['Year Founded']).sum()
plotT4_phase2 = df[['Year Founded', "is_phase2"]]
plotT4_phase2 = plotT4_phase2.groupby(plotT4_phase2['Year Founded']).sum()
plotT4_phase3 = df[['Year Founded', "is_phase3"]]
plotT4_phase3 = plotT4_phase3.groupby(plotT4_phase3['Year Founded']).sum()
plotT4_phase4 = df[['Year Founded', "is_phase4"]]
plotT4_phase4 = plotT4_phase4.groupby(plotT4_phase4['Year Founded']).sum()
plotT4_preclinical = df[['Year Founded', "is_preclinical"]]
plotT4_preclinical = plotT4_preclinical.groupby(plotT4_preclinical['Year Founded']).sum()
T4 = pd.concat([plotT4_preclinical, plotT4_phase0, plotT4_phase1, plotT4_phase2, plotT4_phase3, plotT4_phase4], axis=1)
 
 # Plot for Task 4
fig, ax = subplots()
T4.plot(kind='bar', stacked=True, ax=ax)
plt.xlabel("Year", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
ax.legend(["Preclinical", "Phase 0", "Phase I", "Phase II", "Phase III", "Phase IV"])
plt.tight_layout()
plt.show()

 # Start Task 5
dfEd['Majors'] = dfEd['Majors'].str.replace('\); ', ')|')
dfEd['Majors'] = dfEd['Majors'].str.split('|')
newEd = dfEd[['Company ID', 'Company Name', 'Year Founded', 'Majors']]
newEd2 = (newEd['Majors'].apply(lambda x: pd.Series(x))
                        .stack()
                        .reset_index(level=1, drop=True)
                        .to_frame('Majors')
                        .join(newEd[['Company ID', 'Company Name', 'Year Founded']], how='left')
)
newEd2['Company-Person ID'] = np.arange(0, np.size(newEd2, 0)) + 1
#foo = newEd2['Majors'].str.replace('\(Board\)','').str.replace('\(Prior\)','').str.replace('\(Prior Board\)','').str.split('\(', 1, expand=True)
#foo1 = foo[0]
#foo2 = newEd2['Majors'].str.replace('\(Board\)','').str.replace('\(Prior\)','').str.replace('\(Prior Board\)','').str.split('\(', 1, expand=True)[:,1]
#include all (Prior Board, Deceased)
newEd2['Person'] = newEd2['Majors'].str.replace('\(Board\)','').str.replace('\(Prior\)','').str.replace('\(Prior Board\)','').str.split('\(', 1, expand=True)[0]
newEd2['New Majors'] = newEd2['Majors'].str.replace('\(Board\)','').str.replace('\(Prior\)','').str.replace('\(Prior Board\)','').str.split('\(', 1, expand=True)[1]
newEd2['New Majors'] = newEd2['New Majors'].str.rsplit(')',1,expand=True)[0]
newEd2['New Majors'] = newEd2['New Majors'].str.split('; ')
newEd2 = newEd2.reset_index(level=0, drop=True)
newEd3 = (newEd2['New Majors'].apply(lambda x: pd.Series(x))
                        .stack()
                        .reset_index(level=1, drop=True)
                        .to_frame('New Majors')
                        .join(newEd2[['Majors', 'Company ID', 'Company Name', 'Year Founded', 'Company-Person ID', 'Person']], how='left')
)
newEd3['University'] = newEd3['New Majors'].str.split(' - ', expand=True)[0]
newEd3['Major'] = newEd3['New Majors'].str.split(' - ', expand=True)[1]
 #fig, ax = subplots()
fig
newEd3['Major'].value_counts().head(15).plot('bar')
plt.title("Biotech Industry Majors", **sfont, fontsize=20)
plt.show()
 #fig, ax = subplots()
fig
newEd3['University'].value_counts().head(15).plot('bar')
plt.title("Biotech Industry Universities", **sfont, fontsize=20)
plt.show()

#For Output
writer = pd.ExcelWriter('output.xlsx')
T1.to_excel(writer,'T1')
T2.to_excel(writer,'T2')
T3.to_excel(writer, 'T3')
T4.to_excel(writer, 'T4')
writer.save()
 ## Don't have file (also don't have wordcloud so maybe he wouldn't be able to run without installing it?)
## For wordcloud
#import os
#from os import path
#from wordcloud import WordCloud
#file_loc = "C:/Users/pc/Desktop/Edback.csv"
#dfwc = pd.read_csv(file_loc, header=0,encoding='latin-1')
#
#bin_words=('Richard', 'Peter', 'University', "University of", "Technology", 'Prior', 'Board', 'of', 'James', 'the', 'The', 'Michael', 'John', 'David', 'State', 'Robert', 'William', 'Thomas','Stephen' ,'Andrew', 'Mark')
#wordcloud = WordCloud( width=800, height=600,  min_font_size=11, stopwords=bin_words, background_color='white',relative_scaling=0.5).generate(' '.join(dfwc['Majors']))
#plt.imshow(wordcloud, interpolation='bilinear')
#plt.axis("off")
#plt.show()