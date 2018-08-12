import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import sys, os
import seaborn as sns
import nltk
## nltk.download()

## IDEAS
#Check out PyLucene/EnglishAnalyzer for text analysis...
#Try LSI - latent semantic analysis using nltk in python
#Try ML for clustering/classification? Naive Bayes...
#Trial + Error to improve classifications

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

# ----- Other Processing -----
df["All Description"] = df["Business Description"] + df["Long Business Description"] + df["Product Description"] # Possibly need spaces between
AllDesc_OneString = ' '.join(df['All Description'])

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
for_plot_T2 = df[['Year Founded', 'Company Type']]
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
#ax.legend(["Private Companies", "Public Companies"])
plt.tight_layout()

# ----- Task 3: Types of Drug -----
# Enter keywords to match here:
drugTypeKeywords = pd.concat(
        [pd.DataFrame({'mAbs_keywords': ["mAbs", "monoclonal", "mabs", "monoclonal antibodies"]}),
         pd.DataFrame({'RDNA_keywords': ["RDNA", "R-DNA", "Recombinant", "rDNA", "r-DNA", "recombinant"]}),
         pd.DataFrame({'antisense_keywords': ["Antisense", "antisense", "3GA"]}),
         pd.DataFrame({'geneTherapy_keywords': ["Gene Therapy", "gene therapies", "gene therapy", "Gene therapies"]}),
         pd.DataFrame({'chemicals_keywords': ["Chemicals", "chemicals"]})], 
         axis=1)

def findHits(df_searchIn, df_searchFor):
    df_searchFor = df_searchFor[~pd.isnull(df_searchFor)]
    return [any(x in str for x in df_searchFor) for str in df_searchIn]

df[drugTypeKeywords.columns] = pd.DataFrame([findHits(df['All Description'], drugTypeKeywords[drug]) for drug in drugTypeKeywords.columns]).T.set_index(df.index)

df_yearGrouped = df.groupby(df['Year Founded']).sum()
T3 = df_yearGrouped[drugTypeKeywords.columns]

# Bar Graph for Task 3
sns.set()
fig, ax = subplots()
df_yearGrouped[drugTypeKeywords.columns].plot(kind='bar', stacked=True, ax=ax)
plt.xlabel("Year Founded", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
ax.legend(["mAbs", "RDNA", "Antisense", "Gene Therapy", "Chemicals"])
plt.tight_layout()

# Line Graph for Task 3 (convolution is simply a moving average - for smoothing)
N = 4
T3_convolved = pd.DataFrame([np.convolve(df_yearGrouped[drugtype], np.ones((N,))/N, mode='same') for drugtype in drugTypeKeywords.columns]).T.set_index(df_yearGrouped.index)
T3_convolved.columns = drugTypeKeywords.columns

fig, ax = subplots()
T3_convolved.plot(kind='line', stacked=False, ax=ax)
plt.xlabel("Year Founded", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
ax.legend(["mAbs", "RDNA", "Antisense", "Gene Therapy", "Chemicals"])
plt.tight_layout()
#sns.relplot(x="Year", y="Number of companies", hue=")

# ----- Task 4: Clinical development stage -----
clinDevKeywords = pd.concat(
        [pd.DataFrame({'preclinical_keywords': ["preclinical"]}),
         pd.DataFrame({'phase0_keywords': ["phase 0"]}),
         pd.DataFrame({'phase1_keywords': ["phase I", "phase 1"]}),
         pd.DataFrame({'phase2_keywords': ["phase II", "phase 2"]}),
         pd.DataFrame({'phase3_keywords': ["phase III", "phase 3"]}),
         pd.DataFrame({'phase4_keywords': ["phase IV", "phase 4"]})], 
         axis=1)

df[clinDevKeywords.columns] = pd.DataFrame([findHits(df['All Description'], clinDevKeywords[phase]) for phase in clinDevKeywords.columns]).T.set_index(df.index)
df_yearGrouped = df.groupby(df['Year Founded']).sum()
T4 = df_yearGrouped[clinDevKeywords.columns]

# Plot for Task 4
fig, ax = subplots()
df_yearGrouped[clinDevKeywords.columns].plot(kind='bar', stacked=True, ax=ax)
plt.xlabel("Year Founded", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
ax.legend(["Preclinical", "Phase 0", "Phase I", "Phase II", "Phase III", "Phase IV"])
plt.tight_layout()

N = 4
T4_convolved = pd.DataFrame([np.convolve(df_yearGrouped[phase], np.ones((N,))/N, mode='same') for phase in clinDevKeywords.columns]).T.set_index(df_yearGrouped.index)
T4_convolved.columns = clinDevKeywords.columns

fig, ax = subplots()
T4_convolved.plot(kind='line', stacked=False, ax=ax)
plt.xlabel("Year Founded", **sfont, fontsize=14)
plt.ylabel("Number of Companies ", **sfont, fontsize=14)
plt.title("Biotech Industry", **sfont, fontsize=20)
ax.legend(["Preclinical", "Phase 0", "Phase I", "Phase II", "Phase III", "Phase IV"])
plt.tight_layout()
plt.show()

# Might want to show ratios of phases, possibly over time?

# ----- Task 5: Educational background -----
# Make columns of people, not companies
dfEd['Majors'] = dfEd['Majors'].str.replace('\); ', ')|')
dfEd['Majors'] = dfEd['Majors'].str.split('|')
dfEd_person = (dfEd['Majors'].apply(lambda x: pd.Series(x))
                        .stack()
                        .reset_index(level=1, drop=True)
                        .to_frame('Majors')
                        .join(dfEd[['Company ID', 'Company Name', 'Year Founded']], how='left')
)
dfEd_person['Company-Person ID'] = np.arange(0, np.size(dfEd_person, 0)) + 1

# Tidy up and organize: remove anything between brackets less than 14 characters long such as (Board) or (Prior Board), then split off names from their backgrounds
dfEd_person['Person'] = dfEd_person['Majors'].str.replace('\(.{,14}\)','').str.split('\(', 1, expand=True)[0]
dfEd_person['New Majors'] = dfEd_person['Majors'].str.replace('\(.{,14}\)','').str.split('\(', 1, expand=True)[1]
dfEd_person['New Majors'] = dfEd_person['New Majors'].str.rsplit(')',1,expand=True)[0]

# Make columns of majors, not people
dfEd_person['New Majors'] = dfEd_person['New Majors'].str.split('; ')
dfEd_person = dfEd_person.reset_index(level=0, drop=True)
dfEd_major = (dfEd_person['New Majors'].apply(lambda x: pd.Series(x))
                        .stack()
                        .reset_index(level=1, drop=True)
                        .to_frame('New Majors')
                        .join(dfEd_person[['Majors', 'Company ID', 'Company Name', 'Year Founded', 'Company-Person ID', 'Person']], how='left')
)

# Split off subjects from universities
dfEd_major['University'] = dfEd_major['New Majors'].str.split(' - ', expand=True)[0]
dfEd_major['Major'] = dfEd_major['New Majors'].str.split(' - ', expand=True)[1]

# Plot Bar Charts
#fig, ax = subplots()
fig
dfEd_major['Major'].value_counts().head(15).plot('bar')
plt.title("Biotech Industry Majors", **sfont, fontsize=20)
plt.show()

#fig, ax = subplots()
fig
dfEd_major['University'].value_counts().head(15).plot('bar')
plt.title("Biotech Industry Universities", **sfont, fontsize=20)
plt.show()

#Output writer
writer = pd.ExcelWriter('output.xlsx')
T1.to_excel(writer,'T1')
T2.to_excel(writer,'T2')
T3.to_excel(writer, 'T3')
T4.to_excel(writer, 'T4')
writer.save()

## WordCloud stuff below: need to sort out file and package access
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

