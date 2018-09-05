import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import sys, os
import seaborn as sns
import nltk
import string
import re
import difflib
#import fuzzywuzzy
## nltk.download()

## ISSUES
# Too few cases in many plots (e.g. clinical development phases) - not finding enough relevant companies
# "intends to enter phase X" seems to come up semi-frequently..

## IDEAS
# Check out PyLucene/EnglishAnalyzer for text analysis...
# Try LSI - latent semantic analysis using nltk in python
# Try ML for clustering/classification? Naive Bayes..
### Use current classifications to similarity match to improve completeness of data
# Trial + Error to improve classifications

## TO-DO LIST
# Ensure subsidiaries being removed properly (done), and initial filtering cases...
# Cases: no evidence -> filter in, 
#        drug distribution/operations/etc -> filter out, 
#        drug development but also distribution/operation/marketing -> filter in (provide evidence this exists), 
#        non-drug -> filter out (how do we ensure this?)
# Drug-type graph both ways: one company counted multiple times AND company categorized as "multiple" if multiple (done)
# Geographical breakdown for all tasks (done)
# Processed dataframe exported as excel output (done)
# Clear to read dataframe: Standard columns/Public/mAbs/Phase II (done)
# Categorization columns, "other" otherwise (done.. ?)
# Produce a table for each graph, so can see exact numbers (done)
# T5 key executives only (done)
# Seaborn graphs for all (seaborn cannot do stacked bar) (done)

## TO-DO LIST 2
# T5: do not allow an executive of multiple companies to count more than once (done)
# T5: Only look at educational background of the clean dataset (done)
# New T4 category: "market" (done)
# T3 chemicals list (done? needs improvement..)
# Better T1 filtering (cases 1-5)
# Improve all classifications - look at ml and nlp techniques.



sys.path.append(os.path.realpath('..'))
#dirname = sys.path.append(os.path.realpath('..'))
dfB1 = pd.read_excel('Biotech companies.xls', skiprows=7)
dfB2 = pd.read_excel('Biotech companies-2.xls', skiprows=7)
df_unitTest = pd.read_excel('UnitTestSheet.xlsx', skiprows=7)
dfEd = pd.read_excel('Educational Background v3.xls', skiprows=7)
df = pd.concat([dfB1, dfB2]).reset_index(drop=True)

# ----- Other Processing -----
df["All Description"] = df[["Company Name", "Business Description", "Long Business Description", "Product Description"]].apply(lambda x: ' || '.join(x), axis=1)
#dfEd["All Description"] = dfEd[["Company Name", "Business Description", "Long Business Description", "Product Description"]].apply(lambda x: ' || '.join(x), axis=1)
AllDesc_OneString = ' '.join(df['All Description'])
df_all = df
allCountries = ["European Developed Markets (Primary)", "United States of America (Primary)", "Canada (Primary)"]
sfont = {'fontname':'Arial'}

# ----- General/Useful Functions -----
def stripNonAlphaNum(text):
    return ' '.join(re.compile(r'\W+', re.UNICODE).split(text))

def norm(text):
    return ''.join(stripNonAlphaNum(text).lower().split())

# Convert data to different forms (wideform and longform... don't end up using longform but keeping because useful to know how to do...)
def to_wideform(df, id_vars="Year Founded", value_vars="Most Developed Phase"): return pd.melt(df, id_vars=id_vars, value_vars=value_vars, value_name=value_vars, var_name="Frequency").groupby([id_vars, value_vars])[value_vars].count().unstack(value_vars)

def findHits(df_searchIn, df_searchFor):
    df_searchFor = df_searchFor[~pd.isnull(df_searchFor)]
    return [any(norm(x) in norm(str) for x in df_searchFor) for str in df_searchIn]

def calc_moving_avg(df_wideform, N):
    df_convolved = pd.DataFrame([np.convolve(df_wideform[column], np.ones((N,))/N, mode='same') for column in df_wideform.columns]).T.set_index(df_wideform.index)
    df_convolved.columns = df_wideform.columns
    return df_convolved

# ----- Top-Level Filtering -----
def Remove_notDrugDevTech(df):
    # df must have an "All Description" column
    filterOut_caseSensitive = ['LLC', 'Services']
    filterOut = ['Institute', 'Cannabis', 'Consulting', 'Marijuana', 'cannabidiol', 'Cannabinoid', 'Weed', 'Hemp']
    boolIndex_DrugDevTech = np.logical_or([any(x in s for x in filterOut_caseSensitive) for s in df['All Description']], 
                                      [any(norm(x) in norm(s) for x in filterOut) for s in df['All Description']])
    df_notDrugDevTech = df[boolIndex_DrugDevTech]
    df_DrugDevTech = df[np.invert(boolIndex_DrugDevTech)]
    return df_DrugDevTech, df_notDrugDevTech, boolIndex_DrugDevTech

def Remove_Subsidiaries(df):
    # Get first word of company name to match to parent company
    first_word = df['Company Name'].str.split().str[0] # Note: some company names have commas after - still works but cleaner to remove punctuation
    result = pd.concat([df, first_word], axis=1) # Combine dataframes
    result.columns.values[-1] = "For Match" # Change final column name
    
    # Find matches
    result['Success']=[norm(x[0]) in norm(x[1]) for x in zip(result['For Match'], result['Parent Company'])]
    result['Success'] = result['Success'].apply(str)
    
    # Remove matches (remove where both first word matches and is operating subsidiary)
    boolindex = ~(result['Company Status'].str.contains('Operating Subsidiary') & result['Success'].str.contains('True'))
    result = result[~(result['Company Status'].str.contains('Operating Subsidiary') & result['Success'].str.contains('True'))]
    return result, boolindex

df_DrugDevTech, df_notDrugDevTech, _ = Remove_notDrugDevTech(df_all)
df = df_DrugDevTech
df, _ = Remove_Subsidiaries(df)

dfEd["In Original Dataset"] = dfEd["Company Name"].isin(df["Company Name"])
# Below does not work because no "Parent Company" information in excel file
#dfEd_DrugDevTech, dfEd_notDrugDevTech, _ = Remove_notDrugDevTech(dfEd)
#dfEd = dfEd_DrugDevTech
#dfEd, _ = Remove_Subsidiaries(dfEd)

# ----- Unit Testing -----
df_unitTest.dropna(subset=["Drug Dev Tech Test", "Subsidiary"], inplace = True, how='all')
df_unitTest["All Description"] = df_unitTest[["Company Name", "Business Description", "Long Business Description", "Product Description"]].apply(lambda x: ' || '.join(x), axis=1)
df_unitTest_all = df_unitTest
df_unitTest_DDT, df_unitTest_nDDT, df_unitTest_boolIndex = Remove_notDrugDevTech(df_unitTest_all)
df_unitTest["IsDDT"] = np.invert(df_unitTest_boolIndex)
df_unitTest["Failed DDT Test"] = np.abs(df_unitTest["IsDDT"] - df_unitTest["Drug Dev Tech Test"]) #1 for failed, 0 for passed

_, df_unitTest_boolIndex = Remove_Subsidiaries(df_unitTest)
df_unitTest["IsSub"] = np.invert(df_unitTest_boolIndex)
df_unitTest["Failed Sub Test"] = np.abs(df_unitTest["IsSub"] - df_unitTest["Subsidiary"]) #1 for failed, 0 for passed
# Most failures at the moment are companies that include LLC or Services but that definitely develop and manufacture drugs.

# ----- Task 1: Companies by year founded -----
def getT1CountryTable(df, countries):
    T1_table = pd.DataFrame()
    for country in countries:
        df_temp = df[findHits(df["Geographic Locations"], np.array(country))]
        T1_table[country] = df_temp['Year Founded'].groupby(df_temp['Year Founded']).count()
    return T1_table

def graph_T1(df, graphType = 'line', countries=allCountries, countryComparison = False):
    if countryComparison: T1 = getT1CountryTable(df, countries)
    else:
        df = df[findHits(df["Geographic Locations"], np.array(countries))]
        T1 = df['Year Founded'].groupby(df['Year Founded']).count().to_frame() #For output file
        T1.rename(columns={T1.columns[0]: "Number of Companies"}, inplace=True)
    caption = re.sub(r"[\(\[].*?[\)\]]", "", ("Countries: " + ", ".join(countries))).replace(" , ", ", ")
    fig, ax = plt.subplots()
    if graphType == 'bar':
        if countryComparison: T1.plot(kind="bar", ax=ax, stacked=True)
        else: T1.plot(kind="bar", color='#ffb81c', ax=ax, stacked=True) # Counts number per year
    elif graphType == 'line':
        ax = sns.lineplot(data=T1, dashes=False)
    sfont = {'fontname':'Arial'}
    plt.xlabel("Year Founded", **sfont, fontsize=14)
    plt.ylabel("Number of Companies ", **sfont, fontsize=14)
    plt.title("Biotech Industry", **sfont, fontsize=20)
    ax.annotate(caption, (0,0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.tight_layout()
    plt.show()

graph_T1(df, 'bar')
graph_T1(df, 'bar', ["European Developed Markets (Primary)"])
graph_T1(df, 'bar', ["United States of America (Primary)"])
graph_T1(df, 'line', allCountries, True)
graph_T1(df, 'bar', allCountries, True)

print("Task 1 Table: Total Number of Pharmaceutical Companies")
T1_table = getT1CountryTable(df, allCountries).fillna(0)
print(T1_table)

# ----- Task 2: Public and Private Companies -----
def graph_T2(df, countries=allCountries):
    df = df[findHits(df["Geographic Locations"], np.array(countries))]
    for_plot_T2 = df[['Year Founded', 'Company Type']]
    for_plot_T2pr = for_plot_T2[for_plot_T2['Company Type'].str.contains('Pr')]
    for_plot_T2pu = for_plot_T2[for_plot_T2['Company Type'].str.contains('Pu')]
    
    T2 = pd.concat([for_plot_T2pr.groupby(for_plot_T2pr['Year Founded']).count(), 
                    for_plot_T2pu.groupby(for_plot_T2pu['Year Founded']).count()], axis=1)
    T2.columns = ["Private Companies", "Public Companies"]
    
    # Plot for Task 2
    caption = re.sub(r"[\(\[].*?[\)\]]", "", ("Countries: " + ", ".join(countries))).replace(" , ", ", ")
    fig, ax = plt.subplots()
    T2.plot(kind='bar', stacked=True, ax=ax, color=['#ffb81c', '#544c41'])
    plt.xlabel("Year", **sfont, fontsize=14)
    plt.ylabel("Number of Companies ", **sfont, fontsize=14)
    plt.title("Public and Private Companies", **sfont, fontsize=20)
    ax.legend(["Private Companies", "Public Companies"])
    ax.annotate(caption, (0,0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.tight_layout()
    plt.show()
    return T2

graph_T2(df)
graph_T2(df, ["United States of America (Primary)"])
graph_T2(df, ["European Developed Markets (Primary)"])
T2_table = graph_T2(df).fillna(0)
print("Task 2 Table: Private and Public Companies")
print(T2_table)

# ----- Task 3: Types of Drug -----
# drugTypeKeywords can be edited here for better classification
# mAbs drugs always end in mab, but mab search also finds 'programmable' and 'consumable', 'presumable'....
# AON for antisense might have similar problems
drugTypeKeywords = pd.concat(
        [pd.DataFrame({'mAbs': ["mAbs", "monoclonal", "mabs", "monoclonal antibodies", "moAb", "hybridoma", "immunoglobulin", "phage display", "single B cell", "imab", "zumab", "mumab", "amab", "emab", "omab", "tmab", "gumab", "iumab", "cumab", "rumab", "numab", "lumab", "tumab", "xaumab"]}),
         pd.DataFrame({'RDNA': ["RDNA", "R-DNA", "Recombinant", "rDNA", "r-DNA", "recombinant", "chimeric DNA", "molecular cloning", "palindromic sequence"]}),
         pd.DataFrame({'Antisense': ["Antisense", "antisense", "3GA", "AONs", "oligonucleotides", "siRNA"]}),
         pd.DataFrame({'Gene Therapy': ["Gene Therapy", "gene therapies", "gene therapy", "Gene therapies", "gene transfer", "glybera", "gendicine", "neovasculgen", "gene editing", "SCGT", "GGT"]}),
         pd.DataFrame({'Chemicals': [" Chemicals ", " chemicals ", " chemical ", "NBCD", "polypeptide", "liposome", "small molecules", "oral", "orally"]})], 
         axis=1)

def findDrugType(keywords):
    if sum(keywords) == 2: return "Multiple"
    elif sum(keywords) == 0: return "None"
    else: return keywords[keywords].index[0]

# Possible room for speed-up..
df[drugTypeKeywords.columns] = pd.DataFrame([findHits(df['All Description'], drugTypeKeywords[drug]) for drug in drugTypeKeywords.columns]).T.set_index(df.index)
df["Drug Category"] = [findDrugType(df.iloc[ind][drugTypeKeywords.columns]) for ind in range(len(df))]

# v1 for #companies hit for each category, v2 for all companies counted once (includes "Multiple" category)
def graph_T3(df, graphType = 'line', version=1, moving_avg=True, countries=allCountries, omittedCategories = ['None']):
    # Filter (country and category), convert to wideform, calculate moving average if necessary
    df = df[findHits(df["Geographic Locations"], np.array(countries))]
    df = df[np.invert(findHits(df["Drug Category"], np.array(omittedCategories)))]
    if version==1:
        df_wideform = df.groupby(df['Year Founded']).sum()[drugTypeKeywords.columns]
    elif version==2:
        df_wideform = to_wideform(df, "Year Founded", "Drug Category")
    if moving_avg: df_wideform = calc_moving_avg(df_wideform, 4)
    
    # Ploting script
    caption = re.sub(r"[\(\[].*?[\)\]]", "", ("Countries: " + ", ".join(countries))).replace(" , ", ", ")
    fig, ax = plt.subplots()
    sns.set()
    if graphType == 'bar':
        df_wideform.plot(kind='bar', stacked=True, ax=ax)
    elif graphType == 'line':
        ax = sns.lineplot(data=df_wideform, dashes=False)
    plt.xlabel("Year Founded", **sfont, fontsize=14)
    plt.ylabel("Number of Companies ", **sfont, fontsize=14)
    plt.title(("Pharmaceutical Categories (v" + str(version) + ")"), **sfont, fontsize=16)
    ax.annotate(caption, (0,0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.show()

# Output
graph_T3(df, 'line', 1)
graph_T3(df, 'line', 2)
graph_T3(df, 'bar', 1, False)
graph_T3(df, 'bar', 2, False)
graph_T3(df, 'bar', 2, False, ["European Developed Markets (Primary)"])
graph_T3(df, 'bar', 2, False, ["United States of America (Primary)"])
graph_T3(df, 'line', 1, True, ["European Developed Markets (Primary)"])
graph_T3(df, 'line', 2, True, ["European Developed Markets (Primary)"])
graph_T3(df, 'line', 2, True, ["United States of America (Primary)"])

T3_table = to_wideform(df, "Year Founded", "Drug Category")
T3_table = T3_table[list(T3_table.columns.values[4:5]) + list(T3_table.columns.values[3:4]) + list(T3_table.columns.values[0:3]) + list(T3_table.columns.values[5:])].fillna(0)
print("Task 3 Table: Drug Categories")
print(T3_table)

# ----- Task 4: Clinical development stage -----
# clinDevKeywords dataframe can be edited here for better classification
clinDevKeywords = pd.concat(
        [pd.DataFrame({'preclinical_keywords': ["preclinical", "early development"]}),
         pd.DataFrame({'phase0_keywords': ["phase 0"]}),
         pd.DataFrame({'phase1_keywords': ["phase I", "phase 1", "phase 1a", "phase 1b", "phase Ia", "phase Ib", "phase one"]}),
         pd.DataFrame({'phase2_keywords': ["phase II", "phase 2", "phase 2a", "phase 2b", "phase IIa", "phase IIb", "phase two"]}),
         pd.DataFrame({'phase3_keywords': ["phase III", "phase 3", "phase 3a", "phase 3b", "phase three"]}),
         pd.DataFrame({'market_keywords': ["launched", "FDA approved", "FDA approval", "phase IV", "phase 4", "mature product", "completed phase 3", "completed phase three", "completed phase III"]})], 
         axis=1)
# Tried adding "clinical trials" to phase 1 but didnt give intended results...
# market_keywords: "commercializes", "markets" # gives too many false positives if included, but too many true positives left out if not included...

# Attempt to make single column and find latest phase. Theres probably a simpler way...
def findPhase(keywords):
    if keywords['market_keywords']: return "Market"
    elif keywords['phase3_keywords']: return "Phase III"
    elif keywords['phase2_keywords']: return "Phase II"
    elif keywords['phase1_keywords']: return "Phase I"
    elif keywords['phase0_keywords']: return "Phase 0"
    elif keywords['preclinical_keywords']: return "Preclinical"
    else: return "None"

df[clinDevKeywords.columns] = pd.DataFrame([findHits(df['All Description'], clinDevKeywords[phase]) for phase in clinDevKeywords.columns]).T.set_index(df.index)
df = df.reset_index(level=0, drop=True)
df["Most Developed Phase"] = [findPhase(df.iloc[ind][clinDevKeywords.columns]) for ind in range(len(df))]

def graph_T4(df, graphType = 'line', moving_avg=True, countries = allCountries, omittedPhases = ['None']):
    # Filter (phase and country), convert to wideform, reorder columns, convolve if necessary.
    df = df[np.invert(findHits(df["Most Developed Phase"], np.array(omittedPhases)))]
    df = df[findHits(df["Geographic Locations"], np.array(countries))]
    df_wideform = to_wideform(df, "Year Founded", "Most Developed Phase")
    df_wideform = df_wideform[list(df_wideform.columns.values[-1:]) + list(df_wideform.columns.values[0:-1])]
    if moving_avg: df_wideform = calc_moving_avg(df_wideform, 4)
    
    # Plotting script
    caption = re.sub(r"[\(\[].*?[\)\]]", "", ("Countries: " + ", ".join(countries))).replace(" , ", ", ")
    fig, ax = plt.subplots()
    sns.set()
    if graphType == 'bar':
        df_wideform.plot(kind='bar', stacked=True, ax=ax)
    elif graphType == 'line':
        ax = sns.lineplot(data=df_wideform, dashes=False)
    plt.xlabel("Year Founded", **sfont, fontsize=14)
    plt.ylabel("Number of Companies ", **sfont, fontsize=14)
    plt.title("Biotech Industry (Most Developed Phase)", **sfont, fontsize=16)
    ax.annotate(caption, (0,0), (0, -60), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.show()

# Output
graph_T4(df, 'line', True, allCountries)
graph_T4(df, 'line', True, ["European Developed Markets (Primary)"])
graph_T4(df, 'line', True, ["United States of America (Primary)"])
graph_T4(df, 'line', False, ["Canada (Primary)"])
graph_T4(df, 'bar', False)
graph_T4(df, 'bar', False, ["European Developed Markets (Primary)"])
graph_T4(df, 'bar', False, ["United States of America (Primary)"])
graph_T4(df, 'bar', False, ["Canada (Primary)"])
# Might want to show ratios of phases, possibly over time?

T4_table = to_wideform(df).fillna(0)
T4_table = T4_table[list(T4_table.columns.values[0:1]) + list(T4_table.columns.values[-1:]) + list(T4_table.columns.values[1:-1])]
print("Task 4 Table: Clinical Development Phases")
print(T4_table)

# ----- Task 5: Educational background -----
dfEd = dfEd[dfEd["In Original Dataset"]]
# Make columns of people, not companies
dfEd['Majors'] = dfEd['Majors'].str.replace('\); ', ')|')
# Remove significant anomalies that interfere with processing:
dfEd['Majors'] = dfEd['Majors'].str.replace('University of California - San Diego', 'University of California San Diego')
dfEd['Majors'] = dfEd['Majors'].str.replace('University of Pennsylvania - The Wharton School', 'University of Pennsylvania The Wharton School')
dfEd['Majors'] = dfEd['Majors'].str.replace('University of Wisconsin - Madison', 'University of Wisconsin Madison')
dfEd['Majors'] = dfEd['Majors'].str.replace('New York University - Leonard N. Stern School of Business', 'New York University Leonard N. Stern School of Business')
dfEd['Majors'] = dfEd['Majors'].str.split('|')
dfEd['Company ID'] = dfEd.index
dfEd_person = (dfEd['Majors'].apply(lambda x: pd.Series(x))
                        .stack()
                        .reset_index(level=1, drop=True)
                        .to_frame('Majors')
                        .join(dfEd[['Company ID', 'Company Name', 'Year Founded', 'Key Executives (Current and Prior)']], how='left')
)
dfEd_person['Company-Person ID'] = np.arange(0, np.size(dfEd_person, 0)) + 1

# Tidy up and organize: remove anything between brackets less than 14 characters long such as (Board) or (Prior Board), then split off names from their backgrounds
dfEd_person['Person'] = dfEd_person['Majors'].str.replace('\(.{,14}\)','').str.split('\(', 1, expand=True)[0]
dfEd_person['New Majors'] = dfEd_person['Majors'].str.replace('\(.{,14}\)','').str.split('\(', 1, expand=True)[1]
dfEd_person['New Majors'] = dfEd_person['New Majors'].str.rsplit(')',1,expand=True)[0]


dfEd_person['New Majors'] = dfEd_person['New Majors'].str.split('; ')
dfEd_person = dfEd_person.reset_index(level=0, drop=True)

# Filter for only entries where person with majors exists, then work out if a key exec
dfEd_person['Is Person'] = ['-' != s for s in dfEd_person['Person']]
dfEd_person = dfEd_person[dfEd_person['Is Person']]
dfEd_person = dfEd_person.reset_index(level=0, drop=True)
dfEd_person['Is Key Exec'] = [norm(dfEd_person['Person'][ind]) in norm(s) for ind, s in enumerate(dfEd_person['Key Executives (Current and Prior)'])]
dfEd_person = dfEd_person[dfEd_person['Is Key Exec']]
dfEd_person = dfEd_person.reset_index(level=0, drop=True)

# Filter out duplicates of people (for people that were key executives at multiple companies)
# Simple method: only match by normalized name, which is simpler and almost as effective:
# dfEd_person.drop_duplicates(subset='Person (normalized)', keep='first', inplace=True)
# The problems that can and probably do happen here are: 
#     1.) what happens when people share a common name?
#     2.) what about when a name is changed or spelt differently at different times
# I have attempted to deal with 1. but not with 2:
dfEd_person['Person (normalized)'] = [norm(person) for person in dfEd_person['Person']]
dfEd_person['Duplicate Name'] = dfEd_person.duplicated(subset='Person (normalized)', keep=False) # all repeated names marked as true
dfEd_person['Duplicate (not first)'] = dfEd_person.duplicated(subset='Person (normalized)', keep="first") # mark all dupes as true except for first occurance
dfEd_person['Duplicate (first)'] = dfEd_person['Duplicate Name'] & ~(dfEd_person['Duplicate (not first)'])
dfEd_person['New Majors (normalized)'] = [norm(' '.join(majors)) for majors in dfEd_person['New Majors']]
dfEd_person_ogs = dfEd_person[dfEd_person['Duplicate (first)']]
dfEd_person_copies = dfEd_person[dfEd_person['Duplicate (not first)']]
dfEd_person_copies['Majors Similarity'] = np.nan
dfEd_person_copies = dfEd_person_copies.reset_index(level=0, drop=True)

dfEd_person_copies['Majors Similarity'] = [difflib.SequenceMatcher(None, entry['New Majors (normalized)'], dfEd_person_ogs.loc[dfEd_person_ogs['Person (normalized)'] == entry['Person (normalized)']]['New Majors (normalized)'].values[0]).ratio() for index, entry in dfEd_person_copies.iterrows()]
dfEd_person_copies['Same Name Different Person'] = dfEd_person_copies['Majors Similarity'] < 0.45

dfEd_person = pd.concat([dfEd_person_ogs, dfEd_person_copies[dfEd_person_copies['Same Name Different Person']] ,dfEd_person[~dfEd_person['Duplicate Name']]], join="inner")

# Make columns of majors, not people
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
fig, ax = plt.subplots()
dfEd_major['Major'].value_counts().head(15).plot('bar')
plt.title("Biotech Industry Majors", **sfont, fontsize=20)
plt.show()

fig, ax = plt.subplots()
dfEd_major['University'].value_counts().head(15).plot('bar')
plt.title("Biotech Industry Universities", **sfont, fontsize=20)
plt.show()

T5_table = dfEd_major['Major'].value_counts()
print("Task 5 Table: Key Executive Majors")
print(T5_table.head(50))

# ----- Task 5: Educational background (Degrees not majors)-----
# This was a quick copy-paste replacing majors for degrees - can be done much more quickly, tidily, and efficiently..
#dfEd = dfEd[dfEd["In Original Dataset"]]
# Make columns of people, not companies
dfEd['Degrees'] = dfEd['Degrees'].str.replace('\); ', ')|')
# Remove significant anomalies that interfere with processing:
dfEd['Degrees'] = dfEd['Degrees'].str.replace('University of California - San Diego', 'University of California San Diego')
dfEd['Degrees'] = dfEd['Degrees'].str.replace('University of Pennsylvania - The Wharton School', 'University of Pennsylvania The Wharton School')
dfEd['Degrees'] = dfEd['Degrees'].str.replace('University of Wisconsin - Madison', 'University of Wisconsin Madison')
dfEd['Degrees'] = dfEd['Degrees'].str.replace('New York University - Leonard N. Stern School of Business', 'New York University Leonard N. Stern School of Business')
dfEd['Degrees'] = dfEd['Degrees'].str.split('|')
dfEd['Company ID'] = dfEd.index
dfEd_person = (dfEd['Degrees'].apply(lambda x: pd.Series(x))
                        .stack()
                        .reset_index(level=1, drop=True)
                        .to_frame('Degrees')
                        .join(dfEd[['Company ID', 'Company Name', 'Year Founded', 'Key Executives (Current and Prior)']], how='left')
)
dfEd_person['Company-Person ID'] = np.arange(0, np.size(dfEd_person, 0)) + 1

# Tidy up and organize: remove anything between brackets less than 14 characters long such as (Board) or (Prior Board), then split off names from their backgrounds
dfEd_person['Person'] = dfEd_person['Degrees'].str.replace('\(.{,14}\)','').str.split('\(', 1, expand=True)[0]
dfEd_person['New Degrees'] = dfEd_person['Degrees'].str.replace('\(.{,14}\)','').str.split('\(', 1, expand=True)[1]
dfEd_person['New Degrees'] = dfEd_person['New Degrees'].str.rsplit(')',1,expand=True)[0]


dfEd_person['New Degrees'] = dfEd_person['New Degrees'].str.split('; ')
dfEd_person = dfEd_person.reset_index(level=0, drop=True)

# Filter for only entries where person with majors exists, then work out if a key exec
dfEd_person['Is Person'] = ['-' != s for s in dfEd_person['Person']]
dfEd_person = dfEd_person[dfEd_person['Is Person']]
dfEd_person = dfEd_person.reset_index(level=0, drop=True)
dfEd_person['Is Key Exec'] = [norm(dfEd_person['Person'][ind]) in norm(s) for ind, s in enumerate(dfEd_person['Key Executives (Current and Prior)'])]
dfEd_person = dfEd_person[dfEd_person['Is Key Exec']]
dfEd_person = dfEd_person.reset_index(level=0, drop=True)

# Filter out duplicates of people (for people that were key executives at multiple companies)
# Simple method: only match by normalized name, which is simpler and almost as effective:
# dfEd_person.drop_duplicates(subset='Person (normalized)', keep='first', inplace=True)
# The problems that can and probably do happen here are: 
#     1.) what happens when people share a common name?
#     2.) what about when a name is changed or spelt differently at different times
# I have attempted to deal with 1. but not with 2:
dfEd_person['Person (normalized)'] = [norm(person) for person in dfEd_person['Person']]
dfEd_person['Duplicate Name'] = dfEd_person.duplicated(subset='Person (normalized)', keep=False) # all repeated names marked as true
dfEd_person['Duplicate (not first)'] = dfEd_person.duplicated(subset='Person (normalized)', keep="first") # mark all dupes as true except for first occurance
dfEd_person['Duplicate (first)'] = dfEd_person['Duplicate Name'] & ~(dfEd_person['Duplicate (not first)'])
dfEd_person['New Degrees (normalized)'] = [norm(' '.join(degrees)) for degrees in dfEd_person['New Degrees']]
dfEd_person_ogs = dfEd_person[dfEd_person['Duplicate (first)']]
dfEd_person_copies = dfEd_person[dfEd_person['Duplicate (not first)']]
dfEd_person_copies['Degrees Similarity'] = np.nan
dfEd_person_copies = dfEd_person_copies.reset_index(level=0, drop=True)

dfEd_person_copies['Degrees Similarity'] = [difflib.SequenceMatcher(None, entry['New Degrees (normalized)'], dfEd_person_ogs.loc[dfEd_person_ogs['Person (normalized)'] == entry['Person (normalized)']]['New Degrees (normalized)'].values[0]).ratio() for index, entry in dfEd_person_copies.iterrows()]
dfEd_person_copies['Same Name Different Person'] = dfEd_person_copies['Degrees Similarity'] < 0.45

dfEd_person = pd.concat([dfEd_person_ogs, dfEd_person_copies[dfEd_person_copies['Same Name Different Person']] ,dfEd_person[~dfEd_person['Duplicate Name']]], join="inner")

# Make columns of majors, not people
dfEd_degree = (dfEd_person['New Degrees'].apply(lambda x: pd.Series(x))
                        .stack()
                        .reset_index(level=1, drop=True)
                        .to_frame('New Degrees')
                        .join(dfEd_person[['Degrees', 'Company ID', 'Company Name', 'Year Founded', 'Company-Person ID', 'Person']], how='left')
)

# Split off subjects from universities
dfEd_degree['University'] = dfEd_degree['New Degrees'].str.split(' - ', expand=True)[0]
dfEd_degree['Degrees'] = dfEd_degree['New Degrees'].str.split(' - ', expand=True)[1]

# Plot Bar Charts
fig, ax = plt.subplots()
dfEd_degree['Degrees'].value_counts().head(15).plot('bar')
plt.title("Biotech Industry Degrees", **sfont, fontsize=20)
plt.show()

#fig, ax = plt.subplots()
#dfEd_major['University'].value_counts().head(15).plot('bar')
#plt.title("Biotech Industry Universities", **sfont, fontsize=20)
#plt.show()

T5_table = dfEd_degree['Degrees'].value_counts()
print("Task 5 Table: Key Executive Degrees")
print(T5_table.head(20))

df_output = df[['Company Name', 'Exchange:Ticker', 'Industry Classifications',
       'Geographic Locations', 'Year Founded',
       'Company Status', 'Exchanges (All Equity Listings)',
       'Total Revenue [LTM] ($USDmm, Historical rate)', 'Parent Company',
       'Long Business Description', 'Business Description',
       'Product Description', 'Product Name',
       'Company Type', 'Drug Category', 'Most Developed Phase']]

#Output writer
writer = pd.ExcelWriter('output.xlsx')
T1_table.to_excel(writer, 'T1')
T2_table.to_excel(writer, 'T2')
T3_table.to_excel(writer, 'T3')
T4_table.to_excel(writer, 'T4')
T5_table.to_excel(writer, 'T5')
df_output.to_excel(writer, 'Final (cleaned) dataframe')
df_DrugDevTech.to_excel(writer, 'Drug Dev&Tech Companies')
df_notDrugDevTech.to_excel(writer, 'Not Drug Dev&Tech Companies')
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

# Old Junk
#T1 = df['Year Founded'].groupby(df['Year Founded']).count().to_frame() #For output file
#T1.rename(columns={T1.columns[0]: "Number of Companies"}, inplace=True)

#T1.plot(kind="bar", color='#ffb81c') # Counts number per year
#sfont = {'fontname':'Arial'}
#plt.xlabel("Year Founded", **sfont, fontsize=14)
#plt.ylabel("Number of Companies ", **sfont, fontsize=14)
#plt.title("Biotech Industry", **sfont, fontsize=20)
#plt.tight_layout()

#for_plot_T2 = df[['Year Founded', 'Company Type']]
#for_plot_T2pr = for_plot_T2[for_plot_T2['Company Type'].str.contains('Pr')]
#for_plot_T2pu = for_plot_T2[for_plot_T2['Company Type'].str.contains('Pu')]
#
#T2 = pd.concat([for_plot_T2pr.groupby(for_plot_T2pr['Year Founded']).count(), 
#                for_plot_T2pu.groupby(for_plot_T2pu['Year Founded']).count()], axis=1)
#T2.columns = ["Private Companies", "Public Companies"]

# Plot for Task 2
#fig, ax = plt.subplots()
#T2.plot(kind='bar', stacked=True, ax=ax, color=['#ffb81c', '#544c41'])
#plt.xlabel("Year", **sfont, fontsize=14)
#plt.ylabel("Number of Companies ", **sfont, fontsize=14)
#plt.title("Biotech Industry", **sfont, fontsize=20)
#ax.legend(["Private Companies", "Public Companies"])
#plt.tight_layout()

#df_yearGrouped = df.groupby(df['Year Founded']).sum()
#T3 = df_yearGrouped[drugTypeKeywords.columns]
# Bar Graph for Task 3
#sns.set()
#fig, ax = plt.subplots()
#df_yearGrouped[drugTypeKeywords.columns].plot(kind='bar', stacked=True, ax=ax)
#plt.xlabel("Year Founded", **sfont, fontsize=14)
#plt.ylabel("Number of Companies ", **sfont, fontsize=14)
#plt.title("Biotech Industry", **sfont, fontsize=20)
#ax.legend(["mAbs", "RDNA", "Antisense", "Gene Therapy", "Chemicals"])
#plt.tight_layout()

# Line Graph for Task 3 (convolution is simply a moving average - for smoothing)
#N = 4
#T3_convolved = pd.DataFrame([np.convolve(df_yearGrouped[drugtype], np.ones((N,))/N, mode='same') for drugtype in drugTypeKeywords.columns]).T.set_index(df_yearGrouped.index)
#T3_convolved.columns = drugTypeKeywords.columns

#fig, ax = plt.subplots()
#T3_convolved.plot(kind='line', stacked=False, ax=ax)
#plt.xlabel("Year Founded", **sfont, fontsize=14)
#plt.ylabel("Number of Companies ", **sfont, fontsize=14)
#plt.title("Biotech Industry", **sfont, fontsize=20)
#ax.legend(["mAbs", "RDNA", "Antisense", "Gene Therapy", "Chemicals"])
#plt.tight_layout()
#sns.relplot(x="Year", y="Number of companies", hue=")

#T4_sns_wideform = pd.melt(df, id_vars="Year Founded", value_vars="Most Developed Phase", value_name="Most Developed Phase", var_name="Frequency").groupby(['Year Founded', 'Most Developed Phase'])['Most Developed Phase'].count().unstack('Most Developed Phase')

#T4_sns_wideform = to_wideform(df)
#T4_sns = pd.melt(df, id_vars="Year Founded", value_vars="Most Developed Phase", value_name="Most Developed Phase", var_name="Frequency").groupby(['Year Founded', 'Most Developed Phase'], as_index=False).count()
#T4_sns = T4_sns[~(T4_sns["Most Developed Phase"]=='None')]

# T4 Original Bar Chart
#fig, ax = plt.subplots()
#df_yearGrouped[clinDevKeywords.columns].plot(kind='bar', stacked=True, ax=ax)
#plt.xlabel("Year Founded", **sfont, fontsize=14)
#plt.ylabel("Number of Companies ", **sfont, fontsize=14)
#plt.title("Biotech Industry (Old plot - all mentioned phases)", **sfont, fontsize=20)
#plt.tight_layout()

#for index, entry in dfEd_person_copies.iterrows():
#    print("New person:")
#    print(entry['New Majors (normalized)'])
#    og_entry = dfEd_person_ogs.loc[dfEd_person_ogs['Person (normalized)'] == entry['Person (normalized)']]
#    print(og_entry['New Majors (normalized)'].values[0])
#    fuzzymatch = difflib.SequenceMatcher(None, entry['New Majors (normalized)'], og_entry['New Majors (normalized)'].values[0]).ratio()
#    print(fuzzymatch)
#    entry['Majors Similarity'] = fuzzymatch