# This script makes dictionaries for the report generation each year.

# Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pickle

# # ---------------------------------------------------------------
# # LOADING DATAFRAMES
# # ---------------------------------------------------------------

# Main PAT Maths results dataframe
data_maths = pd.read_excel("C:/Users/Debbie.Chong/Downloads/pat-maths-4th-edition-test-10-19122025-075235.xlsx")
# PAT Reading results
data_reading = pd.read_excel("C:/Users/Debbie.Chong/Downloads/pat-reading-5th-edition-test-10-19122025-075129.xlsx")

# Master list of students (please update this yearly)
class_list = pd.read_csv("2025_StudentClassList.csv") #2025 class list as an example

# Question substrands (also update yearly and for different year levels)
maths_substrands = pd.read_csv("PAT_Maths_Year10_Substrands.csv")

# List of students and their grades
student_CG = pd.read_csv("Student_CG.csv")


# # ---------------------------------------------------------------
# # FORMATTING DATAFRAMES
# # ---------------------------------------------------------------

# # PAT Spelling
# data_spelling = pd.read_excel()

# # Setting question columns
# q_end_col_spelling = 

# q_strand_spelling = data_spelling[2:6,np.r_[0,13:q_end_col_spelling]]
# students_spelling = data_spelling.drop(data_spelling.index[0:10])

# # Getting strands
# strand_end_row_spelling = 
# strand_start_col_spelling = 
# strand_end_col_spelling = 

# strand_key_spelling = data_spelling.iloc[2:strand_end_row_spelling,np.r_[strand_start_col_spelling:strand_end_col_spelling]]

# # Resetting question columns
# student_q_col_end_spelling = 53
# q_strands_spelling_endcol = 41

# students_spelling.iloc[0,13:student_q_col_end_spelling] = q_strands_spelling.iloc[3,1:q_strands_spelling_endcol] # setting to question number
# # Changing column names of student dataset
# students_spelling.columns = students_spelling.iloc[0] # getting first row
# students_spelling.drop(students_spelling.index[0], inplace=True) # dropping first row
# students_spelling.reset_index(drop = True, inplace=True)


# Setting variables for cleaning and formatting
# Question columns
q_end_col_maths = 53
q_end_col_reading = 48
# Applying
# Maths
q_strands_maths = data_maths.iloc[2:6,np.r_[0,13:q_end_col_maths]]
students_maths = data_maths.drop(data_maths.index[0:10])
# Reading
q_strands_reading = data_reading.iloc[2:6,np.r_[0,13:q_end_col_reading]]
students_reading = data_reading.drop(data_reading.index[0:10])

# Making strand dataframes
# Strand variables
strand_end_row_maths = 8
strand_end_row_reading = 6
strand_start_col_maths = 54
strand_end_col_maths = 56
strand_start_col_reading = 49
strand_end_col_reading = 51

# Make small dataframe of strands
# Maths
strand_key_maths = data_maths.iloc[2:strand_end_row_maths,np.r_[strand_start_col_maths:strand_end_col_maths]]
# Reading
strand_key_reading = data_reading.iloc[2:strand_end_row_reading,np.r_[strand_start_col_reading:strand_end_col_reading]]


# # ---------------------------------------------------------------
# # RESET QUESTION COLUMNS
# # ---------------------------------------------------------------

# Assigning variables
student_q_col_end_maths = 53
student_q_col_end_reading = 48
q_strands_maths_endcol = 41
q_strands_reading_endcol = 36


# Maths
students_maths.iloc[0,13:student_q_col_end_maths] = q_strands_maths.iloc[3,1:q_strands_maths_endcol] # setting to question number
# Changing column names of student dataset
students_maths.columns = students_maths.iloc[0] # getting first row
students_maths.drop(students_maths.index[0], inplace=True) # dropping first row
students_maths.reset_index(drop = True, inplace=True)

# Reading
students_reading.iloc[0,13:student_q_col_end_reading] = q_strands_reading.iloc[3,1:q_strands_reading_endcol]
# Changing column names
students_reading.columns = students_reading.iloc[0] # getting first row
students_reading.drop(students_reading.index[0], inplace=True) # dropping first row
students_reading.reset_index(drop = True, inplace=True)


# # ---------------------------------------------------------------
# # MATCH TO SYNERGETIC INFORMATION
# # ---------------------------------------------------------------
# Cleaning columns
def clean_df(df):

    # Making a copy
    df = df.copy()

    # Concatenating name columns
    df['Full Name'] = df[['Given name', 'Middle names', 'Family name']].fillna('').agg(
                lambda x: ' '.join(filter(None, x)), axis=1
                )
    
    # Removing old name columns & other columns as required
    df.drop(columns = ['Given name', 'Middle names', 'Family name', 'Unique ID', 'Inactive tags', 'Stanine',
                       'Year level (at time of test)', 'Tags (at time of test)'], inplace = True)
    
    # Moving 'Full Name' column to the front
    df.insert(0,"Full Name", df.pop('Full Name'))

    return df

# Applying
students_maths_clean = clean_df(students_maths)
students_reading_clean = clean_df(students_reading)
#students_spelling_clean = clean_df(students_spelling)


# # ---------------------------------------------------------------
# # CHECK STUDENT ID
# # ---------------------------------------------------------------

# Checking that student ID is the same as in Synergetic (we merge based on studnt name and year level)
def match_student_ids(df1, df2, name_col_df1='Full Name', name_col_df2='StudentNameExternal',
                      year_col_df1='Year level (current)', year_col_df2='StudentYearLevel',
                      id_col_name='ID', insert_pos=2):
    """
    Merges student IDs from df2 into df1 based on both the student's first and last name and the DOB.
    Returns the first dataframe with an aditional ID column (based on Synergetic data).
    """
    
    # Preparing first dataframe
    df1 = df1.copy()
    df1['year_clean'] = df1[year_col_df1].astype(str).str.extract(r'(\d+)').astype(int) # Getting year level only (stripping characters)
    # Getting all parts of the name
    df1['name_parts'] = df1[name_col_df1].str.strip().str.split()
    df1['first_name']  = df1['name_parts'].str[0].str.lower() # Strips first whitespace to get first name
    df1['middle_name'] = df1['name_parts'].apply(
        lambda x: " ".join(x[1:-1]).lower() if len(x) > 2 else "" # Joining the middle two sub-strings to get the middle name, else return blank string
    )
    df1['last_name']   = df1['name_parts'].str[-1].str.lower() # strips last whitespace to get last name

    # Preparing second dataframe
    df2 = df2.copy()
    # df2['name_original'] = df2[name_col_df2], # Keeping original names for class matching
    df2['name_parts'] = df2[name_col_df2].str.strip().str.split() # Getting all parts of the name
    df2['first_name_synergetic']  = df2['name_parts'].str[0].str.lower()
    df2['middle_name_synergetic'] = df2['name_parts'].apply(
        lambda x: " ".join(x[1:-1]).lower() if len(x) > 2 else ""
    )
    df2['last_name_synergetic'] = df2['name_parts'].str[-1].str.lower()

    # Keeping clean lookup dataframe
    df2_lookup = df2[[name_col_df2, 'first_name_synergetic', 'middle_name_synergetic', 
                     'last_name_synergetic', year_col_df2, id_col_name, 'ClassCampus']
    ].drop_duplicates()
    
    # Merging dataframes
    merged_df = df1.merge(
        df2_lookup,
        left_on = ['first_name', 'last_name', 'year_clean'],
        right_on = ['first_name_synergetic', 'last_name_synergetic', year_col_df2],
        how = 'left'
    )

    # If students have both first and last names, use the middle names for better match
    mask = merged_df.groupby(['first_name', 'last_name', 'year_clean'])[id_col_name].transform('count') > 1

    if mask.any():
        duplicate_df = merged_df[mask].copy()
        # Merging on middle names for duplicate students
        dup_merge = duplicate_df.merge(
            df2_lookup,
            left_on=['first_name', 'middle_name', 'last_name', 'year_clean'],
            right_on=['first_name_synergetic', 'middle_name_synergetic', 'last_name_synergetic', year_col_df2],
            how='left',
            suffixes=('', '_final')
        )
        # Replace ID and Full Name where we got a match using middle name
        for col in [id_col_name, name_col_df2]:
            merged_df.loc[mask, col] = dup_merge[col + '_final'].values
    
    # Adding Synergetic name list
    merged_df['Full Name (Synergetic)'] = merged_df[name_col_df2]

    # Dropping unnecessary columns
    merged_df.drop(
        columns=[
            'name_parts', 'year_clean',
            'first_name', 'middle_name', 'last_name',
            'first_name_synergetic', 'middle_name_synergetic', 'last_name_synergetic',
            name_col_df2, year_col_df2,
        ],
        errors='ignore',
        inplace=True,
    )

    # Cleaning and positioning ID column
    merged_df[id_col_name] = merged_df[id_col_name].astype('Int64').astype('string')
    id_series = merged_df.pop(id_col_name)
    merged_df.insert(insert_pos, id_col_name, id_series)

    # Move Synergetic name column
    synergetic_name = merged_df.pop('Full Name (Synergetic)')
    merged_df.insert(1, 'Full Name (Synergetic)', synergetic_name)

    # Change 'ECG' to 'EMC'
    merged_df['ClassCampus'] = merged_df['ClassCampus'].str.replace('EGC', 'EMC')

    # Moving Campus column
    campus_col = merged_df.pop('ClassCampus')
    merged_df.insert(8, 'Campus', campus_col)

    return merged_df

students_maths_match = match_student_ids(students_maths_clean, class_list)
students_reading_match = match_student_ids(students_reading_clean, class_list)
#students_spelling_match = match_student_ids(students_spelling_clean, class_list)


# # ---------------------------------------------------------------
# # CLEAN QUESTIONS DATAFRAME
# # ---------------------------------------------------------------
# Maths
q_strands_maths = (
    q_strands_maths
    .set_index(q_strands_maths.columns[0]) # set first column as row labels
    .T # transpose
    .reset_index(drop=True) # clean up the old index
    .rename_axis(None, axis=1) # removing name from index
    .reindex(columns=['Question number','Strand','Question difficulty','Percentage correct'])
)
# Reading
q_strands_reading = (
    q_strands_reading
    .set_index(q_strands_reading.columns[0]) # set first column as row labels
    .T # transpose
    .reset_index(drop=True) # clean up the old index
    .rename_axis(None, axis=1) # removing name from index
    .reindex(columns=['Question number','Strand','Question difficulty','Percentage correct'])
)
# # Spelling
# q_strands_spelling = (
#     q_strands_spelling
#     .set_index(q_strands_spelling.columns[0]) # set first column as row labels
#     .T # transpose
#     .reset_index(drop=True) # clean up the old index
#     .rename_axis(None, axis=1) # removing name from index
#     .reindex(columns=['Question number','Strand','Question difficulty','Percentage correct'])    
# )


# # ---------------------------------------------------------------
# # CLEAN STRAND KEY COLUMNS
# # ---------------------------------------------------------------

# Maths
strand_key_maths = strand_key_maths.rename(columns={strand_key_maths.columns[0]:'Key',
                                        strand_key_maths.columns[1]:'Name'})
# Convert the single letter 'Strand' code to full name for ease of reading
strand_map_maths = dict(zip(strand_key_maths['Key'], strand_key_maths['Name'])) # Convert to dictionary for mapping
q_strands_maths['Strand'] = q_strands_maths['Strand'].map(strand_map_maths) # Mapping

# Reading
strand_key_reading = strand_key_reading.rename(columns={strand_key_reading.columns[0]:'Key', strand_key_reading.columns[1]:'Name'})
# Convert the single letter 'Strand' code to full name for ease of reading
strand_map_reading = dict(zip(strand_key_reading['Key'], strand_key_reading['Name'])) # Convert to dictionary for mapping
q_strands_reading['Strand'] = q_strands_reading['Strand'].map(strand_map_reading) # Mapping

# # Spelling
# strand_key_spelling = strand_key_spelling.rename(columns={strand_key_spelling.columns[0]:'Key',
#                                         strand_key_spelling.columns[1]:'Name'})
# # Convert the single letter 'Strand' code to full name for ease of reading
# strand_map_spelling = dict(zip(strand_key_spelling['Key'], strand_key_spelling['Name'])) # Convert to dictionary for mapping
# q_strands_spelling['Strand'] = q_strands_spelling['Strand'].map(strand_map_spelling) # Mapping


# # ---------------------------------------------------------------
# # MATCH STUDENT TO CLASS
# # ---------------------------------------------------------------

# Function
def match_student_class(names_dataframe, class_dataframe):
    
    # Names column
    # Make copy
    dataframe_copy = names_dataframe.copy()
    # Cleaning names columns
    dataframe_copy['name_clean'] = dataframe_copy['Full Name (Synergetic)'].str.strip().str.lower()
    # Convert ID to string
    dataframe_copy['ID'] = dataframe_copy['ID'].astype(str)

    # Class names
    class_dataframe_copy = class_dataframe.copy()
    class_dataframe_copy['name_clean'] = class_dataframe_copy['StudentNameExternal'].str.strip().str.lower()
    # Convert ID to string
    class_dataframe_copy['ID'] = class_dataframe_copy['ID'].astype(str)

    # Filter out students with missing IDs before merging
    class_dataframe_copy = class_dataframe_copy[~class_dataframe_copy['ID'].isna()]

    # Merge the dataframe
    merged_df = class_dataframe_copy.merge(
        dataframe_copy,
        on=['name_clean', 'ID'],
        how='left',
        suffixes=('_class', '_student') 
    )

    # Filter out classes (where 'Score' is NaN --> no test data)
    merged_df = merged_df[~merged_df['Score'].isna()]

    # Grouping by class code (as dictionary)
    classes = dict(tuple(merged_df.groupby('ClassCode')))
    # To access: classes['K10-HPE4-2']

    return classes

class_reading_all = match_student_class(students_reading_match, class_list)
class_maths_all = match_student_class(students_maths_match, class_list)
#class_spelling_all = match_student_class(student_spelling_match, class_list)

# Keep classes who are in both dictionaries (for now)
# Get common keys
common_classes = set(class_maths_all.keys()) & set(class_reading_all.keys())
# Filter for the common classes
class_maths = {key: class_maths_all[key] for key in common_classes}
class_reading = {key: class_reading_all[key] for key in common_classes}
# Print uncommon classes
uncommon_keys = class_maths_all.keys() ^ class_reading_all.keys()
print(f"Uncommon keys: {uncommon_keys}")
# Output(19-12-2025): Uncommon keys: {'K1112-VCDUX-5', 'K1011-MATMU12-3', 'K1112-MATMUX-3', 'I10-ENG4-3', 'K1011-VCDU12-5'}

# Printing number of classes
print(f"Number of classes: {len(class_maths)}")
print(f"Class codes: {list(class_maths.keys())}")


# Take first 2 classes only (for testing)
# test_class_maths = dict(list(class_maths.items())[:2])
# test_class_reading = dict(list(class_reading.items())[:2])

# test_classes = ['I10S2-SCI4-5', 'S05-STEM-A']
# test_class_maths = {k: v for k, v in class_maths.items() if k in test_classes}
# test_class_reading = {k: v for k, v in class_reading.items() if k in test_classes}

# Save to file
# Dictionaries
with open("class_testscores.pkl", "wb") as f:
    pickle.dump(class_maths, f)
    pickle.dump(class_reading, f)
    #pickle.dump(class_spelling, f)

# Variables
# Making dictionary
var_dict = {
    'q_strands_maths': q_strands_maths,
    'q_strands_reading': q_strands_reading,
    #'q_strands_spelling': q_strands_spelling,
    'students_maths_match': students_maths_match,
    'students_reading_match': students_reading_match
    #'students_spelling_match': students_spelling_match
}
# Saving
var_dict_path = 'Variables.pkl'
with open(var_dict_path, 'wb') as f:
    pickle.dump(var_dict, f)




## Load dictionaries at runtime
# with open("class_maths.pkl", "rb") as f:
#     class_maths = pickle.load(f)
# with open("class_reading.pkl", "rb") as f:
#     class_reading = pickle.load(f)
# with open("q_strands_reading.pkl", "rb") as f:
#     q_strands_reading = pickle.load(f)
# with open("students_maths_match.pkl", "rb") as f:
#     students_maths_match = pickle.load(f)
# with open("students_reading_match.pkl", "rb") as f:
#     students_reading_match = pickle.load(f)
