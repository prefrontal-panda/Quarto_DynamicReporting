# This script makes dictionaries for the report generation each year.

# Loading libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pickle
from rapidfuzz import process, fuzz

# # ---------------------------------------------------------------
# # LOADING DATAFRAMES
# # ---------------------------------------------------------------

# PAT Maths results
data_maths_y7 = pd.read_excel("PAT_ReportGen_Files/PAT_Maths_Year7.xlsx")
data_maths_y8 = pd.read_excel("PAT_ReportGen_Files/PAT_Maths_Year8.xlsx")
data_maths_y9 = pd.read_excel("PAT_ReportGen_Files/PAT_Maths_Year9.xlsx")
data_maths_y10 = pd.read_excel("PAT_ReportGen_Files/PAT_Maths_Year10.xlsx")

# PAT Reading results
data_reading_y7 = pd.read_excel("PAT_ReportGen_Files/PAT_Reading_Year7.xlsx")
data_reading_y8 = pd.read_excel("PAT_ReportGen_Files/PAT_Reading_Year8.xlsx")
data_reading_y9 = pd.read_excel("PAT_ReportGen_Files/PAT_Reading_Year9.xlsx")
data_reading_y10 = pd.read_excel("PAT_ReportGen_Files/PAT_Reading_Year10.xlsx")

# Question substrands (also update yearly and for different year levels)
maths_substrands_y7 = pd.read_csv("PAT_ReportGen_Files/PAT_Maths_Year7_Substrands.csv")
maths_substrands_y8 = pd.read_csv("PAT_ReportGen_Files/PAT_Maths_Year8_Substrands.csv")
maths_substrands_y9 = pd.read_csv("PAT_ReportGen_Files/PAT_Maths_Year9_Substrands.csv")
maths_substrands_y10 = pd.read_csv("PAT_ReportGen_Files/PAT_Maths_Year10_Substrands.csv")

# Master list of students (please update this yearly)
class_list = pd.read_csv("2025_StudentClassList.csv") #2025 class list as an example

# List of students and their grades
student_CG = pd.read_csv("Student_CG.csv")


# # ---------------------------------------------------------------
# # FORMATTING DATAFRAMES
# # ---------------------------------------------------------------

# NOTE: COULD TRY TO USE THESE IN A DICTIONARY/LIST FORMAT

# Making initial dataframes
# Maths question strands
q_strands_maths_y7 = data_maths_y7.iloc[2:6,np.r_[0,13:53]]
q_strands_maths_y8 = data_maths_y8.iloc[2:6,np.r_[0,13:53]]
q_strands_maths_y9 = data_maths_y9.iloc[2:6,np.r_[0,13:53]]
q_strands_maths_y10 = data_maths_y10.iloc[2:6,np.r_[0,13:53]]

# Maths dataframe
students_maths_y7 = data_maths_y7.drop(data_maths_y7.index[0:10])
students_maths_y8 = data_maths_y8.drop(data_maths_y8.index[0:10])
students_maths_y9 = data_maths_y9.drop(data_maths_y9.index[0:10])
students_maths_y10 = data_maths_y10.drop(data_maths_y10.index[0:10])

# Reading question strands
q_strands_reading_y7 = data_reading_y7.iloc[2:6,np.r_[0,13:48]]
q_strands_reading_y8 = data_reading_y8.iloc[2:6,np.r_[0,13:47]]
q_strands_reading_y9 = data_reading_y9.iloc[2:6,np.r_[0,13:48]]
q_strands_reading_y10 = data_reading_y10.iloc[2:6,np.r_[0,13:48]]

# Reading dataframe
students_reading_y7 = data_reading_y7.drop(data_reading_y7.index[0:10])
students_reading_y8 = data_reading_y8.drop(data_reading_y8.index[0:10])
students_reading_y9 = data_reading_y9.drop(data_reading_y9.index[0:10])
students_reading_y10 = data_reading_y10.drop(data_reading_y10.index[0:10])

# Forming strand dataframes
# Maths
strand_key_maths_y7 = data_maths_y7.iloc[2:8,np.r_[54:56]]
strand_key_maths_y8 = data_maths_y8.iloc[2:8,np.r_[54:56]]
strand_key_maths_y9 = data_maths_y9.iloc[2:8,np.r_[54:56]]
strand_key_maths_y10 = data_maths_y10.iloc[2:8,np.r_[54:56]]

# Reading
strand_key_reading_y7 = data_reading_y7.iloc[2:6,np.r_[49:51]]
strand_key_reading_y8 = data_reading_y8.iloc[2:6,np.r_[48:50]]
strand_key_reading_y9 = data_reading_y9.iloc[2:6,np.r_[49:51]]
strand_key_reading_y10 = data_reading_y10.iloc[2:6,np.r_[49:51]]

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


# # ---------------------------------------------------------------
# # RESET QUESTION COLUMNS
# # ---------------------------------------------------------------

# Set to question number
# Maths
students_maths_y7.iloc[0,13:53] = q_strands_maths_y7.iloc[3,1:41]
students_maths_y8.iloc[0,13:53] = q_strands_maths_y8.iloc[3,1:41]
students_maths_y9.iloc[0,13:53] = q_strands_maths_y9.iloc[3,1:41]
students_maths_y10.iloc[0,13:53] = q_strands_maths_y10.iloc[3,1:41]

# Reading
students_reading_y7.iloc[0,13:48] = q_strands_reading_y7.iloc[3,1:36]
students_reading_y8.iloc[0,13:47] = q_strands_reading_y8.iloc[3,1:35]
students_reading_y9.iloc[0,13:48] = q_strands_reading_y9.iloc[3,1:36]
students_reading_y10.iloc[0,13:48] = q_strands_reading_y10.iloc[3,1:36]

# Changing column names
# Function
def change_q_col_names(dataframe, row_to_change = 0):
    dataframe.columns = dataframe.iloc[row_to_change] # getting first row
    dataframe.drop(dataframe.index[row_to_change], inplace = True) # dropping first row
    dataframe.reset_index(drop = True, inplace = True)

    #return dataframe (use this for no in-place change and call it like new_var = func(old_var))
# Applying
# Maths
change_q_col_names(students_maths_y7, row_to_change= 0) # Modifying in-place
change_q_col_names(students_maths_y8, row_to_change= 0)
change_q_col_names(students_maths_y9, row_to_change= 0)
change_q_col_names(students_maths_y10, row_to_change= 0)
# Reading
change_q_col_names(students_reading_y7, row_to_change= 0)
change_q_col_names(students_reading_y8, row_to_change= 0)
change_q_col_names(students_reading_y9, row_to_change= 0)
change_q_col_names(students_reading_y10, row_to_change= 0)


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
# Maths
students_maths_clean_y7 = clean_df(students_maths_y7)
students_maths_clean_y8 = clean_df(students_maths_y8)
students_maths_clean_y9 = clean_df(students_maths_y9)
students_maths_clean_y10 = clean_df(students_maths_y10)

# Reading
students_reading_clean_y7 = clean_df(students_reading_y7)
students_reading_clean_y8 = clean_df(students_reading_y8)
students_reading_clean_y9 = clean_df(students_reading_y9)
students_reading_clean_y10 = clean_df(students_reading_y10)


# # ---------------------------------------------------------------
# # CHECK STUDENT ID
# # ---------------------------------------------------------------

# Checking that student ID is the same as in Synergetic (we merge based on studnt name and year level)
def match_student_ids(df_students, df_master,
                      student_name_col='Full Name',
                      master_name_col='StudentNameExternal',
                      year_col_students='Year level (current)',
                      year_col_master='StudentYearLevel',
                      id_col_master='ID',
                      fuzzy_threshold=85,
                      insert_pos=2):
    
    """
    Merges student IDs from the masterlist to the student dataframe based on both the student's first and last name and the DOB.
    Returns the first dataframe with an aditional ID column (based on Synergetic data).
    """

    df = df_students.copy()
    master = df_master.copy()

    # --- Clean year columns ---
    df['year_clean'] = df[year_col_students].astype(str).str.extract(r'(\d+)').astype(int)
    master[year_col_master] = master[year_col_master].astype(int)

    # --- Split names ---
    df['first'] = df[student_name_col].str.split().str[0].str.lower()
    df['middle'] = df[student_name_col].str.split().apply(lambda x: " ".join(x[1:-1]).lower() if len(x) > 2 else "")
    df['last'] = df[student_name_col].str.split().str[-1].str.lower()

    master['first'] = master[master_name_col].str.split().str[0].str.lower()
    master['middle'] = master[master_name_col].str.split().apply(lambda x: " ".join(x[1:-1]).lower() if len(x) > 2 else "")
    master['last'] = master[master_name_col].str.split().str[-1].str.lower()

    # ------------------------------------------------------------------
    # Collapse masterlist to ONE row per name+year
    # ------------------------------------------------------------------
    master_unique = (
        master
        .groupby(['first', 'last', year_col_master])
        .agg({
            id_col_master: lambda x: x.iloc[0] if x.nunique() == 1 else pd.NA,
            master_name_col: 'first',
            'ClassCampus': lambda x: x.iloc[0] if x.nunique() == 1 else pd.NA,
        })
        .reset_index()
    )

    # --- Safe merge (cannot duplicate rows anymore) ---
    merged_df = df.merge(
        master_unique,
        left_on=['first', 'last', 'year_clean'],
        right_on=['first', 'last', year_col_master],
        how='left'
    )

    # --- Fuzzy match only where ID is still missing ---
    unmatched = merged_df[id_col_master].isna()

    for idx, row in merged_df[unmatched].iterrows():
        candidates = master[
            (master[year_col_master] == row['year_clean']) &
            (master['last'] == row['last'])
        ]

        if candidates.empty:
            continue

        names = candidates[master_name_col].tolist()

        first_last = f"{row['first']} {row['last']}"
        first_middle = f"{row['first']} {row['middle']} {row['last']}".strip()

        match = process.extractOne(first_last, names, scorer=fuzz.token_sort_ratio)
        if not match or match[1] < fuzzy_threshold:
            match = process.extractOne(first_middle, names, scorer=fuzz.partial_token_sort_ratio)

        if match and match[1] >= fuzzy_threshold:
            matched_row = candidates[candidates[master_name_col] == match[0]].iloc[0]
            merged_df.at[idx, id_col_master] = matched_row[id_col_master]
            merged_df.at[idx, master_name_col] = matched_row[master_name_col]
            merged_df.at[idx, 'ClassCampus'] = matched_row['ClassCampus']

    # --- Final columns ---
    merged_df['Full Name (Synergetic)'] = merged_df[master_name_col]

    merged_df.drop(
        columns=['first', 'middle', 'last', 'year_clean', year_col_master, master_name_col],
        errors='ignore',
        inplace=True
    )

    merged_df[id_col_master] = merged_df[id_col_master].astype('Int64').astype('string')
    id_series = merged_df.pop(id_col_master)
    merged_df.insert(insert_pos, id_col_master, id_series)

    synergetic_name = merged_df.pop('Full Name (Synergetic)')
    merged_df.insert(1, 'Full Name (Synergetic)', synergetic_name)

    # Renaming
    merged_df = merged_df.rename(columns={'ClassCampus': 'Campus'})

    return merged_df

# Maths
students_maths_match_y7 = match_student_ids(students_maths_clean_y7, class_list)
students_maths_match_y8 = match_student_ids(students_maths_clean_y8, class_list)
students_maths_match_y9 = match_student_ids(students_maths_clean_y9, class_list)
students_maths_match_y10 = match_student_ids(students_maths_clean_y10, class_list)

# Reading
students_reading_match_y7 = match_student_ids(students_reading_clean_y7, class_list)
students_reading_match_y8 = match_student_ids(students_reading_clean_y8, class_list)
students_reading_match_y9 = match_student_ids(students_reading_clean_y9, class_list)
students_reading_match_y10 = match_student_ids(students_reading_clean_y10, class_list)


# # ------------------------------------------------------------------
# # CHECK FOR ANY STUDENTS WITH SAME NAME + YEAR LEVEL + CAMPUS COMBO
# # -------------------------------------------------------------------

# Function
def students_with_duplicate_names(df,
                                  name_col='Full Name',
                                  year_col='Year level (current)'):
    temp = df.copy()

    # Extract year number
    temp['year_clean'] = temp[year_col].astype(str).str.extract(r'(\d+)').astype(int)

    # First + last only
    temp['first'] = temp[name_col].str.split().str[0].str.lower()
    temp['last'] = temp[name_col].str.split().str[-1].str.lower()

    # Count duplicates
    dup_mask = temp.duplicated(
        subset=['first', 'last', 'year_clean'],
        keep=False
    )

    result = temp.loc[dup_mask].sort_values(
        by=['year_clean', 'last', 'first']
    )

    # Clean up for display
    return result.drop(columns=['first', 'last', 'year_clean'])

# Get and print duplicates
# Get
duplicates_y7 = students_with_duplicate_names(students_maths_match_y7)
duplicates_y8 = students_with_duplicate_names(students_maths_match_y8)
duplicates_y9 = students_with_duplicate_names(students_maths_match_y9)
duplicates_y10 = students_with_duplicate_names(students_maths_match_y10)
# Print
print("Year 7:")
print(duplicates_y7[['Full Name', 'Username', 'ID', 'DOB', 'Active tags']])
print("Year 8:")
print(duplicates_y8[['Full Name', 'Username', 'ID', 'DOB', 'Active tags']])
print("Year 9:")
print(duplicates_y9[['Full Name', 'Username', 'ID', 'DOB', 'Active tags']])
print("Year 10:")
print(duplicates_y10[['Full Name', 'Username', 'ID', 'DOB', 'Active tags']])

# Manual overrides
manual_overrides = pd.DataFrame([
    {'Full Name': '', 'DOB': '', 'Correct ID': ''},
    {'Full Name': '', 'DOB': '', 'Correct ID': ''},
    {'Full Name': '', 'DOB': '', 'Correct ID': ''}
])

# Apply manual override
students_maths_match_y7 = students_maths_match_y7.merge(
    manual_overrides,
    on=['Full Name', 'DOB'],
    how='left'
)
students_reading_match_y7 = students_maths_match_y7.merge(
    manual_overrides,
    on=['Full Name', 'DOB'],
    how='left'
)

# Get the correct ID
# Year 7
# Maths
students_maths_match_y7['ID'] = students_maths_match_y7['Correct ID'].combine_first(students_maths_match_y7['ID'])
students_maths_match_y7.drop(columns='Correct ID', inplace=True)
# Reading
students_reading_match_y7['ID'] = students_reading_match_y7['Correct ID'].combine_first(students_reading_match_y7['ID'])
students_reading_match_y7.drop(columns='Correct ID', inplace=True)

# Year 9
# Maths
students_maths_match_y9 = students_maths_match_y9.merge(
    manual_overrides,
    on=['Full Name', 'DOB'],
    how='left'
)
students_maths_match_y9['ID'] = students_maths_match_y9['Correct ID'].combine_first(students_maths_match_y9['ID'])
students_maths_match_y9.drop(columns='Correct ID', inplace=True)
# Reading
students_reading_match_y9 = students_reading_match_y9.merge(
    manual_overrides,
    on=['Full Name', 'DOB'],
    how='left'
)
students_reading_match_y9['ID'] = students_reading_match_y9['Correct ID'].combine_first(students_reading_match_y9['ID'])
students_reading_match_y9.drop(columns='Correct ID', inplace=True)

# # ---------------------------------------------------------------
# # CLEAN QUESTIONS DATAFRAME
# # ---------------------------------------------------------------
# Function
def clean_q_df(dataframe):
    return(
        dataframe
        .set_index(dataframe.columns[0]) # set first column as row labels
        .T # transpose
        .reset_index(drop=True) # clean up the old index
        .rename_axis(None, axis=1) # removing name from index
        .reindex(columns=[
            'Question number',
            'Strand',
            'Question difficulty',
            'Percentage correct'
        ])
    )

# Applying
# Maths
q_strands_maths_y7 = clean_q_df(q_strands_maths_y7)
q_strands_maths_y8 = clean_q_df(q_strands_maths_y8)
q_strands_maths_y9 = clean_q_df(q_strands_maths_y9)
q_strands_maths_y10 = clean_q_df(q_strands_maths_y10)
# Reading
q_strands_reading_y7 = clean_q_df(q_strands_reading_y7)
q_strands_reading_y8 = clean_q_df(q_strands_reading_y8)
q_strands_reading_y9 = clean_q_df(q_strands_reading_y9)
q_strands_reading_y10 = clean_q_df(q_strands_reading_y10)


# # ---------------------------------------------------------------
# # CLEAN STRAND KEY COLUMNS
# # ---------------------------------------------------------------

# Function
def map_strand_names(strand_key_df, q_strands_df, strand_col='Strand'):
    # Standardise key dataframe column names
    strand_key_df.rename(
        columns = {
            strand_key_df.columns[0]: 'Key',
            strand_key_df.columns[1]: 'Name'
        },
        inplace = True
    )

    # Creating mapping dictionary
    strand_map = dict(zip(strand_key_df['Key'], strand_key_df['Name']))

    # Applying mapping
    q_strands_df[strand_col] = q_strands_df[strand_col].map(strand_map)

    return q_strands_df

# Applying
# Maths
q_strands_maths_y7 = map_strand_names(strand_key_maths_y7, q_strands_maths_y7)
map_strand_names(strand_key_maths_y8, q_strands_maths_y8)
map_strand_names(strand_key_maths_y9, q_strands_maths_y9)
map_strand_names(strand_key_maths_y10, q_strands_maths_y10)
# Reading
map_strand_names(strand_key_reading_y7, q_strands_reading_y7)
map_strand_names(strand_key_reading_y8, q_strands_reading_y8)
map_strand_names(strand_key_reading_y9, q_strands_reading_y9)
map_strand_names(strand_key_reading_y10, q_strands_reading_y10)

# Turning in to dictionaries
# Maths
year_level_q_strands_maths = {
    'Year7': q_strands_maths_y7,
    'Year8': q_strands_maths_y8,
    'Year9': q_strands_maths_y9,
    'Year10': q_strands_maths_y10,
}
# Reading
year_level_q_strands_reading = {
    'Year7': q_strands_reading_y7,
    'Year8': q_strands_reading_y8,
    'Year9': q_strands_reading_y9,
    'Year10': q_strands_reading_y10,
}

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

# # ---------------------------------------------------------------
# # BUILDING NESTED DICTIONARY & APPLYING FUNCTION
# # ---------------------------------------------------------------

# Making dataframe dictionary
# Maths
year_level_dataframes_maths = {
    'Year7': students_maths_match_y7,
    'Year8': students_maths_match_y8,
    'Year9': students_maths_match_y9,
    'Year10': students_maths_match_y10,
}
# Make empty dictionary
class_maths = {}
# Adding to dictionary
for year_level, year_df in year_level_dataframes_maths.items():
    # year_level_dataframes should be a dictionary of year level dataframes (see example below)
    # year_level will be 'Year10', 'Year9', etc.
    # year_df will be students_maths_match_y10, students_maths_match_y9, etc.
    year_classes = match_student_class(year_df, class_list)
    
    # Add to nested structure
    for class_code, df in year_classes.items():
        if class_code not in class_maths:
            class_maths[class_code] = {}
        class_maths[class_code][year_level] = df

# Reading
year_level_dataframes_reading = {
    'Year7': students_reading_match_y7,
    'Year8': students_reading_match_y8,
    'Year9': students_reading_match_y9,
    'Year10': students_reading_match_y10,
}
# Make empty dictionary
class_reading = {}
# Adding to dictionary
for year_level, year_df in year_level_dataframes_reading.items():
    # year_level_dataframes should be a dictionary of year level dataframes (see example below)
    # year_level will be 'Year10', 'Year9', etc.
    # year_df will be students_maths_match_y10, students_maths_match_y9, etc.
    year_classes = match_student_class(year_df, class_list)
    
    # Add to nested structure
    for class_code, df in year_classes.items():
        if class_code not in class_reading:
            class_reading[class_code] = {}
        class_reading[class_code][year_level] = df

# year_level_dataframes = {
#     'Year10': students_maths_match_y10,
#     'Year9': students_maths_match_y9,
#     'Year8': students_maths_match_y8,
#     ...
# }

## End result:
# class_reports_maths['K10-HPE4-2']['Year10']  # DataFrame for Year 10 in this class
# class_reports_reading['K10-HPE4-2']['Year10'] # DataFrame for Year 10 in this class


# # ---------------------------------------------------------------
# # CLEANING AND SAVING
# # ---------------------------------------------------------------

# Printing number of classes
print(f"Number of classes: {len(class_maths)}")
# print(f"Class codes: {list(class_maths.keys())}")

# We will be generating reports for all classes
# For those without data in either Maths or Reading, we will print a statment in the report

# Saving dictionaries to files
# Test Scores
with open("pat_testscores_byclass.pkl", "wb") as f:
    pickle.dump(class_maths, f)
    pickle.dump(class_reading, f)
    #pickle.dump(class_spelling, f)

# Year-level dataframes
with open("yrlvl_df.pkl", "wb") as f:
    pickle.dump(year_level_dataframes_maths, f)
    pickle.dump(year_level_dataframes_reading, f)

# Question substrands
with open("q_strands.pkl", "wb") as f:
    pickle.dump(year_level_q_strands_maths, f)
    pickle.dump(year_level_q_strands_reading, f)
