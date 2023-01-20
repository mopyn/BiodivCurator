####################################################
''' Description
A collection of useful functions to curate tabular data
'''
####################################################


###############################
# Metadata
author = "Michael Oellermann"
last_updated = "2022-07-14"
version = "0.0.3"
###############################


###############################
# Import libraries
import os, glob
import numpy as np
import pandas as pd
import re # for regular expressions
import openpyxl
import requests
from LatLon23 import string2latlon # for geodata conversions
from mendeleev import element #https://pypi.org/project/mendeleev/
###############################


###############################
# File processing
###############################

# Function to find all files in a target folder
def list_files(file_path, abs_path = True, file_type = '*.*'):
    """Function to list all files in folder

    Args:
        file_path (str): Absolute file path for search folder
        absolute_path (bool, optional): Returns absolute file path. Defaults to True.

    Returns:
        _type_: _description_
    """
    # List files with absolute path
    files = glob.glob(os.path.join(file_path, file_type))

    # Extract file names only
    if not abs_path:       
        files = [os.path.split(file)[-1] for file in files]

    return files

#-------------------------------------------------------------------------------
# Find all data files in directory
def find_data_files(sub_folder):
    """Function that locates and returns all files in given folder separated by data file type

    Args:
        sub_folder (str): Path to the folder

    Returns:
        lists: Three lists of full paths for each data file separated by xlsx, txt and csv
    """
    # List only files within sub_folder (no subfolder names)
    files = [f for f in os.listdir(sub_folder) if re.search("\.[a-z]", f)]
    print(f'This folder contains {files}')

    # Create absolute file path
    files = [f'{sub_folder}{file}' for file in files]

    # Get Excel files
    xlsx_files = [file for file in files if "xlsx" in file]

    # Get csv files
    csv_files = [file for file in files if "csv" in file]

    # Get txt files
    txt_files = [file for file in files if "txt" in file]
     
    return xlsx_files, csv_files, txt_files


#-------------------------------------------------------------------------------
# Function to detect and laod files into dictionary
def load_df(sub_folder, sep = "\t", encoding = "utf-8"):
    """Load different data table formats (xlsx, csv, txt) simoultenously into a single dictionary

    Note: txt files can contain comma or tab separated columns.

    Args:
        sub_folder (str): subfolder to lookup files
        encoding (str): In case default utf-8 encoding does not work, change to 'unicode_escape' or "iso8859-1"

    Returns:
        dict: Dictionary with all data tables
    """
    # Find all data files
    xlsx_files, csv_files, txt_files = find_data_files(sub_folder)

    # Load data
    # Check if data have already been loaded into dictionary
    if 'df_dict' in locals():
        print(f'{len(df_dict)} datasets {[df_names for df_names in df_dict.keys()]} have already been loaded')
    # If not load files   
    else:
        print("Reading data...")
        df_dict = {}
        # Excel files
        if xlsx_files:
        # Read multiple excel sheets and store in dictionary of dataframes
            for file in xlsx_files:
                df_dict_temp = pd.read_excel(file, sheet_name=None)
                # Create file name from file path
                file_name = file.split("\\")[-1:][0].split(".")[0]
                print(file_name, end=" ")
                # Add filenames to dict key as unique ID to avoid identical sheet names being overwritten
                df_dict_temp = {f'{file_name}_{str(key)}': val for key, val in df_dict_temp.items()}
                # Append dictionary to final dictionary
                df_dict.update(df_dict_temp)
                print(f'susccesfully loaded')
        # csv files
        if csv_files:
            # Read multiple csv files and store in dictionary of dataframes
            for file in csv_files:
                # Create file name from file path
                file_name = file.split("\\")[-1:][0].split(".")[0]
                print(file_name, end=" ")
                df_dict[file_name] = pd.read_csv(file, encoding = encoding, sep = sep)
                print(f'susccesfully loaded')
        # txt files
        if txt_files:
            # Read multiple csv files and store in dictionary of dataframes
            for file in txt_files:
                # Create file name from file path
                file_name = file.split("\\")[-1:][0].split(".")[0]
                print(file_name, end=" ")
                #df_dict[file_name] = pd.read_csv(file, sep=r'\,|\t', engine = "python")
                df_dict[file_name] = pd.read_csv(file, sep=sep, encoding = encoding)
                print(f'susccesfully loaded')
        
        # Report the number of dataframes and filenames
        print(f'{len(df_dict)} dataframe(s) loaded')
        
        return df_dict

#-------------------------------------------------------------------------------


###############################
# String processing
###############################

#-------------------------------------------------------------------------------
# Find all dataframe columns where values contain a certain substring
def cols_with_char(dataframe, char = "<"):
    """Find all dataframe columns where values contain a certain substring

    Args:
        dataframe pandas.core.frame.DataFrame): dataframe
        char (str, optional): String contained in dataframe column. Defaults to "<".

    Returns:
        list: list of columns names containing the substring
    """
    # Convert all columns to objects to enable string search
    dataframe = dataframe.astype(str)
    return [col for col in dataframe if dataframe[col].str.contains(char).any()]


#-------------------------------------------------------------------------------
# Function to detect and count specified characters in data frame
def count_char(df, char = ","):
    """Functions that counts the number of a specified characters in each column
    Note: This method ignores numeric values including NaN
   
    Args:
        df (pandas.core.frame.DataFrame): pandas dataframe
        char (str): Char symbol specified by user Defaults to ",".

    Returns:
    int: Total sum of replaced characters

    Dependency: numpy
    """

    # Convert to pandas data frame if series to avoid function errors for pandas series
    if isinstance(df, pd.Series):
        df = df.to_frame()

    total_sum = 0
    for column in df.columns:
        # Search for substring in all non-numeric values in dataframe column allowing regular expression
        char_found = [bool(re.search(char, x)) if isinstance(x, str) else False for x in df[column]]
        # Store row indices where substring was found
        char_index = df.index[char_found].tolist()
        # Calculate total number of occurences of substring for this column
        char_sum = np.sum(char_found)
        # Calculate cumulative sum
        total_sum += char_sum
        # Print the number of found characters for each column
        print(f'There are {char_sum} {char} in {column} at index {char_index}')
    return total_sum # return total sum of found characters

#-------------------------------------------------------------------------------
# Function to replace any character in the whole dataframe
def replace_char(dataframe, char = ",", new_char = ".", verbose = False):
    """Function that replaces any specified character in the whole dataframe

    Args:
        df (pandas.core.frame.DataFrame): pandas dataframe
        char (str): Character to be replaced Defaults to ",".
        new_char (str): New character specified by user Defaults to ".".
    
    Returns:
        pandas.core.frame.DataFrame: dataframe with replaced character
    """
    # Create deep copy
    df = dataframe.copy()

    if verbose:
        # Calculate and print total number of replacements
        total_sum = count_char(df, char)
        print(f'"{total_sum} occurence(s) replaced"')
    
    # Convert to pandas data frame if series to avoid function errors for pandas series
    if isinstance(df, pd.Series):
        df = df.to_frame().copy()

    for column in df.columns:
        # Replace all characters if those are strings only and not numeric
        #df.loc[:,column] = [x.replace(char, new_char) if isinstance(x, str) else x for x in df[column]]
        df.loc[:,column] = [re.sub(char, new_char, x) if isinstance(x, str) else x for x in df[column]]

    return df


#-------------------------------------------------------------------------------
# Function to rename multiple column headers across multple dataframes
def replace_param(df_dict, old_params, new_params):
    """Function to rename multiple column headers across multple dataframes
    stored in a dictionary

    Args:
        df_dict (dictionary): dictionary containing the dataframes
        old_params (list): List of header strings to be replaced
        new_params (list): List of new header strings
    """
    # Loop over the dataframes stored in dictionary    
    for key, df in df_dict.items():
        # Loop over the old and new header names
        for i, _ in enumerate(old_params):
            # Find and replace old header names
            df.columns = [x.replace(old_params[i], new_params[i]) for x in df.columns]


#-------------------------------------------------------------------------------
# Add brackets to units in header column
def add_brackets(string_list, units = [], bracket = ["[", "]"], sep = " "):
    """Function that adds brackets or other enclosing symbols to units    

    Args:
        string_list (list): List of strings e.g. headers
        units (list): List of strings containing units. If empty the last string following the seperator is used. Defaults to [].
        bracket (list, optional): List of two strings indicating the enclosing symbols. Defaults to ["[", "]"].
        sep (str, optional): Seperator used to split the header string. Defaults to " ".

    Returns:
        _type_: _description_
    """
    # If units is empty extract the last split item
    if not units:
        units = [unit.split(sep)[-1:][0] for unit in string_list]
    # Then look for each unit in header string and add brackets
    new_header = [re.sub(unit, f'{bracket[0]}{unit}{bracket[1]}', header)  for header, unit in zip(string_list, units)]
    return new_header


# Add brackets
def add_suffix(str_list, suffix, ignore = ""):
    """Adding a suffix to each string from a list of strings

    Args:
        list (list): list of strings
        suffix (str): string suffix to be added
        ignore (str): sub-string matching entries that should be ignored

    Returns:
        list: list of strings with added suffix
    """
    return [header if ignore in header else header + suffix for header in str_list]

###############################
# Data conversions
###############################

#-------------------------------------------------------------------------------
# Function to convert any date to the PANGAEA standard format
def toPangaeaDate(date_col, date_format = '%Y-%m-%d %H:%M:%S', output_format = '%Y-%m-%dT%H:%M:%S'):
    """This function converts any date format into the standard PANGAEA date format

    Args:
        date_col (datetime or string): Data series of dates
        date_format (str, optional): The original date format needs to be entered to assure proper conversion Defaults to '%Y-%m-%d %H:%M:%S'.

    Returns:
        datetime: data series with converted dates
    """
    # Convert to date time format
    ### Important to provide exact original date format!!
    new_date = pd.to_datetime(date_col, format = date_format, utc=True) 
    # Convert to Pangaea standard time format as JJJJ-MM-TT T00:00:00 
    new_date = new_date.dt.strftime(output_format)
    print(f'Old date/time format {date_col.iloc[0]} converted to PANGAEA date format {new_date.iloc[0]}')
    # Return formated dates
    return new_date

#-------------------------------------------------------------------------------


###############################
# Dataframe restructering
###############################

#-------------------------------------------------------------------------------
# Function to convert dataframe from wide to long based on substring contained in headers
def df_wide_to_long(df = None, 
                    split_string = None, 
                    split_colname = None, 
                    split_value = [], 
                    subset_headers = None, 
                    col_add_name = None, 
                    col_add_data = None,
                    str_match = "contains"):
    """This function convert a dataframe from wide to long based on a substring contained in headers

    Args:
        df (pandas.core.frame.DataFrame): pandas data frame
        split_string (list): character or substring contained in datafame columns
        split_colname (str): Header name of the new substring grouping column
        split_value (list): Optional values entered into substring grouping column instead of the substring itself
        subset_headers (list): Optional header names of columns that are subset. 
        They need to be identical to allow subsequent concatenation of all subsetted dataframes
        col_add_name (list): Header name for an additional column to be included at position 0
        col_add_data (list): Data for an additional column to be included at position 0
        str_match (str): How substring should be filtered: Contains = subtring match, exact = exact match

    Returns:
        _type_: _description_
    """

    # Mandataory data entry check
    if split_string is None:
        print("WARNING: Pleae provide a string to split the dataframe such as '5 cm'")
      
    # create empty list to store subset dataframes
    df_split = []
    # Loop over split_string
    for i, char in enumerate(split_string):
        if str_match == "contains":
            # Get columns containing the split_string in header
            df_sub = df.filter(like=char, axis=1)
        elif str_match == "exact":
            # Get columns that match substring exactly
            df_sub = df[char]

        # Convert to pandas data frame if series to avoid function errors for pandas series
        if isinstance(df_sub, pd.Series):
            df_sub = df_sub.to_frame()

        if subset_headers:
            # Rename headers of subsetted columns
            # This assures that subsets can be concatenated by common column names
            df_sub.columns = subset_headers
        else:
            # if no subset headers are provided simply provide indices
            df_sub.columns = list(range(0, df_sub.shape[1]))

        # Check if split_colname is empty
        if split_colname is None:
            split_colname = "Splitting_character"
        
        if len(split_value) > 0:
            # Add grouping column based on splitting character
            df_sub.insert(0, split_colname, split_value[i])
        else:
            # If there is no split_value the use the split_string as label
            df_sub.insert(0, split_colname, split_string[i])
        
        if col_add_name and col_add_data:
            # Add additional columns at position 0
            for i, _ in enumerate(col_add_name):
                df_sub.insert(0, col_add_name[i], col_add_data[i])

        # Attach subsets to list
        df_split.append(df_sub)
        
    # Join all subsetted dataframes into one
    return pd.concat(df_split).reset_index(drop = True)


#-------------------------------------------------------------------------------
# Function to remove NaN rows from dataframe
def drop_na_rows(df, columns = None, drop = "all", reset_index = False):
    """Function to delete NaN rows from dataframe, either all, from the beginning or end only or both

    Args:
        df (pandas.core.frame.DataFrame): pandas data frame
        columns (list): List of columns for which all have to contain NaNs. Defaults to None.
        drop (str): Indicates if either all rows, from the beginning, the end or both should be dropped. Defaults to "all".
        reset_index (bool): Resetting the index after rows are dropped. Defaults to False.

    Returns:
        pandas.core.frame.DataFrame: pandas data frame
    """
    # delete all nan rows 
    if drop == "all":
        df_new = df.dropna(subset = columns, how = "all")

    # Get index of first non NaN row
    start_row = df[~df[columns].isna().all(axis = 1)].index[0]
    # Get index of last non NaN row
    end_row = df[~df[columns].isna().all(axis = 1)][::-1].index[0]

    # Delete only all nan rows at the beginning till first value occurs
    if drop == "start":
        # Extract dataframe from first non NaN value
        df_new = df.iloc[start_row:]
    # Delete only all nan rows at the end till first value occurs
    if drop == "end":
        # Extract dataframe from before last non NaN row
            ## Indexing needs +1 to include last numeric row
        df_new = df.iloc[:end_row+1]
    # Delete all nan rows at the beginning AND end till first value occurs but not in between
    if drop == "start_end":
        df_new = df.iloc[start_row:end_row+1]

    # Reset index
    if reset_index:
        df_new = df_new.reset_index(drop=True)

    return df_new
#-------------------------------------------------------------------------------

###############################
# Geodata functions
###############################

def geo_decimal(latitude, longitude, format = "d%°%m%'%S%''%H", decimals = 6):
    """Function to convert geo location data into decimals

    Args:
        latitude (list): Latitude in degrees format
        longitude (list): Longitude in degrees format
        format (str): format of provided geo location. Defaults to "d%°%m%'%S%''%H".
        decimals (int): Number of decimals for returned latitude and longitude. Defaults to 6.

    Returns:
        list: latitude and longitude in decimals
    """
    # Convert invalid formatting
    # Convert " signs to ''
    latitude = [x.replace("\"", "\'\'") for x in latitude]
    longitude = [x.replace("\"", "\'\'") for x in longitude]

    # Convert latitude and longitude from degree to decimal format
    lat_dec = [round(string2latlon(lat, long, format).lat.decimal_degree, decimals) for lat, long in zip(latitude, longitude)]
    long_dec = [round(string2latlon(lat, long, format).lon.decimal_degree, decimals) for lat, long in zip(latitude, longitude)]

    return lat_dec, long_dec

#-------------------------------------------------------------------------------

#################################
# Functions for chemistry dataset
#################################

#-------------------------------------------------------------------------------
# Function to spell out chemical element abbreviations
def translate_element(chemical_symbol):
    """Function to spell out chemical element abbreviations

    Args:
        chemical_symbol (str): Single string such as "Cu"

    Returns:
        str: full name of chemical element
    """
    return element(chemical_symbol).name


#-------------------------------------------------------------------------------
# Function to translate list of element abbreviations
def translate_element_list(element_list, sep = " "):
    """Translates abbreviated chemical elements contained in list of strings

    Args:
        element_list (list): List of strings containing abbreviated elements
        sep (str, optional): separator to split extract elements in string. Defaults to " ".

    Returns:
        list: list of strings with spelled out chemical elements
    """
    # Separate element abbreviations from everything else
    elements, units = zip(*[(x.split(sep)[0], " ".join(x.split(sep)[1:])) for x in element_list])

    # Translate each symbol
    translated = [translate_element(element) for element in elements]

    # Merge translated elements and units back together
    translated = [f'{trans}{sep}{unit}' for trans, unit in zip(translated, units)]

    return translated



#################################
# Web functions
#################################

def check_url(url_list):
    """Function to check if url can be reached

    Args:
        url_list (list): list of urls

    Returns:
        list: result of url checks
    """
    # Or < 400
    return [True if requests.get(url).status_code == 200 else False for url in url_list]