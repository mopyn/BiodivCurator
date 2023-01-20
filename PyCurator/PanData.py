####################################################
''' Description
A collection of functions to load and work with PANGAEA data
'''
####################################################

###############################
# Import libraries
import os, glob, re
import pandas as pd
import numpy as np
from functools import reduce
import re
from difflib import get_close_matches
from decimal import Decimal # To handle scientific notations

from PyCurator.PyCurator import PyCurator as pc
###############################

#######################################
# Load PANGAEA data from local database
#######################################


#-------------------------------------------------------------------------------
# Load PANGAEA data from local database
def load_db(db_path, db_type = "parameter", cols = None):
    """To load different types of Pangaea databases

    Args:
        db_path (str): File path to database files
        db_type (str): database type. Defaults to "parameter".
        cols (list, optional): list of database columns to be included. Defaults to None.

    Returns:
        pandas.core.frame.DataFrame: database returned as pandas dataframe
    """
    # List all db files
    files = [os.path.basename(path) for path in glob.glob(f'{db_path}*')]
    
    # Grep filename of chosen database
    db_name = [file for file in files if db_type in file][0]

    # Load database
    db = pd.read_csv(f'{db_path}{db_name}', sep = '\t', on_bad_lines = "warn") # error_bad_lines=False

    # Remove all empty columns
    db.dropna(how='all', axis=1, inplace=True)

    if cols:
        db = db[cols]

    print(f'Database {db_name} loaded with {db.shape[0]} rows and {db.shape[1]} columns')

    return db



#-------------------------------------------------------------------------------
# Function to concatenate multiple string columns of dataframe
def join_cols(db, columns, sep = "_"):
    """Function to concatenate strings of multiple dataframe columns

    Args:
        db (pandas.core.frame.DataFrame): pandas data frame
        columns (list): list of strings matching the database headers
        sep (str, optional): separator between joined strings. Defaults to " ".

    Returns:
        list: list of joined string columns
    """
    assert len(columns) > 1

    slist = [db[x].astype(str) for x in columns]

    return reduce(lambda x, y: x + sep + y, slist[1:], slist[0])



#-------------------------------------------------------------------------------
# Function to remove brackets from strings
def strip_brackets(str_list):
    """Strip all brackets from list of strings

    Args:
        str_list (list): list of strings

    Returns:
        list: List of input strings without brackets
    """
    return [re.sub('[\[\]]', "", item).strip() for item in str_list]


#-------------------------------------------------------------------------------
# Function to extract the unit from the parameter header
def get_unit(param_list, char = "["):
    """Extract unit from parameter list

    Args:
        df (list): list with parameters containing units
        char (str, optional): starting enclosing symbol marking the unit. Defaults to "[".

    Returns:
        _type_: _description_
    """
    # Extract unit based on char
    unit = [f'{char}{param.split(char)[1]}' if char in param else "" for param in param_list]
    # Remove brackets
    unit = strip_brackets(unit)

    return unit


#-------------------------------------------------------------------------------
# Strip any comments from string list
def strip_comments(str_list, char = "\/\/"):
    """Strip any comments from string list based on provided splitting character

    Args:
        str_list (list): list of strings
        char1 (str): 1st character marking the beginning of comment. Defaults to ""\/\/".

    Returns:
        _type_: _description_
    """
    # List without comments
    no_comments = [re.split(char, x)[0] for x in str_list]
    # List with comments extracted based on splitting character
    #comments = [re.split(char, x)[1:][0] if char in x else "" for x in str_list]

    return no_comments


#-------------------------------------------------------------------------------
# Function to abbreviate species names
def abbreviate_species(str_list, except_str = ["spp.", "sp."]):
    """Abbreviate species names

    Args:
        str_list (list): list of species names_
        except_str (list, optional): no abbreviation for these strings. Defaults to ["spp.", "sp."].

    Returns:
        list: list of abbreviated strings
    """
    # Abbreviate first word in list
    return [f'{x.split(" ")[0][0]}. {" ".join(x.split(" ")[1:])}' if not any(map(x.__contains__, except_str)) else x for x in str_list]


#-------------------------------------------------------------------------------
# Function to create search dataframe to match database columns
def create_find_df(str_list):
    """Create data frame with Parameter and Unit columns to enable database search
    
    Args:
        str_list (list): list of parameters and units in the format "Event []"

    Returns:
        pandas.core.frame.DataFrame: dataframe with the columns Parameter and Unit
    """
    # Check if comment contains units
    # Separate parameter from units based on " [" symbol
    parameters = [x.split(" [")[0] for x in str_list]
    # Get units if there are any
    units = [f'[{x.split(" [")[1]}' if re.findall("\[", x) else "" for x in str_list]
    # Remove brackets from unit
    units = strip_brackets(units)

    # Return data frame
    return pd.DataFrame({"Parameter": parameters, "Unit": units})


#-------------------------------------------------------------------------------
# Find exact matched and unmatched rows of one dataframe in another
def find_df_matches(df_find, database, left_on = ["Parameter", "Unit"], right_on = ["Parameter", "Unit"]):
    """Find matched and unmatched rows of one dataframe in another based on multiple features
    Note: Both dataframes need to have identical headers

    Args:
        df_find (pandas.core.frame.DataFrame): dataframe with entries to be found
        database (pandas.core.frame.DataFrame): dataframe that is searched
        features (list, optional): List of features to be matched. Defaults to ["Parameter", "Unit"].

    Returns:
        pandas.core.frame.DataFrame: two dataframes with matched or unmatched rows of df_find
    """

    # Replace all Nan in database
    database = database.replace(np.nan, "")

    # Concatenate all features in df_find and database to a single string
    df_find_concat = pd.Series([''.join(row.astype(str)) for row in df_find[left_on].values])
    database_concat = pd.Series([''.join(row.astype(str)) for row in database[right_on].values])

    # Look up matched entries based on two features
    matched_results = df_find_concat.isin(database_concat)    # Extract matched entries
    matched = df_find[matched_results]

    # Extract unmatched entries of df_find
    unmatched = df_find[~matched_results]
    # Print result
    print(f'{matched.shape[0]} exact matches and {unmatched.shape[0]} unmatched items')

    return matched, unmatched


#-------------------------------------------------------------------------------
# Extract dataframe features not found in database
def get_unmatched_df(dataframe, database):
        """To get dataframe features that could not be found in database

        Args:
            dataframe (pandas.core.frame.DataFrame): dataframe with features to be searched in database
            database (pandas.core.frame.DataFrame): database to be searched in
 
        Returns:
            pandas.core.frame.DataFrame: dataframe with unmatched features
        """

        # Find matched and unmatched items of one dataframe in another
        _, unmatched = find_df_matches(create_find_df(dataframe.columns), database)
        
        #### Extract unmatched columns from data table ####
        # For this we first join parameters and units for the search to account for identical parameters with different units
        param_search = [f'{param} [{unit}]' for param, unit in zip(unmatched["Parameter"], unmatched["Unit"])]
        # Remove any parameters with numeric IDs
        param_search = [param for param in param_search if param if not re.search(r'^[0-9]+', param)]
        # Then we find all feature columns using partial match to account for comments
        unmatched_cols = [dataframe.columns[dataframe.columns.str.contains(x, regex=False)][0] for x in param_search]
        # to then extract those from the original dataframe
        dataframe = dataframe[unmatched_cols]

        return dataframe


#-------------------------------------------------------------------------------
# Function to find n number of close matches
def get_close_match(df_find, database, n_matches):
    """Function that finds close n number of matches of one list in another

    Args:
        df_find (list): list to be matched
        database (list): list with content to be matched with
        n_matches (int): number of closest matches sorted by score

    Returns:
        pandas.core.frame.DataFrame: dataframe with df_list and close matches
    """

    # Create empty data frame to store matches
    close_matches = pd.DataFrame()

    # Loop over df_find and append close matches of list 2 to data frame
    for index, str_to_find in enumerate(df_find):
        # Get close matches for list element in list 2
        close_match = get_close_matches(str_to_find, database, n=n_matches)

        # Append to data frame if there where matches
        if close_match:
            # Create data frame
            close_match = list(np.append(str_to_find, close_match)) # numpy list append works better than list.append
            close_matches = pd.concat([close_matches, pd.DataFrame(close_match).T], ignore_index=True)
        
        else:
            # If no match replace with empty string
            close_match = ["unmatched"] * n_matches
            # Create data frame
            close_match = list(np.append(str_to_find, close_match)) # numpy list append works better than list.append
            close_matches = pd.concat([close_matches, pd.DataFrame(close_match).T], ignore_index=True)
            

        # Print progress index and close matches for each loop
        print(f'{index+1}/{len(df_find)}: Close matches for {str_to_find} are {close_match[1:]}')
        
    print(len(close_match))
    # Add column event labels with prefix and increasing index
    col_names = ["Search_term"]
    col_names.extend(["close_match_" + str(x) for x in list(range(1, len(close_match)))])
    
    # Rename column header
    close_matches.columns = col_names
        
    return close_matches


#-------------------------------------------------------------------------------
# Function that finds close matches in a database allowing multiple search columns
def close_db_matches(df_find, database, n_matches, features = ["Parameter", "Unit"], save = True):
    """Function to find approximate matches in a database allowing multiple search columns
    Note: 

    Args:
        df_find (pandas.core.frame.DataFrame): Dataframe with one or more search columns
        database (pandas.core.frame.DataFrame): Dataframe with one or more search columns
        n_matches (int): N number of the best matching strings
        features (list, optional): list of headers to be included in search. Defaults to ["Parameter", "Unit"].
        save (bool, optional): Save as excel file if true. Defaults to True.

    Returns:
        _type_: _description_
    """
    #Reset index in case dataset slice was passed to df_find
    df_find = df_find.reset_index(drop=True)
    
    # Concatenate search terms in new column
    # For dataframe with search terms
    df_find["search_term"] = df_find[features].apply(" ".join, axis=1).fillna("")
    # For database
    database = database.fillna("")
    database["search_term"] = database[features].apply(" ".join, axis=1)

    # Find closest matches (uses function from difflib package)
    close_matches = get_close_match(df_find["search_term"], database["search_term"], n_matches = n_matches)

    # Append close matches to df_find
    close_matches = pd.concat([df_find, close_matches.iloc[:, 1:]], axis=1)

    # Save as close matches as Excel file
    if save:
        close_matches.to_excel("close_matches.xlsx", index=False)

    return close_matches


#-------------------------------------------------------------------------------
# Create empty import data table based on the supplied headers
def create_import_table(headers = []):
    """Create empty import data table based on the supplied headers

    Args:
        headers (list): List of header strings. Defaults to [].

    Returns:
        pandas.core.frame.DataFrame: empty import table
    """
    # Create an empty data frame from parameters
    return pd.DataFrame(columns=headers)



#-------------------------------------------------------------------------------
# Remove quality flags from dataframe
def del_quality_flag(dataframe, flag_char = "<"):
    """Remove quality flags from dataframe and convert to numeric

    Args:
        dataframe (pandas.core.frame.DataFrame): dataframe
        flag_char (str, optional): quality flag to be searched for. Defaults to "<".

    Returns:
        _type_: _description_
    """
    # Function to find columns with the quality flag
    flag_cols = pc.cols_with_char(dataframe, flag_char)
    # Delete the quality flag
    dataframe[flag_cols] = pc.replace_char(dataframe[flag_cols], flag_char, "", verbose = False)
    # Convert columns to numeric
    dataframe[flag_cols] = dataframe[flag_cols].apply(pd.to_numeric)

    return dataframe


#-------------------------------------------------------------------------------
# Calculate lower limit for each numeric dataframe columns
def lower_limit(dataframe):
    """Calculate lower limit for each numeric dataframe columns rounding to zero for positives and the next lower tens for negatives

    Args:
        dataframe (pandas.core.frame.DataFrame): dataframe with numeric values

    Returns:
        list: List with lower limits
    """
    # Try to convert to numeric columns
    # dataframe = dataframe.apply(pd.to_numeric)
    
    lower_limit = []
    # Loop over all dataframe datatypes
    for i, type in enumerate(dataframe.dtypes):
        # Append nothing if column is non-numeric
        if type == "object":
            lower_limit.append("")
        # Append nothing if column contains nan only
        elif dataframe.iloc[:,i].isnull().values.all() == True:
            lower_limit.append("")
        # If there are numeric values calculate minimum
        else:
            # Calculate minimum value of data column
            df_min = dataframe.iloc[:,i].dropna().min()
            # if minimum is larger than 0 append 0
            if df_min > 0:
                lower_limit.append(0)
            else:
                # if negative multiple by 10 e.g. -4=40
                lower_limit.append(int(df_min)*10)
    
    return lower_limit


#-------------------------------------------------------------------------------
# Calculate upper limit for each numeric dataframe columns
def upper_limit(dataframe):
    """Calculate upper limit for each numeric dataframe columns rounding to the next higher tens
    Non-numeric columns are set to empty

    Args:
        dataframe (pandas.core.frame.DataFrame): dataframe with numeric values

    Returns:
        list: List with upper limits
        round(number/100)*100
    """

    upper_limit = []
    # Loop over all dataframe datatypes
    for i, type in enumerate(dataframe.dtypes):
        # Append nothing if column is non-numeric
        if type == "object":
            upper_limit.append("")
        # Append nothing if column contains nan only
        elif dataframe.iloc[:,i].isnull().values.all() == True:
            upper_limit.append("")
        # If there are numeric values calculate maximum
        else:
            # Calculate the maximum for all numeric columns
            upper_limit.append(int(dataframe.iloc[:,i].dropna().max()))

    # Calculate the maximum for all numeric columns
    #upper_limit = [int(dataframe[col].max()) if not type == "object" or not dataframe[col].isnull().values.all() else "" for type, col in zip(dataframe.dtypes, dataframe.columns)]

    # Count the length of each integer
    upper_limit = [len(str(int(x))) if isinstance(x, (float, int)) else 0 for x in upper_limit]

    # Convert to 9*n format
    upper_limit = [int(f'{"9"*x}9') for x in upper_limit]

    # if there is % unit than use 100 as maximum
    # needs to be still done

    return upper_limit


#-------------------------------------------------------------------------------
# Calculate the number of decimals for each dataframe column

# Calculate the number of decimals for each dataframe column
def get_decimals(df_column):
    """Calculate the number of decimals for each dataframe column

    Note: This function can also handle scientific notation

    Args:
        df_column (pandas.Series): pandas.Series

    Returns:
        pandas.Series: pandas.Series with total number of decimals
    """
    return [abs(Decimal(str(x)).as_tuple().exponent) for x in df_column.dropna()]



#-------------------------------------------------------------------------------
# Find the highest decimal count for each column
def max_decimals(dataframe, dec_limit = 4):
    """Find the highest decimal count for each column

    Args:
        dataframe (pandas.core.frame.DataFrame): dataframe with numeric values
        dec_limit (int): maximal allowed decimals. Defaults to 4.

    Returns:
        list: list with decimals
    """
    # Crete empty list to save decimals
    dec = []
    # Loop over each column
    for i, _ in enumerate(dataframe.columns):
        # Check if data column contains floats
        if dataframe.iloc[:,i].dtype == "float64":
            # Extract decimals and calculate maximum number of decimals in column; if there is empty sequence set default to 0 decimals
            dec.append(max(get_decimals(dataframe.iloc[:,i]), default=0))
        # # Check if data column contains integer
        else:
            # otherwise decimals are 0
            dec.append(0)
    
    # Reduce number of decimals to reasonable degree
    dec = [dec_limit if x > dec_limit else x for x in dec]

    return dec



#-------------------------------------------------------------------------------
# Create default data format for each data column
def default_format(dataframe, dec_limit = 4):
    """Create default data format for each data column e.g. ###.0.00

    Args:
        dataframe (pandas.core.frame.DataFrame): dataframe with numeric values
        dec_limit (int): maximal allowed decimals. Defaults to 4.

    Returns:
        list: List with default formatting strings
    """
    # Calculate number of integer digits
    # Strings are set to 0
    len_digits = [len(str(int(x)))-1 if isinstance(x, (float, int)) else 0 for x in upper_limit(dataframe)]
    # Calculate maximum number of decimals
    max_dec = max_decimals(dataframe, dec_limit = dec_limit)
    # Create default format    
    default_format = ["".join(["#"]*len)+"0" if dec == 0 else "".join(["#"]*len)+"0." + "".join(["0"]*dec) for len, dec in zip(len_digits, max_dec)]

    return default_format


#-------------------------------------------------------------------------------
# Encode datatypes into number codes for each dataframe column
def data_type(dataframe):
    """Identify datatypes for each dataframe column and allocate corresponding number code

    Args:
        dataframe (pandas.core.frame.DataFrame): dataframe

    Returns:
        list: list of numeric data type codes
        1 = Numeric
        2 = Text
        3 = Datetime
        4 = binary
        5 = Feature
        6 = URI
        7 = Event
    """
    # Convert columns to best possible dtypes
    dataframe = dataframe.convert_dtypes()
    
    # Change value to 1 if numeric
    data_types = [1 if "64" in str(x) else str(x) for x in dataframe.dtypes]
    # Change value to 2 if string
    data_types = [2 if "string" in str(x) else x for x in data_types]
    # Change value to 3 if datetime
    data_types = [3 if "datetime" in str(x) else str(x) for x in data_types]
    # Change value to 4 if boolean
    data_types = [4 if "bool" in str(x) else str(x) for x in data_types]

    return data_types



#-------------------------------------------------------------------------------
# Create import table for parameters
def get_imp_param(unmatched_param_df, 
                    file_path,
                    DefaultMethodID = "", 
                    ReferenceID = "", 
                    Description = "", 
                    url_parameter = "",
                    df_name = "",
                    dec_limit = 4,
                    del_duplicates = False,
                    ):

    """Create import table for parameters

    Args:
        unmatched_param_df (_type_): _description_
        DefaultMethodID (str, optional): _description_. Defaults to "".
        ReferenceID (str, optional): _description_. Defaults to "".
        Description (str, optional): _description_. Defaults to "".
        url_parameter (str, optional): _description_. Defaults to "".

    Returns:
        _type_: _description_
    """
    # Check if there are unmatched entries
    if not unmatched_param_df.empty: 
        # Create import table
        imp_df = create_import_table(headers=[
                                        'ParameterName',
                                        'Abbreviation',
                                        'Unit',
                                        'LowerLimit',
                                        'UpperLimit',
                                        'DefaultFormat',
                                        'DefaultMethodID',
                                        'DefaultDataType',
                                        'ReferenceID',
                                        'Description',
                                        'URL Parameter'
                                                ])

        # Add parameter names
        imp_df.ParameterName = strip_comments(unmatched_param_df.columns, " \[#*.*")
        # Add abbreviations
        imp_df.Abbreviation = abbreviate_species(strip_comments(unmatched_param_df.columns, " \[#*.*"))
        # Add units
        imp_df.Unit = get_unit(unmatched_param_df.columns)
        # Add LowerLimit
        imp_df.LowerLimit = lower_limit(unmatched_param_df)
        # Add UpperLimit --> rounded to the next higher tens
        imp_df.UpperLimit = upper_limit(unmatched_param_df)
        # Add DefaultFormat
        imp_df.DefaultFormat = default_format(unmatched_param_df, dec_limit = dec_limit)
        # Add DefaultMethodID
        imp_df.DefaultMethodID = DefaultMethodID
        # Add DefaultDataType
        imp_df.DefaultDataType = data_type(unmatched_param_df)
        # Add ReferenceID
        imp_df.ReferenceID = ReferenceID
        # Add Description
        imp_df.Description = Description
        # Add URL Parameter
        imp_df["URL Parameter"] = url_parameter

        # Remove duplicate parameters
        if del_duplicates == True:
            imp_df = imp_df.drop_duplicates(subset = ['ParameterName', 'Abbreviation', 'Unit'],
                        keep = 'last').reset_index(drop = True)
        # Save import dataframe as tab delimited file
        imp_df.to_csv(f'{file_path}{df_name}_ParamImp.txt', index=False, sep="\t", encoding='utf-8')

        return imp_df