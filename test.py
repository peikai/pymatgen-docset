import numpy as np
import pandas as pd
from pymatgen.core import Composition
import sqlite3


def contain_unavailable_elements(formula, unavailable_element_list):
    # Check if any element in the formula is in the unavailable_element_list
    try:
        comp = Composition(formula)
        boolean_list = [e.symbol in unavailable_element_list for e in comp.elements]
        return any(boolean_list)  # More pythonic than (True in boolean_list)
    except:
        # Return True if formula can't be parsed (to exclude it)
        return True


working_ion = 'Mg'
# electrode stability: stable, metastable, and all of them
stable_or_not = 'all'
# thermo types: GGA_GGA_U_R2SCAN, GGA_GGA_U or R2SCAN
thermo_type = 'GGA_GGA_U'

# set limits for price and abundance
price_threshold = 'Pt'
abundance_threshold = 'Te'

# read table from local database
CE_database = f'DB/{thermo_type}/CE_{working_ion}.sqlite'
with sqlite3.connect(CE_database) as dbConnection_CE:
    query = "SELECT * FROM FoMs_table"
    CE_dataframe = pd.read_sql(query, dbConnection_CE)

# the elements that are of high expense (more expensive than Pt), scarcity (scarcer than Te), radioactivity, inactivity will be excluded
element_dataframe = pd.read_csv('Tables/Element-information.csv', usecols=['symbol', 'availability', 'abundance(mg/kg)', 'price(USD/kg)'], index_col='symbol')
price_criterion = element_dataframe.loc[price_threshold, 'price(USD/kg)']
costly_element_array = element_dataframe.loc[element_dataframe['price(USD/kg)'] > price_criterion].index
unstable_element_array = element_dataframe.loc[element_dataframe['availability'] == 'Unstable'].index
inactive_element_array = element_dataframe.loc[element_dataframe['availability'] == 'Inactive'].index
abundance_criterion = element_dataframe.loc[abundance_threshold, 'abundance(mg/kg)']
extreme_scarce_element_array = element_dataframe.loc[element_dataframe['abundance(mg/kg)'] < abundance_criterion].index
unavailable_element_array = np.unique(np.hstack((costly_element_array, unstable_element_array, inactive_element_array, extreme_scarce_element_array)))
# available_element_array = np.setdiff1d(element_dataframe.index, unavailable_element_array)

CE_dataframe['contain_unavailable_elements'] = CE_dataframe['formula'].apply(
    contain_unavailable_elements, unavailable_element_list=unavailable_element_array.tolist())
CE_dataframe = CE_dataframe.loc[CE_dataframe['contain_unavailable_elements'] == False]
CE_dataframe.drop(columns=['contain_unavailable_elements'], inplace=True)

# screen candidates with specific criteria, for example, capacity > 600 mAh/g and voltage > 1 V for Li as working ion
capacity_criterion_dict = {'Li': 600, 'Na': 300, 'K': 150,
                           'H': 800, 'Mg': 500, 'Ca': 300}
candidate_dataframe = CE_dataframe.loc[(CE_dataframe['grav_capacity'] >= capacity_criterion_dict[working_ion])].copy()
# candidate_dataframe = candidate_dataframe.loc[CE_dataframe['average_voltage'] >= 1]

# experiment-reported phases are listed on the upper part of the table, then are listed by the *figure_of_merit* field in ascending sort.
# Using inplace=True to avoid SettingWithCopyWarning
candidate_dataframe.sort_values(by=['theoretical', 'figure_of_merit'], ascending=[True, True], inplace=True)
candidate_dataframe.round({'grav_capacity': 3, 'average_voltage': 3, 'figure_of_merit': 3}, inplace=True)

candidate_dataframe.to_csv(f'Tables/{thermo_type}/{working_ion}/candidates_{stable_or_not}.csv', float_format='%.3f', index=False)

candidate_dataframe_Zero2End = candidate_dataframe.loc[(candidate_dataframe['fromZero'] == True) & (candidate_dataframe['toEnd'] == True)]
candidate_dataframe_Zero2End.to_csv(f'Tables/{thermo_type}/{working_ion}/candidates_{stable_or_not}_Zero2End.csv', float_format='%.3f', index=False)

candidate_dataframe_Zero2Cutoff = candidate_dataframe.loc[(candidate_dataframe['fromZero'] == True) & (candidate_dataframe['toEnd'] == False)]
candidate_dataframe_Zero2Cutoff.to_csv(f'Tables/{thermo_type}/{working_ion}/candidates_{stable_or_not}_Zero2Cutoff.csv', float_format='%.3f', index=False)

candidate_dataframe_NoneZero2End = candidate_dataframe.loc[(candidate_dataframe['fromZero'] == False) & (candidate_dataframe['toEnd'] == True)]
candidate_dataframe_NoneZero2End.to_csv(f'Tables/{thermo_type}/{working_ion}/candidates_{stable_or_not}_NoneZero2End.csv', float_format='%.3f', index=False)

candidate_dataframe_NoneZero2Cutoff = candidate_dataframe.loc[(candidate_dataframe['fromZero'] == False) & (candidate_dataframe['toEnd'] == False)]
candidate_dataframe_NoneZero2Cutoff.to_csv(f'Tables/{thermo_type}/{working_ion}/candidates_{stable_or_not}_NoneZero2Cutoff.csv', float_format='%.3f', index=False)
