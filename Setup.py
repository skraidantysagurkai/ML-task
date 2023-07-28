# This script loads the IMF 2020 OCT data, fills missing values , converts datatypes, so to
# make work with the data easier.
import pandas as pd

# Please change the data path to your path
IMF_OCT_DATA_PATH = '/Users/rokassabaitis/Downloads/WEOOct2020all.xlsx'
imf_data = pd.read_excel(IMF_OCT_DATA_PATH, engine='openpyxl')

# Converting numbers in year columns from object type to float64
imf_data_numbers = imf_data.loc[:, 1980: 2025]
num_col_names = imf_data.loc[:, 1980: 2025].columns

imf_data_numbers = imf_data_numbers.replace({',': '', '--': '0.0'}, regex=True).astype(float)
imf_data_numbers = imf_data_numbers.fillna(0.0)
imf_data[num_col_names] = imf_data_numbers

imf_data['Estimates Start After'] = imf_data['Estimates Start After'].fillna(0.0)
imf_data = imf_data.fillna('Missing')

imf_data.to_pickle("IMF_DATA.pkl")

print("Setup Complete")
