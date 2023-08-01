"""
    This file is made to house the create_model and make_prediction functions.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


# Dictionary to match a country to a continent
COUNTRY_CONTINENT_DIC = {
    'Algeria': 'Africa',
    'Angola': 'Africa',
    'Benin': 'Africa',
    'Botswana': 'Africa',
    'Burkina Faso': 'Africa',
    'Burundi': 'Africa',
    'Cape Verde': 'Africa',
    'Cameroon': 'Africa',
    'Central African Republic': 'Africa',
    'Chad': 'Africa',
    'Comoros': 'Africa',
    'Congo': 'Africa',
    'Democratic Republic of the Congo': 'Africa',
    'Djibouti': 'Africa',
    'Egypt': 'Africa',
    'Equatorial Guinea': 'Africa',
    'Eritrea': 'Africa',
    'Eswatini': 'Africa',
    'Ethiopia': 'Africa',
    'Gabon': 'Africa',
    'Gambia': 'Africa',
    'Ghana': 'Africa',
    'Guinea': 'Africa',
    'Guinea-Bissau': 'Africa',
    'Ivory Coast': 'Africa',
    'Kenya': 'Africa',
    'Lesotho': 'Africa',
    'Liberia': 'Africa',
    'Libya': 'Africa',
    'Madagascar': 'Africa',
    'Malawi': 'Africa',
    'Mali': 'Africa',
    'Mauritania': 'Africa',
    'Mauritius': 'Africa',
    'Morocco': 'Africa',
    'Mozambique': 'Africa',
    'Namibia': 'Africa',
    'Niger': 'Africa',
    'Nigeria': 'Africa',
    'Rwanda': 'Africa',
    'Sao Tome and Principe': 'Africa',
    'Senegal': 'Africa',
    'Seychelles': 'Africa',
    'Sierra Leone': 'Africa',
    'Somalia': 'Africa',
    'South Africa': 'Africa',
    'South Sudan': 'Africa',
    'Sudan': 'Africa',
    'Tanzania': 'Africa',
    'Togo': 'Africa',
    'Tunisia': 'Africa',
    'Uganda': 'Africa',
    'Zambia': 'Africa',
    'Zimbabwe': 'Africa',
    'Afghanistan': 'Asia',
    'Armenia': 'Asia',
    'Azerbaijan': 'Asia',
    'Bahrain': 'Asia',
    'Bangladesh': 'Asia',
    'Bhutan': 'Asia',
    'Brunei': 'Asia',
    'Cambodia': 'Asia',
    'China': 'Asia',
    'Georgia': 'Asia',
    'Hong Kong': 'Asia',
    'India': 'Asia',
    'Indonesia': 'Asia',
    'Iran': 'Asia',
    'Iraq': 'Asia',
    'Israel': 'Asia',
    'Japan': 'Asia',
    'Jordan': 'Asia',
    'Kazakhstan': 'Asia',
    'Kuwait': 'Asia',
    'Kyrgyzstan': 'Asia',
    'Laos': 'Asia',
    'Lebanon': 'Asia',
    'Macau': 'Asia',
    'Malaysia': 'Asia',
    'Maldives': 'Asia',
    'Mongolia': 'Asia',
    'Myanmar': 'Asia',
    'Nepal': 'Asia',
    'North Korea': 'Asia',
    'Oman': 'Asia',
    'Pakistan': 'Asia',
    'Palestine': 'Asia',
    'Philippines': 'Asia',
    'Qatar': 'Asia',
    'Saudi Arabia': 'Asia',
    'Singapore': 'Asia',
    'South Korea': 'Asia',
    'Sri Lanka': 'Asia',
    'Syria': 'Asia',
    'Taiwan': 'Asia',
    'Tajikistan': 'Asia',
    'Thailand': 'Asia',
    'East Timor': 'Asia',
    'Turkmenistan': 'Asia',
    'United Arab Emirates': 'Asia',
    'Uzbekistan': 'Asia',
    'Vietnam': 'Asia',
    'Yemen': 'Asia',
    'Albania': 'Europe',
    'Andorra': 'Europe',
    'Austria': 'Europe',
    'Belarus': 'Europe',
    'Belgium': 'Europe',
    'Bosnia and Herzegovina': 'Europe',
    'Bulgaria': 'Europe',
    'Croatia': 'Europe',
    'Cyprus': 'Europe',
    'Czech Republic': 'Europe',
    'Denmark': 'Europe',
    'Estonia': 'Europe',
    'Finland': 'Europe',
    'France': 'Europe',
    'Germany': 'Europe',
    'Greece': 'Europe',
    'Hungary': 'Europe',
    'Iceland': 'Europe',
    'Ireland': 'Europe',
    'Italy': 'Europe',
    'Latvia': 'Europe',
    'Liechtenstein': 'Europe',
    'Lithuania': 'Europe',
    'Luxembourg': 'Europe',
    'Malta': 'Europe',
    'Moldova': 'Europe',
    'Monaco': 'Europe',
    'Montenegro': 'Europe',
    'Netherlands': 'Europe',
    'North Macedonia': 'Europe',
    'Norway': 'Europe',
    'Poland': 'Europe',
    'Portugal': 'Europe',
    'Romania': 'Europe',
    'Russia': 'Europe',
    'San Marino': 'Europe',
    'Serbia': 'Europe',
    'Slovakia': 'Europe',
    'Slovenia': 'Europe',
    'Spain': 'Europe',
    'Sweden': 'Europe',
    'Switzerland': 'Europe',
    'Turkey': 'Europe',
    'Ukraine': 'Europe',
    'United Kingdom': 'Europe',
    'Vatican City': 'Europe',
    'Antigua and Barbuda'
    'Bahamas': 'North America',
    'Barbados': 'North America',
    'Belize': 'North America',
    'Canada': 'North America',
    'Costa Rica': 'North America',
    'Cuba': 'North America',
    'Dominica': 'North America',
    'Dominican Republic': 'North America',
    'El Salvador': 'North America',
    'Grenada': 'North America',
    'Guatemala': 'North America',
    'Haiti': 'North America',
    'Honduras': 'North America',
    'Jamaica': 'North America',
    'Mexico': 'North America',
    'Nicaragua': 'North America',
    'Panama': 'North America',
    'Saint Kitts and Nevis': 'North America',
    'Saint Lucia': 'North America',
    'Saint Vincent and the Grenadines': 'North America',
    'Trinidad and Tobago': 'North America',
    'United States': 'North America',
    'Australia': 'Oceania',
    'Fiji': 'Oceania',
    'Kiribati': 'Oceania',
    'Marshall Islands': 'Oceania',
    'Micronesia': 'Oceania',
    'Nauru': 'Oceania',
    'New Zealand': 'Oceania',
    'Palau': 'Oceania',
    'Papua New Guinea': 'Oceania',
    'Samoa': 'Oceania',
    'Solomon Islands': 'Oceania',
    'Tonga': 'Oceania',
    'Tuvalu': 'Oceania',
    'Vanuatu': 'Oceania',
    'Argentina': 'South America',
    'Bolivia': 'South America',
    'Brazil': 'South America',
    'Chile': 'South America',
    'Colombia': 'South America',
    'Ecuador': 'South America',
    'Guyana': 'South America',
    'Paraguay': 'South America',
    'Peru': 'South America',
    'Suriname': 'South America',
    'Uruguay': 'South America',
    'Venezuela': 'South America'
}


def create_model():
    """
        This function creates a model and trains it. First there are steps for data manipulation and filtering.
        Then we create a pipeline and train it.
        :return: Trained pipeline object
    """
    imf_data = pd.read_pickle("IMF_DATA.pkl")

    def remove_predictions(row):
        DATA_END_DATE = 2025
        DATA_START_DATE = 1980

        prediction_start_year = int(row['Estimates Start After'])

        if prediction_start_year < DATA_START_DATE:
            row.loc[DATA_START_DATE: DATA_END_DATE] = 0
        elif prediction_start_year < DATA_END_DATE:
            row.loc[prediction_start_year + 1: DATA_END_DATE] = 0
        return row

    def country_to_continent(row):
        row['Continent'] = COUNTRY_CONTINENT_DIC.get(row['Country'], None)
        return row
    imf_data = imf_data.apply(remove_predictions, axis=1)

    # Fetching the GDP per capita data denoted in US dollars
    gdp_pattern = r'\bGross domestic product per capita.*'
    gdp_data = imf_data[imf_data['Subject Descriptor'].str.contains(gdp_pattern)]
    gdp_data = gdp_data[gdp_data['Units'].str.contains(r'U.S. dollars')]

    # Dropping all other columns apart from 'Country' and 1980 : 2025
    columns_to_be_dropped = ['WEO Country Code', 'ISO', 'WEO Subject Code', 'Subject Descriptor', 'Subject Notes',
                             'Units', 'Scale', 'Country/Series-specific Notes', 'Estimates Start After']
    gdp_data.drop(columns_to_be_dropped, axis=1, inplace=True)

    # Dropping rows that contain GDP related data
    gdp_pattern = r'\bGross domestic product.*'
    other_data = imf_data[imf_data['Subject Descriptor'].str.contains(gdp_pattern) == False]

    # Keeping only Units that I will use, as there are some Subject descriptors that have over 195 values
    # r'Index', r'U.S. dollars',
    units_to_be_dropped = [r'Missing']

    # Looping over units_to_be_dropped and dropping the rows that contain the expression
    for expression in units_to_be_dropped:
        other_data = other_data[other_data['Units'].str.contains(expression) == False]

    # Dropping columns in other_data that will be of no use
    columns_to_be_dropped = ['Subject Descriptor', 'WEO Country Code', 'ISO',
                             'Subject Notes', 'Country/Series-specific Notes', 'Estimates Start After', 'Units',
                             'Scale']
    other_data.drop(columns_to_be_dropped, axis=1, inplace=True)

    # Melting other_data to long format so the year can become a column
    other_data = other_data.melt(id_vars=['Country', 'WEO Subject Code'], var_name='Year', value_name='Value')

    # Pivoting other_data to make every subject code a seperate column and the resetting the index
    other_data = other_data.pivot_table(index=['Country', 'Year'], columns='WEO Subject Code',
                                        values='Value').reset_index()

    # Melting the gdp_data, so I can merge based on 'Country' and 'Year' columns
    gdp_data = gdp_data.melt(id_vars='Country', var_name='Year', value_name='GDP per capita')

    # Merging the data on 'Country' and 'Year' columns
    merged_data = other_data.merge(gdp_data, on=['Country', 'Year'], how='inner')

    # Discarding rows if GDP per capita is == 0
    merged_data = merged_data[merged_data['GDP per capita'] != 0]

    merged_data = merged_data.apply(country_to_continent, axis=1)

    # Features that we will use for the Endpoint
    features_final = ['LP', 'Continent', 'NGSD_NGDP', 'LE', 'BCA', 'GGR_NGDP', 'LUR', 'GGSB_NPGDP']

    # Replacing 0 values with nan
    merged_data.replace({0: float('nan')}, inplace=True)

    y_train = merged_data['GDP per capita']
    X_train = merged_data[features_final]

    categorical_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 and
                        X_train[cname].dtype == "object"]

    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = SimpleImputer()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    model = XGBRegressor(n_estimators=750, learning_rate=0.1)

    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                  ])

    # Preprocessing of training data, fit model
    my_pipeline.fit(X_train, y_train)

    return my_pipeline


def make_predictions(json_string, model):
    """
        This function receives parameters and the returns a prediction made by the model.
        :return: float64
    """
    # Important!
    # json string must include LP Continent NGSD_NGDP LE BCA GGR_NGDP LUR GGSB_NPGDP keys
    unordered_data = pd.read_json(json_string, orient='index')

    # Ordering the data because the model needs the data to be in this specific order to make accurate predictions
    order = ['LP', 'Continent', 'NGSD_NGDP', 'LE', 'BCA', 'GGR_NGDP', 'LUR', 'GGSB_NPGDP']
    X = unordered_data.reindex(columns=order)

    prediction = model.predict(X)

    return prediction
