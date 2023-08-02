# **ML task**
Machine Learning task, created for the purpose of learning
## General info
This project consist of two main parts:
1. ML tasks with data from IMF 2020 OCT.
2. HTTP API implementation with the model from part 1

The first part consist of 6 tasks which are described in the jupyter notebooks.\
\
The second part consist of Endpoint.py, Final_model.py, test.py.
Here we create a http endpoint that receives json data, 
creates a XGBRegressor model (with the help of Final_model.py) 
and predicts a countries GDP per capita. The test.py is just a unit test 
that checks if the Endpoint is running properly.

Features used in model, 'LP', 'Continent', NGSD_NGDP', 'LE', 'BCA', 'GGR_NGDP', 'LUR', 'GGSB_NPGDP'
(Population (Millions), continent, Gross national savings (Percent of GDP), Employment (Millions),
Current account balance (USD Billions), General government revenue (Percent of GDP),
Unemployment rate (Percent of total labor force), General government structural balance (Percent of potential GDP))
\
***Note: When sending request to Endpoint the json string has to have these labels, order doesn't matter***
### Setup
First please activate the conda environment with the line:
\
***conda env create --name myenv --file environment.yml \
conda activate myenv***
\
\
Now that we have the libraries we need we can run the ***setup.py*** script
that loads up the data. After that feel free to explore.

**Data : https://www.imf.org/en/Publications/WEO/weo-database/2020/October/download-entire-database**

