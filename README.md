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

