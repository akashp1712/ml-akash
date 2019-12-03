import pandas as pd

## Getting the data ##

# save filepath to variable for easier access
melbourne_file_path = 'melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
print(melbourne_data.describe())

# print a columns names (features) of the data in Melbourne data
print(melbourne_data.columns)

# drop the data with the missing values
# dropna drops missing values
melbourne_data = melbourne_data.dropna(axis=0)


## Choosing features ##

# select the prediction target
y = melbourne_data.Price

# choosing features
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]
print(X.describe())
print(X.head())


## Building your model ##

'''
Main steps:
1. Define: What type of model will it be? A decision tree? 
    Some other type of model? Some other parameters of the model type are specified too.
2. Fit: Capture patterns from provided data. This is the heart of modeling.
3. Predict: Just what it sounds like
4. Evaluate: Determine how accurate the model's predictions are.
'''


from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
print(melbourne_model.fit(X,y))

# Let's test on training data itself

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))


## Model Evaluation ##

from sklearn.metrics import mean_absolute_error
predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

# validation data
