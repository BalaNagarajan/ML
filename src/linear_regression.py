import pandas as pd
from sklearn.linear_model import LinearRegression



# Create a sample data set
data = {'House in Square feet (X)': [1000,2000,3000,4000,5000], 'Price (Y)': [200000,350000,500000,550000,600000]}

# Define the data as a dictionary with two keys: 'House in Square feet (X)' and 'Price (Y)'
# Each key corresponds to a list of values representing the square footage and price of houses

# Create a pandas DataFrame from the data dictionary
df = pd.DataFrame(data)

# Separate features(X) and target variable (Y)
# Assign the 'House in Square feet (X)' column to X and the 'Price (Y)' column to Y
X = df[['House in Square feet (X)']]
Y = df['Price (Y)']

# Assign column names to X
X.columns = ['House in Square feet (X)']

# Create an instance of the LinearRegression model
model = LinearRegression()

# Fit the model using X and Y
model.fit(X, Y)

# Define a new_area variable with a value of 2500
new_area = 2500

# Predict the price of a house with new area square feet
predict_price = model.predict([[new_area]])

# Print the predicted price
print(predict_price)


