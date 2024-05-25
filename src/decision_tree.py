import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# File path to the Iris CSV file
parent_path = os.path.dirname(os.getcwd())
medication_type_path = os.path.join(parent_path, "data", "drug200.csv")


# Load the CSV file into a Pandas DataFrame
df = pd.read_csv(medication_type_path)
print(df.head())

# Feature columns
features = ['Age','Sex','BP','Cholesterol','Na_to_K']

# Encode categorical features using one-hot encoding
categorical_features = ['Sex', 'BP', 'Cholesterol']

# Target column (class labels)
target = 'Drug'

#Handle the categorical features
#Converting to numeric representation - # Sex - F - 1 , M - 0 | BP - HIGH - 0, LOW - 1, NORMAL - 2 | Cholestrol - HIGH - 0, NORMAL - 1
sex_data = LabelEncoder()
df['Sex'] = sex_data.fit_transform(df['Sex'])

bp_data = LabelEncoder()
df['BP'] = sex_data.fit_transform(df['BP'])

cholestrol_data = LabelEncoder()
df['Cholesterol'] = sex_data.fit_transform(df['Cholesterol'])
print(df.head())

# Split data into training and testing sets (optional, for model evaluation)
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Create the decision tree classifier
clf = DecisionTreeClassifier()

# Train the model on the training data
clf.fit(X_train, y_train)

# Evaluate model performance (optional)
accuracy = accuracy_score(y_test, clf.predict(X_test))
print("Accuracy:", accuracy)

# Function to make predictions for new data
def predict_drug_type(new_data):
  """
  This function takes new data (pandas DataFrame) and predicts the drug type(s).

  Args:
      new_data: A pandas DataFrame with the same features as the training data.

  Returns:
      A pandas Series containing the predicted drug type(s) for each row in the new data.
  """
  predictions = clf.predict(new_data)
  return pd.Series(predictions, name='Predicted Drug Type')

# Example usage: predict drug type for a new data point (replace with your data)
# Sex - F - 1 , M - 0 | BP - HIGH - 0, LOW - 1, NORMAL - 2 | Cholestrol - HIGH - 0, NORMAL - 1
new_data = pd.DataFrame({'Age': [35], 'Sex': [0], 'BP': [2], 'Cholesterol': [1], 'Na_to_K': [1.4]})
predictions = predict_drug_type(new_data)

print("Predicted drug type for the new data:")
print(predictions)
