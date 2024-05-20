import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Sample data (replace with your actual data)
data = {
    'X1': [30, 40, 12, 16,10,34,55,67,89,90,23,45,67,78,89],
    'X2': [1, 0, 1, 0,1,0,0,0,1,1,1,0,1,1,1],
    'Spam': [1, 0, 1, 0,1,0,0,0,1,1,1,0,1,1,1]
}

# Load data into a pandas DataFrame
df = pd.DataFrame(data)

# Separate features (X) and target variable (y)
#The double brackets around 'X1' and 'X2' are used to select multiple columns as a DataFrame, not a Series.
X = df[['X1', 'X2']]
y = df['Spam']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions (probabilities and class labels)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Predicted probabilities of spam class
y_pred_class = model.predict(X_test)  # Predicted class labels (0 or 1)

# Print predicted probabilities and class labels
print(f"Predicted spam probabilities: {y_pred_proba}")
print(f"Predicted spam classes (0: Not Spam, 1: Spam): {y_pred_class}")

# Example usage for new emails (assuming X1 and X2 features)
new_emails = [[20, 1], [50, 1], [20,0]]
new_pred_proba = model.predict_proba(new_emails)[:, 1]
new_pred_class = model.predict(new_emails)
print(f"\nPredicted spam probabilities for new emails: {new_pred_proba}")
print(f"Predicted spam classes for new emails: {new_pred_class}")
