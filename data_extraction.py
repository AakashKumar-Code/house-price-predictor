from sklearn.datasets import fetch_california_housing
import pandas as pd
import ssl

# Ignore SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

# Load the dataset
housing = fetch_california_housing()

# Create a DataFrame
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['MedHouseVal'] = housing.target

# Save the DataFrame to a CSV file
data.to_csv('housing.csv', index=False)

print("Dataset saved to housing.csv")
