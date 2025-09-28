import pandas as pd
from sklearn.datasets import load_iris

# 1. Load the Iris dataset
iris = load_iris(as_frame=True) # as_frame=True returns a pandas DataFrame

iris_df = iris.frame


print("Combined Iris DataFrame:")
print(iris_df.head())
quires creating the DataFrame from scratch

# 1. Load the Iris dataset without returning as a DataFrame
iris_data = load_iris()

# 2. Create a DataFrame for the features
features_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)

# 3. Add the target column
features_df['species'] = iris_data.target
features_df.to_csv("iris_data.csv", index=False)
# 4. Print the first 5 rows to confirm
print("\nAlternative combined Iris DataFrame:")
print(features_df.head())

