import mlflow
import pandas as pd

# Load model
model_uri = "models:/IrisClassifier/1"   # version 1
loaded_model = mlflow.sklearn.load_model(model_uri)


# Example prediction
sample = pd.DataFrame([[5.9, 3.0, 5.1, 1.8]],
                      columns=["sepal length (cm)", "sepal width (cm)",
                               "petal length (cm)", "petal width (cm)"])
print("Prediction:", loaded_model.predict(sample))
