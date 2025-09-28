import requests

data = {
    "columns": ["sepal length (cm)", "sepal width (cm)",
                "petal length (cm)", "petal width (cm)"],
    "data": [[5.9, 3.0, 3.1, 1.8]]
}

res = requests.post("http://127.0.0.1:8009/predict", json=data)
print(res.json())
