import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

kernel ="linear"
random_state = 42
# load data 
df = pd.read_csv('iris_data.csv')

def eval_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')       # or 'micro', 'weighted'
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return accuracy, f1, precision, recall

# set feature and target as X and Y
X = df.drop(['species'], axis=1).values
Y = df['species'].values

x_train,x_test,y_train,y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# create a iris experiment
iris_xp = mlflow.set_experiment("Iris dataset experiment")
with mlflow.start_run(experiment_id=iris_xp.experiment_id):
    
    # train model
    svc = SVC(random_state=random_state, kernel=kernel)
    svc.fit(x_train,y_train)

y_pred = svc.predict(x_test)

accuracy,f1,precision,recall = eval_model(y_test,y_pred)
# log the metrics and model 
mlflow.sklearn.log_model(svc,"model",registered_model_name="IrisClassifier")
mlflow.log_metrics({"Accuracy":accuracy,
                    "F1 ":f1,
                    "Precision":precision,
                    "Recall":recall})
mlflow.log_params({'Random state':random_state,
                  'Kernel':kernel})

print(f"SVC: kernel={kernel},random state={random_state}")
print(f"Accuracy Score: {round(accuracy,3)}")
print(f"F1 Score: {round(f1,3)}")
print(f"Precision Score: {round(precision,3)}")
print(f"Recall Score: {round(recall,3)}")
    
