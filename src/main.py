import os
import pandas as pd
from train import get_cluster
from evaluation import evaluate_model

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "Mall_Customers.csv"
)

df = pd.read_csv(csv_path)

model , X = get_cluster(df)

score = evaluate_model(model , X)

print(f"Silhouette Score : {score}")