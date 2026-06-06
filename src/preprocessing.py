import pandas as pd
from sklearn.preprocessing import StandardScaler

def get_X(df) :

    X = df[["Annual Income (k$)", "Spending Score (1-100)" , "Age"]]

    scaler = StandardScaler()
    X_scaled_arr = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled_arr , columns = scaler.get_feature_names_out()) 

    return X_scaled_df