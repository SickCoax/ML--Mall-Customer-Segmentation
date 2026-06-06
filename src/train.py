from preprocessing import get_X
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

def get_cluster(df) :

    X = get_X(df)

    model = AgglomerativeClustering(
        n_clusters =  6,
        linkage = "ward"
    )

    model.fit(X)

    return model , X