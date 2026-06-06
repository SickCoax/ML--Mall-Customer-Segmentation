from sklearn.metrics import silhouette_score

def evaluate_model(model , X) :

    label = model.labels_

    score = silhouette_score(X , label)

    return score