import numpy as np


def predict(neighbor_classes, C):
    # Make sure all classes are considered
    labels = np.concatenate((neighbor_classes, list(range(C))))
    # Find class frequency among neighbors
    weights = np.unique(labels, return_counts=True)[1]
    # Find most popular class
    prediction = np.argmax(weights)

    # If most popular class is ambiguous try with fewer neighbors; else return
    if sum(weights[prediction] == weights) > 1:
        return predict(neighbor_classes[:-2], C)
    else:
        return prediction


def knn(X_train, X_test, y_train, y_test, method, C, n_neighbors=7):
    # Number of classes
    n_classes = len(np.unique(y_train))

    prediction = []
    for doc in X_test:
        doc_to_train = [method(doc, x, C) for x in X_train.T]
        # Find indices of n_neighbors closest documents
        rank = np.argsort(doc_to_train)[:n_neighbors]

        # Make prediction based on most popular class among neighbors
        prediction.append(predict(rank, n_classes))

    # Print and return test error
    test_error = 1 - (prediction == y_test).mean()
    print(method + ' test error is %f' % test_error)
    return test_error
