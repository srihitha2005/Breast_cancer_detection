class MpNeuron:

    def __init__(self):
        self.b = None

    def model(self, x):
        return (sum(x) >= self.b)

    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)
    def fit(self, X, Y):
        accuracy = {}
        for b in range(X.shape[1] + 1):
            self.b = b
            Y_pred = self.predict(X)
            accuracy[b] = accuracy_score(Y, Y_pred)
        best_b = max(accuracy, key=accuracy.get)
        self.b = best_b

        print('Optimal value of b is ', best_b)
        print('Highest accuracy is ', accuracy[best_b])
