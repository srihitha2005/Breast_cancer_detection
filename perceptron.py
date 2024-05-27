class Perceptron:

  def __init__(self):
    self.w = None
    self.b = None

  def model(self, x):
    return 1 if (np.dot(self.w, x) >= self.b) else 0

  def predict(self, X):
    Y = []
    for x in X:
      result = self.model(x)
      Y.append(result)
    return np.array(Y)

  def fit(self, X, Y,epochs,lr):
    self.w = np.ones(X.shape[1])
    self.b = 0
    for i in range(epochs):
      for x, y in zip(X, Y):
        y_pred = self.model(x)
        if y == 1 and y_pred == 0:
          self.w += lr*x
          self.b += lr*1
        elif y == 0 and y_pred == 1:
          self.w -= lr*x
          self.b -= lr*1
