import numpy as np

class Perceptron:
  def __init__(self, learning_rate, epochs):
    self.learning_rate = learning_rate
    self.epochs = epochs
    # initiliaze random weights
    self.weights = np.random.randn(3) * 1e-4 
    print(f"Initial weights before training: \n{self.weights}")
  
  def activationFunction(self, X, weights):
    z = np.dot(X, weights)
    return np.where(z>0, 1, 0)
  
  def fit(self, X, y):
    self.X = X
    self.y = y

    # prepare X with Bias term
    X_with_bias = np.c_[X, -np.ones((len(self.X), 1))]
    print(f"X with bias: \n{X_with_bias}")

    for epoch in range(self.epochs):
      print("----"*10)
      print(f"Epoch: {epoch}/{self.epochs}")
      print("----"*10)
      # Forward pass
      y_hat = self.activationFunction(X_with_bias, self.weights)
      print(f"Predicted values: \n{y_hat}")
      self.error = self.y - y_hat
      print(f"Errors: \n{self.error}")
      # Backpropogation
      self.weights = self.weights + self.learning_rate * np.dot(X_with_bias.T, self.error)
      print(f"Updated weights after epoch {epoch}/{self.epochs}: \n{self.weights}")
      print("======"*10)
  
  def predict(self, X):
    X_with_bias = np.c_[X, -np.ones((len(X), 1))]
    #print(f"Predict: X with bias: \n{X_with_bias}")
    return self.activationFunction(X_with_bias, self.weights)
  
  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"Total loss: {total_loss}")
    return total_loss