import numpy as np
from tqdm import tqdm

class KNN:
    def __init__(self, k):
        self.k = k
        
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    # training
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y 
        
    def predict(self, X):
        Y = []
        for x in tqdm(X):
            distances = []
            for x_train in self.X_train:
                d = self.euclidean_distance(x, x_train)
                distances.append(d)
                
            nearest_neighbors = np.argsort(distances)[:self.k]
            result = np.bincount(self.Y_train[nearest_neighbors])
            y = np.argmax(result)
            Y.append(y)
        
        return Y 
    
    def show_fruit(self, outputs):
        for output in outputs:
            if output == 0:
                print('🍎')
            elif output == 1:
                print('🍌')
            elif output == 2:
                print('🍉')
                
    def show_person(self, outputs):
        for output in outputs:
            if output == 0:
                print('👩🏽')
            elif output == 1:
                print('👨🏽')
                
    # Test
    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        accuracy = np.sum(Y_pred == Y) / len(Y)
        return accuracy