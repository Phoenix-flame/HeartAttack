import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score



class Dataset:
    def __init__(self, path):
        self.path = path
        self.data = None
        self.label = None
        self.load()


    def load(self):
        self.data = pd.read_csv(self.path)
        self.data = np.array(self.data)
        self.label = self.data[:, -1]
        self.data = self.data[:, :-1]
        
        idx = np.random.choice(len(self.data), replace=True, size=int(0.2*len(self.data)))
        self.test_data = self.data[idx]
        self.test_label = self.label[idx]
        
        rest_idx = np.arange(0, len(self.data))
        rest_idx = np.delete(rest_idx, idx)

        self.data = self.data[rest_idx]
        self.label = self.label[rest_idx]
        
    
    def bootstrap(self, k=5, size=150):
        res_data = []
        res_label = []
        for _ in range(k):
            idx = np.random.choice(len(self.data), replace=True, size=size)
            res_data.append(self.data[idx])
            res_label.append(self.label[idx])
        return res_data, res_label            

        
class Bagging:
    def __init__(self, dataset, size=5):
        # Hyperparameters
        self.jungle_size = size
        
        # Other stuff
        self.dataset = dataset
        self.data, self.label = self.dataset.bootstrap(k=size)
        self.jungle = []
    
    def fit(self):
        for i in range(self.jungle_size):
            clf = DecisionTreeClassifier(random_state=0)
            tree = clf.fit(self.data[i], self.label[i], sample_weight=None, check_input=True, X_idx_sorted=None)
            self.jungle.append(tree)
    
    def predict(self):
        pass
    
    def get_accuracy(self):
        res = []
        for i in range(self.jungle_size):
            res.append(self.jungle[i].predict(self.dataset.test_data))
        res = np.array(res)
        print(res)
        tmp = []
        for i in res.T:
            u, c = np.unique(i, return_counts=True)
            max_count = np.argmax(c)
            tmp.append(u[max_count])
        print((np.array(tmp) == self.dataset.test_label).mean())
        
    


dataset = Dataset("./dataset/heart.csv")








bagging = Bagging(dataset)

bagging.fit()
bagging.get_accuracy()