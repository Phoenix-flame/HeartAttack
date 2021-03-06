import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


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
        
        self.name = 'Bagging'
    
    def fit(self):
        for i in range(self.jungle_size):
            clf = DecisionTreeClassifier(random_state=0)
            tree = clf.fit(self.data[i], self.label[i], sample_weight=None, check_input=True, X_idx_sorted=None)
            self.jungle.append(tree)
    
    def predict(self):
        res = []
        for i in range(self.jungle_size):
            res.append(self.jungle[i].predict(self.dataset.test_data))
        res = np.array(res)
        pred = []
        for i in res.T:
            u, c = np.unique(i, return_counts=True)
            max_count = np.argmax(c)
            pred.append(u[max_count])
        return np.array(pred)
    
    def get_accuracy(self):
        pred = self.predict()
        acc = (pred == self.dataset.test_label).mean()
        print(self.name, ' -> ', acc * 100)
        return acc
    
    def confusion_matrix(self):
        pred = self.predict()
        cm = confusion_matrix(self.dataset.test_label, pred)
        plt.title(self.name + " Confusion Matrix")
        sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)
        plt.show()
        
    

class RandomForest:
    def __init__(self, dataset, size=5, n_features=5):
        # Hyperparameters
        self.jungle_size = size
        self.n_features = n_features
        
        # Other Stuff
        self.dataset = dataset
        self.data, self.label = self.dataset.bootstrap(k=size)
        self.jungle = []
        self.random_features = []
        
        self.name = 'Random Forest'
        
        
    def fit(self):
        for i in range(self.jungle_size):
            clf = DecisionTreeClassifier(random_state=0)
            random_features = np.random.choice(self.data[0].shape[1], replace=False, size=self.n_features)
            # print(random_features)
            self.random_features.append(random_features)
            tree = clf.fit(self.data[i][:, random_features], self.label[i], sample_weight=None, check_input=True, X_idx_sorted=None)
            self.jungle.append(tree)
    
    def predict(self):
        res = []
        for i in range(self.jungle_size):
            res.append(self.jungle[i].predict(self.dataset.test_data[:, self.random_features[i]]))
        res = np.array(res)
        pred = []
        for i in res.T:
            u, c = np.unique(i, return_counts=True)
            max_count = np.argmax(c)
            pred.append(u[max_count])
        return np.array(pred)
    
    def get_accuracy(self):
        pred = self.predict()
        acc = (pred == self.dataset.test_label).mean()
        print(self.name, " -> ", acc * 100)
        return acc
    
    def confusion_matrix(self):
        pred = self.predict()
        cm = confusion_matrix(self.dataset.test_label, pred)
        plt.title(self.name + " Confusion Matrix")
        sns.heatmap(cm,annot=True,cmap="Blues",fmt="d",cbar=False)
        plt.show()
        
        
dataset = Dataset("./dataset/heart.csv")








bag = Bagging(dataset)
bag.fit()
# bag.confusion_matrix()
bag.get_accuracy()


rf = RandomForest(dataset)
rf.fit()
# rf.confusion_matrix()
rf.get_accuracy()
