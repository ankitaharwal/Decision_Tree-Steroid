from Steroid.criteria import gini,entropy
from Steroid.impurity import gini_impurity,entropy_impurity
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from copy import deepcopy
import graphviz
import random
from scipy.stats import mode
from sklearn.metrics import classification_report,f1_score
class Node:
    def __init__(self,node_type=None,model=None,model_used=None,f_list=None,criteria=None,value=None,impurity=None,depth=0,importance=None):
        self.children_dict={}
        self.node_type=node_type
        self.model=model
        self.model_used=model_used
        self.f_list=f_list
        self.criteria=criteria
        self.value=value
        self.depth=depth
        self.impurity=impurity
        self.importance=importance #normalized
    def node_predict(self,X,proba=False):
        if (self.node_type!="model"):
            if proba:
                y_pred=[self.value for i in range(X.shape[0])]
            else:
                y_pred=np.array([int(self.node_type.split("_")[0]) for i in range(X.shape[0])])
            return y_pred
        else:
            X_new=np.take(a=X,axis=1,indices=self.f_list)
            y_pred=self.model.predict(X_new)
            return y_pred
