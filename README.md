# Decision-Tree-Steroid
full implementation of decision tree with flexible impurity function and for node to be any machine learning classifier such as logistic regression, svm, as well gini and entropy impurity.
# requirements
1. numpy
2. graphviz
3. scipy
4. sklearn
# How to run ?
1. import important files and setup <br>
from Steroid.models import dt_classifier,dt_node <br>
from Steroid.criteria import gini,entropy<br>
from Steroid.impurity import gini_impurity,entropy_impurity<br>
from sklearn.tree import DecisionTreeClassifier<br>
from sklearn.linear_model import LogisticRegression<br>
2. Import data in X,y as numpy array<br>
from sklearn.datasets import load_iris<r>
data=load_iris()<br>
X=data.data<br>
y=data.target<br>
3. Initialize model<br>
clf=dt_classifier(no_features=[X.shape[1]],<br>
                criteria_models={<br>
                        "Decision Tree entropy":DecisionTreeClassifier(criterion="entropy",max_depth=1,random_state=random_state),<br>
                                             "Logistic Regresion":LogisticRegression(C=1,solver="lbfgs",multi_class="auto",max_iter=1000),<br>
                                             }<br>
4.  fit data into model<br>
  clf.fit(X_train,y_train)<br>
5. predicting target<br>
  y_pred=clf.predict(X_train)<br>
# View graph<br>
graph=clf.export_graph(criteria_digits=4,show_features=False,show_value=True,show_depth=True,show_samples=True,show_leaves=True,show_criteria=True,show_model=True,show_importance=True,importance_type="percentage")<br>
  graph<br>
  <img src="./decision_tree.png">
# some other screenshot of classification
  <img src="./working.png">
  <img src="./working2.png">
