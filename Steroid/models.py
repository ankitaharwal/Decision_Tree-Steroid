from Steroid.utils import *
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categories="auto")
class dt_classifier:
    '''
    max_depth : maximum depth the tree can have 
    ex >    model=dt_classifier(max_depth=3,.....)

    criteia : which criteria to use for choosing between different classifiers for same node
    always we will try to minimize criteria hence for criteria like accuracy we will add -ve sign
    we can also choose custom criteria by this steps or example >
            from sklearn.metrics import accuracy_score
            model=dt_classifier(criteria=-accuracy_score,......)

    no_features : list of number of combinations of features we want to try
    suppose no_features=[2,3] and there are three features x1,x2,x3 then find curresponding
    criteria using (x1,x2) (x2,x3) (x3,x1) and (x1,x2,x3).Then suppose (x1,x2,x3) has minimum
    criteria value it will classify node based on this.
    Note: It will also try all other criteria_models also hence this is only true for only 1
    criteria_models, in case of more than 1 criteria models ex > M1(),M2()
    it would try (x1,x2) ==> (x1,x2,M1()) , (x1,x2,M2()) ....... and so on.
    ex >    model=dt_classifier(suppose no_features=[2,3],......)

    criteria_models : dict of "model_name":model() to try to fit on single node.
    suppose criteria_models={"n1":M1(),"n2":M2(),....}
    Mi() can be any classifier and ni is its name.It is recommended that try to choose criteria
    of Mi() and criteria our main model to be same. 
    ex1 >   from sklearn.tree import DecisionTreeClassifier
            from Steroid.criteria import gini
            model=dt_classifier(criteria=gini,
            criteria_models={"default":DecisionTreeClassifier(criterion=gini.__name__,max_depth=1)},....)
    ex2 >   from sklearn.metrics import log_loss
            from sklearn.linear_model import LogisticRegression
            from sklearn.svc import LinearSVC
            model=dt_classifier(criteria=log_loss,
            criteria_models={"lr":LogisticRegression(),"svc":LinearSVC()},......)

    random_state : random_state for every algorithm inside
    ex >    model=dt_classifier(random_state=42,.....)
    '''
    def __init__(self,
    max_depth=None,
    min_samples=5,
    criteria=gini, 
    impurity=gini_impurity,
    no_features=[1],
    criteria_models={"DT":DecisionTreeClassifier(criterion=gini.__name__,max_depth=1)},
    random_state=None,
    score_fn=f1_score
    ):
        self.max_depth=max_depth
        self.min_samples=min_samples
        self.criteria=criteria
        self.impurity=impurity
        self.no_features=no_features
        self.criteria_models=criteria_models
        self.random_state=random_state
        if random_state!=None:
            random.seed(random_state)
            np.random.seed(random_state)
        self.node_root=Node()
        self.total_importance=0
        self.predict_class=None
        self.features_=None
        self.feature_importances_=None
        self.score_fn=score_fn
    def combinationUtil(self,f_arr, temp_data, start,  end, index, features,f_list_total): 
        if (index == features):
            f_list=[] 
            for j in range(features):
                f_list.append(temp_data[j])
            f_list_total.append(f_list)
            return
        i = start  
        while(i <= end and end - i + 1 >= features - index): 
            temp_data[index] = f_arr[i]
            self.combinationUtil(f_arr, temp_data, i + 1, end, index + 1, features,f_list_total)
            i += 1
    def fit(self,X,y,node=None,depth=0):
        if node==None:
            node=self.node_root
            self.predict_class=list(set(y))
            self.features_=X.shape[1]
            self.feature_importances_=[0 for i in range(self.features_)]
        node.depth=depth+1
        if node.depth==1:
            self.total_importance=0
        y_set=list(set(y))
        
        y_dict_count={i:np.count_nonzero(y==i) for i in y_set}
        node.value=y_dict_count
        node.samples=sum(list(node.value.values()))
        node.impurity=self.impurity(node.value)
        max_depth_reached=False
        if self.max_depth!=None:
            if node.depth==self.max_depth+1:
                max_depth_reached=True
        min_samples_occured=False
        if(self.min_samples!=None):
            if(node.samples<self.min_samples):
                min_samples_occured=True
        if len(y_set)==1 or max_depth_reached or min_samples_occured:
            node.node_type=str(mode(y)[0][0])+"_leaf"
            return
        self.fit_node(X,y,node)
        self.total_importance+=node.importance
        try:
            node_features_importance=[i*node.importance for i in node.model.feature_importances_]
        except:
            print(f"{node.model_used} doesnt has feature_importances_ so each feature approxed by 1/no_of_features")
            node_features_importance=[1/len(node.f_list)*node.importance for i in range(len(node.f_list))]
        for i in range(len(node.f_list)):
            self.feature_importances_[node.f_list[i]]+=node_features_importance[i]
        y_pred=node.node_predict(X)
        #if node classifier has only one predicted class
        if len(set(y_pred))==1:
            node.node_type="model"
            children_list=sorted(list(set(y_pred)))
            
            for child in children_list:
                y_new=np.array([child for i in range(y_pred.shape[0])])
                node.children_dict[child]=Node()
                self.fit(X,y_new,node.children_dict[child],depth+1)
            return
        children_list=sorted(list(set(y_pred)))
        # print(children_list)
        for child in children_list:
            X_new=X[y_pred==child]
            y_new=y[y_pred==child]
            node.children_dict[child]=Node()
            self.fit(X_new,y_new,node.children_dict[child],depth+1)
    def fit_node(self,X,y,node):
        f=X.shape[1]
        f_list_total=[]
        for features in self.no_features:
            f_arr = [i for i in range(f)]
            n = len(f_arr)
            temp_data = [0]*features
            self.combinationUtil(f_arr, temp_data, 0,n - 1, 0, features,f_list_total)
        
        model_list=list(self.criteria_models.keys())
        if self.random_state!=None:
            random.shuffle(f_list_total)
            random.shuffle(model_list)
        models_dict={}
        models_performance_dict={}
        models_f_list_dict={}
        models_used_dict={}
        for f_list in f_list_total:
            for model in model_list:
                model_name='_'.join([str(i) for i in f_list])+"_"+model
                models_dict[model_name]=deepcopy(self.criteria_models[model])
                X_new=np.take(a=X,axis=1,indices=f_list)
                models_dict[model_name].fit(X_new,y)
                y_pred=models_dict[model_name].predict(X_new)
                models_performance_dict[model_name]=self.criteria(y_pred=y_pred,y_true=y)

                models_f_list_dict[model_name]=f_list
                models_used_dict[model_name]=model

        best_performance=min(list(models_performance_dict.values()))
        for key,value in models_performance_dict.items():
            if value==best_performance:
                node.node_type="model"
                node.model=models_dict[key]
                node.model_used=models_used_dict[key]
                node.f_list=models_f_list_dict[key]
                node.criteria=models_performance_dict[key]
                node.importance=(node.impurity-node.criteria)*node.samples/self.node_root.samples
                break
    def predict(self,X):
        n=X.shape[0]
        y_pred=[]
        for i in range(n):
            y_pred.append(self.predict_single(X[i:i+1]))
        y_pred=np.array(y_pred)
        return y_pred
    def predict_proba(self,X):
        n=X.shape[0]
        y_pred=[]
        for i in range(n):
            y_pred.append(self.predict_single_proba(X[i:i+1]))
        y_pred=np.array(y_pred)
        return y_pred
    def predict_single(self,x,node=None):
        if(node==None):
            node=self.node_root
        if node.node_type!="model":
            # print(node.node_predict(x))
            return node.node_predict(x)[0]
        y_pred=node.node_predict(x)
        # print(max(y_pred),y_pred[0])
        new_node=node.children_dict[y_pred[0]]
        return self.predict_single(x,new_node)
    def predict_single_proba(self,x,node=None):
        if(node==None):
            node=self.node_root
        if node.node_type!="model":
            node_predict_proba=node.node_predict(x,proba=True)[0]
            temp=[0 for i in self.predict_class]
            value_sum=sum(node_predict_proba.values())
            for key,value in node_predict_proba.items():
                temp[key]=value/value_sum
            return temp
        y_pred=node.node_predict(x)
        new_node=node.children_dict[y_pred[0]]
        return self.predict_single_proba(x,new_node)
    def score(self,X,y):
        y_pred=self.predict(X)
        value=self.score_fn(y_pred=y_pred,y_true=y,average="micro")
        return value
    def feature_importances_calc(self,normalized=True):
        if normalized:
            return np.array(self.feature_importances_)/self.total_importance
        else:
            return np.array(self.feature_importances_)
    def export_edges(self,node,l,depth=0,index=0,name="D"):
        if(node.node_type!="model"):
            return
        for child_key,child_value in node.children_dict.items():
            index+=1
            i_node=str(name)
            rel=str(child_key)
            print(rel)
            f_node=child_value.model_used
            if (f_node==None):
                f_node=str(i_node)+str(rel)
            else:
                f_node=str(i_node)+str(rel)
            l.append([[i_node,node.model_used,node.criteria,node.f_list,node.value,node.depth,node.samples,node.impurity,node.importance],[rel],[f_node,rel,child_value]])
            self.export_edges(child_value,l,depth+1,index,name+str(child_key))
    def export_graph(self,show_model=True,show_criteria=True,criteria_digits=2,show_features=True,show_value=True,show_samples=True,show_depth=True,show_impurity=True,show_leaves=True,show_importance=True,importance_type="percentage"):
        l=[]
        self.export_edges(self.node_root,l)
        graph=graphviz.Digraph(name="Decision Tree Steroid",node_attr={'color': 'lightblue2', 'style': 'filled','fontsize':'12'},engine="dot")
        node_dict={i[2][0] :[i[2][1],i[2][2]] for i in l if i[2][2].node_type!="model"}
        for key,value in node_dict.items():
            label=""
            label=label+f"leaf : {value[0]}\n"
            if(show_leaves):
                
                if(show_impurity):
                    label=label+f"impurity : {str(value[1].impurity)[:criteria_digits+2]}\n"
                if(show_samples):
                    label=label+f"samples : {value[1].samples}\n"
                if(show_value):
                    label=label+f"value : {value[1].value}\n"
                if(show_depth):
                    label=label+f"depth : {value[1].depth}\n"
                
            graph.node(key,label=label,color="black",fillcolor="lightblue2",shape="box")
        node_dict={i[0][0]:[i[0][1],i[0][2],i[0][3],i[0][4],i[0][5],i[0][6],i[0][7],i[0][8]] for i in l}
        for key,value in node_dict.items():
            label=""
            if(show_model):
                label=label+f"model : {value[0]}\n"
            if(show_features):
                label=label+"features : "+",".join([str(i) for i in value[2]])+"\n"
            if(show_impurity):
                label=label+f"impurity : {str(value[6])[:criteria_digits+2]}\n"
            if(show_samples):
                label=label+f"samples : {value[5]}\n"
            if(show_value):
                label=label+f"value : {value[3]}\n"
            if(show_criteria):
                label=label+f"criteria : {str(value[1])[:criteria_digits+2]}\n"
            
            if(show_depth):
                label=label+f"depth : {value[4]}\n"
            if(show_importance):
                if(importance_type=="percentage") and self.total_importance!=0:
                    label=label+f"importance : {str(value[7]*100/self.total_importance)[:criteria_digits]}%\n"
                else:
                    label=label+f"importance : {str(value[7])[:criteria_digits+2]}\n"
            graph.node(key,label=label,fillcolor="green",color="black",shape="box")
        
        for i in l:
            label=f"{i[1][0]}"
            graph.edge(i[0][0],i[2][0],label=label,constraint="True",fontsize="12")
        return graph
class dt_node:
    def __init__(self,criteria=gini):
        self.criteria=criteria
        self.feature_importances_=None
        self.model=None
    def fit(self,X,y):
        d_criteria=[]
        d_criteria_columns=["criteria","feature value","y_arr index","1 means >= and 0 means <"]
        #best_index is best feature split
        y_arr=ohe.fit_transform(y.reshape(-1,1)).toarray().T.astype(int)
        for i in X.T:
            criteria_i=[]
            set_i=sorted(list(set(i)))
            if len(set_i)==1:
                split_v=set_i[0]
                y_pred=(i==split_v).astype(int)
                for y_set in range(y_arr.shape[0]):
                    criteria_i.append([self.criteria(y_pred,y_arr[y_set]),split_v,y_set,1])
            else:
                for j in range(len(set_i)-1):
                    split_v=(set_i[j]+set_i[j+1])/2
                    y_pred=(i>=split_v).astype(int)
                    for y_set in range(y_arr.shape[0]):
                        greater=((y_pred==y_arr[y_set]).astype(int).mean())
                        smaller=((1-y_pred==y_arr[y_set]).astype(int).mean())
                        if greater>=smaller:
                            criteria_i.append([self.criteria(y_pred,y_arr[y_set]),split_v,y_set,1])
                        else:
                            criteria_i.append([self.criteria(1-y_pred,y_arr[y_set]),split_v,y_set,0])
            criteria_i=np.array(criteria_i)
            split_index=(np.where(criteria_i.T[0]==criteria_i.T[0].min())[0][0])
            d_criteria.append(criteria_i[split_index])
        d_criteria=np.array(d_criteria)
        best_index=np.where(d_criteria.T[0]==d_criteria.T[0].min())[0][0]
        model={"feature":best_index,"value":d_criteria[best_index][1],"y_arr_index":int(d_criteria[best_index][2]),"x>=value":int(d_criteria[best_index][3])}
        y_set=list(set(y))
        y_l=list(y)
        model_l=[y_l.count(i) for i in y_set]
        model_l[model["y_arr_index"]]=0
        other_pred=model_l.index(max(model_l))
        model["y_arr_index_other"]=other_pred
        self.feature_importances_=[0 for i in range(X.shape[1])]
        self.feature_importances_[best_index]=1
        self.model=model
        print(self.model)
    def predict(self,X):
        model=self.model
        if model["x>=value"]==1:
            y_pred=(X.T[model["feature"]]>=model["value"])
        else:
            y_pred=(X.T[model["feature"]]<=model["value"])
        y_pred_new=y_pred.astype(str)
        y_pred_new[y_pred_new=="True"]=model["y_arr_index"]
        y_pred_new[y_pred_new=="False"]=model["y_arr_index_other"]
        return y_pred_new.astype(int)