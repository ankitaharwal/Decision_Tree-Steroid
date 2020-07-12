
#this file contain functions for impurity in decision tree
#which are in the format fn(y_pred,y_true), this function returns 
#impurity whose format is number(float or double) of a node
def gini_impurity(value):
    '''This function returns gini value for a node
    Ex: value={0:50,1:50,2:50}
    gini_impurity(value)==>0.6666
    p_values=[1/3,1/3,1/3]
    impurity=1-sigma(p_values**2)'''
    impurity=1
    p_values=list(value.values())
    sum_p_values=sum(p_values)
    for i in range(len(p_values)):
        p_values[i]/=sum_p_values
    for p in p_values:
        impurity-=p**2
    return impurity
def entropy_impurity(value):
    '''This function returns gini value for a node
    Ex: value={0:50,1:50,2:50}
    entropy_impurity(value)==>1.5849
    p_values=[1/3,1/3,1/3]
    impurity=-sigma(p_values*log2(p_values))'''
    import math
    impurity=0
    p_values=list(value.values())
    sum_p_values=sum(p_values)
    for i in range(len(p_values)):
        p_values[i]/=sum_p_values
    for p in p_values:
        impurity-=p*math.log2(p)
    return impurity
