import numpy as np
from typing import List
from hw1_knn import KNN

# TODO: Information Gain function
def Information_Gain(S, branches):
    # S: float
    # branches: List[List[int]] num_branches * num_cls
    # return: float
    total_entropy = 0
    total_entries = np.sum(branches)
    for child in branches:
        child_entropy = 0
        child_total = sum(child)
        for k in range(len(child)):
            if(child[k] != 0):
                entropy = -1*(child[k]/child_total) * np.log2(child[k]/child_total)
                child_entropy += entropy
        total_entropy += (child_total/total_entries) * child_entropy
    return S - total_entropy
    #raise NotImplementedError
    
# TODO: implement reduced error prunning function, pruning your tree on this function
def reduced_error_prunning(decisionTree, X_test, y_test):
    # decisionTree
    # X_test: List[List[any]]
    # y_test: List
    predicted_labels = decisionTree.predict(X_test)
    initial_accuracy = decisionTree.accuracy(predicted_labels, y_test)
    decisionTree.pruneTree(decisionTree.root_node, X_test, y_test, initial_accuracy,1)
    return  decisionTree
    #raise NotImplementedError


# print current tree
def print_tree(decisionTree, node=None, name='branch 0', indent='', deep=0):
    if node is None:
        node = decisionTree.root_node
    print(name + '{')

    print(indent + '\tdeep: ' + str(deep))
    string = ''
    label_uniq = np.unique(node.labels).tolist()
    for label in label_uniq:
        string += str(node.labels.count(label)) + ' : '
    print(indent + '\tnum of samples for each class: ' + string[:-2])

    if node.splittable:
        print(indent + '\tsplit by dim {:d}'.format(node.dim_split))
        for idx_child, child in enumerate(node.children):
            print_tree(decisionTree, node=child, name='\t' + name + '->' + str(idx_child), indent=indent + '\t', deep=deep+1)
    else:
        print(indent + '\tclass:', node.cls_max)
    print(indent + '}')


#TODO: implement F1 score
def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    #assert len(real_labels) == len(predicted_labels)
    #fp = 2*precision*recall/precision + recall
    #precision = tp/tp+fp
    #recall = tp/tp+fn
    #real_labels = [1,0,0] #predicted_labels = [1,1,0]
    tp = 0
    fp = 0
    fn = 0
    for a,b in zip(real_labels, predicted_labels):
        if (a == 1 and b == 1) :
            tp += 1
        elif(a == 0 and b == 1):
            fp += 1
        elif (a == 1 and b == 0):
            fn +=1
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)

    if precision == 0 or recall == 0:
        f1_score = 0
    else:
        f1_score = 2*precision*recall/(precision + recall)
    return f1_score
    #raise NotImplementedError

#TODO:
def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    distance = 0
    for a,b in zip(point1, point2):
        distance += (a-b)**2
    return np.sqrt(distance)
    #raise NotImplementedError


#TODO:
def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    distance = 0
    for a, b in zip(point1, point2):
        distance += (a*b)
    return distance
    #raise NotImplementedError


#TODO:
def gaussian_kernel_distance(point1: List[float], point2: List[float]) -> float:
    distance = 0
    for a, b in zip(point1, point2):
        distance += (a-b)**2
    distance = -0.5*distance
    distance = -np.exp(distance)
    return distance
    #raise NotImplementedError


#TODO:
def cosine_sim_distance(point1: List[float], point2: List[float]) -> float:
    dot_product = np.dot(point1, point2)
    mag_point1 = np.linalg.norm(point1)
    mag_point2 = np.linalg.norm(point2)
    distance = 1 - dot_product/(mag_point1*mag_point2)
    return distance
    #raise NotImplementedError


# TODO: select an instance of KNN with the best f1 score on validation dataset
def model_selection_without_normalization(distance_funcs, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    best_model = None
    best_k = 0
    best_f1_score = -1
    best_function = ""
    best_choices = []
    highest_range = len(Xtrain)-1
    if highest_range > 31:
        highest_range = 31
    for name, distance_func in distance_funcs.items():
        for k in range(1,highest_range,2):
            model = KNN(k, distance_function = distance_func)
            model.train(Xtrain, ytrain)
            train_f1_score = f1_score(ytrain, model.predict(Xtrain))
            valid_f1_score = f1_score(yval, model.predict(Xval))
            #print("name: ", name, "k: ", k, "train_score: ", train_f1_score, "valid_score: ", valid_f1_score)
            # print("Train score")
            if(best_f1_score<valid_f1_score):
                best_f1_score = valid_f1_score
                best_k = k
                best_model = model
                best_function = name
        best_choices.append([best_k, best_function, best_f1_score])
    print("best_k:", best_k, "best_function:", best_function, "best_f1_score:", best_f1_score)
    return best_model, best_k, best_function
    #raise NotImplementedError


# TODO: select an instance of KNN with the best f1 score on validation dataset, with normalized data
def model_selection_with_transformation(distance_funcs, scaling_classes, Xtrain, ytrain, Xval, yval):
    # distance_funcs: dictionary of distance funtion
    # scaling_classes: diction of scalers
    # Xtrain: List[List[int]] train set
    # ytrain: List[int] train labels
    # Xval: List[List[int]] validation set
    # yval: List[int] validation labels
    # return best_model: an instance of KNN
    # return best_k: best k choosed for best_model
    # return best_func: best function choosed for best_model
    # return best_scaler: best function choosed for best_model
    best_model = None
    best_choice = []
    best_scaler = ""
    best_k = 0
    best_f1_score = 0
    best_function = ""    
    highest_range = len(Xtrain)-1
    if highest_range > 31:
        highest_range = 31
    for scaler_name, scaling_class in scaling_classes.items():
        for name, distance_func in distance_funcs.items():
            scaler = scaling_class()
            Xtrain_scaled = scaler(Xtrain)
            Xval_scaled = scaler(Xval)
            for k in range(1, highest_range, 2):
                model = KNN(k, distance_function=distance_func)
                model.train(Xtrain_scaled, ytrain)
                train_f1_score = f1_score(ytrain, model.predict(Xtrain_scaled))
                valid_f1_score = f1_score(yval, model.predict(Xval_scaled))
                #print("scaler:", scaler_name, " name:", name, "k: ", k, "train score: ", train_f1_score, "valid_f1_score", valid_f1_score)
                if(best_f1_score < valid_f1_score):
                    best_f1_score = valid_f1_score
                    best_k = k
                    best_function = name
                    best_scaler = scaler_name
                    best_model = model
    #print("best_scaler:", best_scaler, "best_function:", best_function, "best_k:", best_k, "score:", best_f1_score)
    return best_model, best_k, best_function, best_scaler
    #raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    #TODO: normalize data
    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[3, 4], [1, -1], [0, 0]],
        the output should be [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]
        """
        normalized_features = []
        for feature in features:
            magnitude = np.linalg.norm(feature)
            if magnitude != 0:
                feat = [a/magnitude for a in feature]
                normalized_features.append(feat)
            else:
                normalized_features.append(feature)
        return normalized_features
        #raise NotImplementedError


class MinMaxScaler:
    """
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
        must be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler = MinMaxScale()
        train_features_scaled = scaler(train_features)
        # now train_features_scaled should be [[0, 1], [1, 0]]

        test_features_sacled = scaler(test_features)
        # now test_features_scaled should be [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """
    def __init__(self):
        #pass
        self.istest = 0
        self.min_set = []
        self.max_set = []

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]
        """
        if self.istest == 0: #for training only calculate this
            for col in range(len(features[0])):
                list = []
                for feature in features:
                    list.append(feature[col]) #append column-wise
                self.min_set.append(min(list))#min of the column
                self.max_set.append(max(list))#max of the column
        #print(features)
        for col in range(len(features[0])):
            max_val = self.max_set[col]
            min_val = self.min_set[col]
            denominator = max_val - min_val
            for f in features:
                if denominator == 0:
                    f[col] = 0
                else:
                    f[col] = (f[col] - min_val) / denominator
        self.istest += 1
        return features
        #raise NotImplementedError
    




