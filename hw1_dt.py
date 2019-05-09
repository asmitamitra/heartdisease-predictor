import numpy as np
import utils as Util


class DecisionTree():
    def __init__(self):
        self.clf_name = "DecisionTree"
        self.root_node = None

    def train(self, features, labels):
        # features: List[List[float]], labels: List[int]
        # init
        assert (len(features) > 0)
        self.feature_dim = len(features[0])
        num_cls = np.unique(labels).size

        # build the tree
        self.root_node = TreeNode(features, labels, num_cls)
        if self.root_node.splittable:
            self.root_node.split()

        return

    def predict(self, features):
        # features: List[List[any]]
        # return List[int]
        y_pred = []
        for idx, feature in enumerate(features):
            pred = self.root_node.predict(feature)
            y_pred.append(pred)
            # print ("feature: ", feature)
            # print ("pred: ", pred)
        return y_pred
    def accuracy(self, predicted_labels, actual_labels):
        total = len(predicted_labels)
        count = 0
        for index in range(total):
            if predicted_labels[index] == actual_labels[index]:
                count += 1
        return count/total
    def pruneTree(self, node, Xtest, ytest, firstmaxAccuracy, counter):
        global global_maxAccuracy
        if counter == 1: #first pass
            global_maxAccuracy = firstmaxAccuracy
        if node.splittable == False:
            return
        for c in node.children:
            if c.splittable == False:
                continue
            else:
                self.pruneTree(c, Xtest, ytest, firstmaxAccuracy, 2)
        node.splittable = False
        children = node.children
        node.children = []
        accuracy = self.accuracy(self.predict(Xtest), ytest)
        if(accuracy > global_maxAccuracy):
            global_maxAccuracy = accuracy
            return
        node.splittable = True
        node.children = children


class TreeNode(object):
    def __init__(self, features, labels, num_cls):
        # features: List[List[any]], labels: List[int], num_cls: int
        self.features = features
        self.labels = labels
        self.children = []
        self.num_cls = num_cls
        # find the most common labels in current node
        count_max = 0
        for label in np.unique(labels):
            if self.labels.count(label) > count_max:
                count_max = labels.count(label)
                self.cls_max = label
                # splitable is false when all features belongs to one class
        if len(np.unique(labels)) < 2:
            self.splittable = False
        else:
            self.splittable = True

        self.dim_split = None  # the index of the feature to be split

        self.feature_uniq_split = None  # the possible unique values of the feature to be split

    #TODO: try to split current node
    def split(self):
        #split on the basis of best_attribute -> highest information gain
        #calculate root entropy
        # print("features =========", self.features, " labels: ", self.labels, " num_cls ====", self.num_cls)
        num_class = np.unique(self.labels, return_counts=True)
        val_set = num_class[0]
        counts = num_class[1]
        total_entries = len(self.labels)
        prob = counts / total_entries
        root_entropy = 0
        for p in prob:
            root_entropy += -1*p*np.log2(p)
        # print("root_entropy",root_entropy)
        information_gains = []
        branch_feature_values = []
        best_information_gain = -1
        branch_feature_list = []
        #now find best attributes
        for index_col in range(len(self.features[0])):#for each feature
            coli = np.array(self.features)[:, index_col]
            branch_feature = np.unique(coli)#[a,b]
            # print("branch_feature ====", branch_feature)
            branch = [0] * len(branch_feature)
            for b in range(len(branch_feature)):
                branch[b] = [0] * self.num_cls
            # print(branch)
            # branch = np.zeros((len(branch_feature), self.num_cls))#num_branches * num_class
            feature_dict = {}
            count = 0
            for fea in branch_feature:
                feature_dict[fea] = count
                count+=1
            # print(feature_dict)
            labels_dict = {}
            count = 0
            for lab in np.unique(self.labels):
                labels_dict[lab] = count
                count += 1
            # print("coli ========", coli)
            for i in range(len(coli)):
                branch_feature_num = feature_dict[coli[i]]
                label_feature_num = labels_dict[self.labels[i]]
                branch[branch_feature_num][label_feature_num] += 1
            gain = Util.Information_Gain(root_entropy,branch)
            branch_feature_values.append([len(branch_feature), index_col])
            information_gains.append(gain)
            branch_feature_list.append(branch_feature.tolist())
            # print("gain====", gain)
            # if gain > best_information_gain:
            #     best_information_gain = gain
            #     self.dim_split = index_col
            #     self.feature_uniq_split = branch_feature.tolist()
        if(len(information_gains) <= 0):
            self.dim_split = None
            self.feature_uniq_split = None
            self.splittable = False
            return
        gain_max = max(information_gains)
        best_index_gain = [i for i in range(len(information_gains)) if information_gains[i] == gain_max]
        #we need to pick feature with most number of attr values
        max_attr = -1
        fea_col = -1
        best_branch_feature_list = []
        best_position = -1
        for ind in best_index_gain:
            if(max_attr < branch_feature_values[ind][0]):
                best_position = ind
                max_attr = branch_feature_values[ind][0]
                fea_col = branch_feature_values[ind][1]
                best_branch_feature_list = branch_feature_list[ind]
        best_information_gain = information_gains[best_position]
        self.dim_split = fea_col
        self.feature_uniq_split = best_branch_feature_list
        if best_information_gain <= -1:
            self.dim_split = None
            self.feature_uniq_split = None
            self.splittable = False
            return
        # print("dividing on the basis of =====", best_information_gain, "dimension:  ===", self.dim_split, "feature: ===", self.feature_uniq_split)
        #split the nodes, and add child nodes
        coli = np.array(self.features)[:, self.dim_split]
        # print("coli===========", coli) #column to be removed
        print("feature_uniq_split == ", self.feature_uniq_split)
        self.feature_uniq_split.sort()
        if len(self.feature_uniq_split)>0:
            for val in self.feature_uniq_split:
                labels_new = []
                features_new = []
                for index, row in enumerate(self.features):
                    if row[self.dim_split] == val : #value for which need to split
                        labels_new.append(self.labels[index])
                        features_new.append(row)
                features_new = np.delete(features_new, self.dim_split, axis = 1)
                num_class = np.unique(labels_new)
                child = TreeNode(features_new.tolist(), labels_new, len(num_class))
                self.children.append(child)
            # print(self.children)
            # split the child nodes
            for child in self.children:
                # print("child =========", child.splittable)
                if child.splittable:
                    child.split()
        return
        #raise NotImplementedError

    # TODO: predict the branch or the class
    def predict(self, feature):
        # feature: List[any]
        # return: int
        if self.splittable and (self.feature_uniq_split and feature[self.dim_split] in self.feature_uniq_split):
            #print('New')
            #print("feature: ", feature)
            #print("split_index ====", self.dim_split)
            #print(feature)
            #print("feature on split: =============", feature[self.dim_split])
            #print("unique feature list ===========", self.feature_uniq_split)
            #print("index of feature_uniq_split ========", self.feature_uniq_split.index(feature[self.dim_split]))
            #print(self.children)


            #can be splitted - non leaf
            # index_child = self.feature_uniq_split.index(feature[self.dim_split])

            child_index = self.feature_uniq_split.index(feature[self.dim_split])
            feature = feature[:self.dim_split] + feature[self.dim_split+1:]
            #print("new feature ==========", feature)
            #self.children[child_index].printall()
            return self.children[child_index].predict(feature)
        else:
            return self.cls_max
        #raise NotImplementedError
