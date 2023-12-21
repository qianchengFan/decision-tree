import numpy as np
import sys
import matplotlib.pyplot as plt


# splite the dataset to list of x and y for later processing
def split_label(data_set):
    x = []
    y = []
    for data in data_set:
        x.append(data[:-1])
        y.append(data[-1])
    return x, y


# get all possible thresholds from the current training set.
def get_thresholds(x_tran):
    thresholds = []
    for feature in range(len(x_tran[0])):
        for value in x_tran:
            if (value[feature], feature) not in thresholds:
                thresholds.append([value[feature], feature])
    return thresholds

# helper function used to split tree node based on threshold
def get_splited(threshold, x_train, y_train):
    x_left_out = []
    y_left_out = []
    x_right_out = []
    y_right_out = []
    for num in range(len(x_train)):
        if x_train[num][threshold[1]] < threshold[0]:
            x_left_out.append(x_train[num])
            y_left_out.append(y_train[num])
        else:
            x_right_out.append(x_train[num])
            y_right_out.append(y_train[num])
    return x_left_out, y_left_out, x_right_out, y_right_out

# get the entropy of the labels within current node
def entropy(y_labels):
    counts = np.bincount(y_labels)
    total = sum(counts)
    all = []
    for i in counts:
        if i != 0:
            all.append((i / total) * np.log2(i / total))
    return -sum(all)

# calculate the gain from the split
def cal_gain(all, left, right):
    ent_all = entropy(all)
    ent_left = entropy(left)
    ent_right = entropy(right)
    remainder = (len(left) / len(all)) * ent_left + (len(right) / len(all)) * ent_right
    gain = ent_all - remainder
    return gain


def get_max_gain(x, y):
    # get all possible thresholds
    thresholds = get_thresholds(x)
    max_gain = 0
    x_left_out = []
    y_left_out = []
    x_right_out = []
    y_right_out = []
    threshold_chosen = None
    # find the threshold that can get max gain
    for threshold in thresholds:
        x_left, y_left, x_right, y_right = get_splited(threshold, x, y)
        gain = cal_gain(y, y_left, y_right)
        if gain > max_gain:
            max_gain = gain
            x_left_out = x_left
            y_left_out = y_left
            x_right_out = x_right
            y_right_out = y_right
            threshold_chosen = threshold
    # return the threshold and the child nodes
    return x_left_out, y_left_out, x_right_out, y_right_out, threshold_chosen


class TreeNode:
    def __init__(self, label=None, threshold=None, threshold_feature=None, left_branch=None, right_branch=None,
                 depth=0):
        self.label = label  # default is None,only be set to a value in leaf node
        self.threshold = threshold  # threshold value
        self.threshold_feature = threshold_feature  # which value in X that the threshold value applies on.
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.depth = depth


def decision_tree_learning(x, y, depth):
    if len(np.unique(y)) <= 1:  # then we got leaf node
        return TreeNode(label=y[0], depth=depth), depth
    else:
        # do tree splitting that get max gain
        x_left_out, y_left_out, x_right_out, y_right_out, threshold_chosen = get_max_gain(x, y)
        # update values and do the same thing to child node recursively
        left_child, left_depth = decision_tree_learning(x_left_out, y_left_out, depth + 1)
        right_child, right_depth = decision_tree_learning(x_right_out, y_right_out, depth + 1)
        node = TreeNode(depth=depth, threshold_feature=threshold_chosen[1], threshold=threshold_chosen[0],
                        left_branch=left_child,
                        right_branch=right_child)
        return node, max(left_depth, right_depth)

# classify a node using the trained tree
def classify(tree, features):
    if (tree.left_branch != None or tree.right_branch != None):
        if (features[tree.threshold_feature] < tree.threshold):
            return classify(tree.left_branch, features)
        else:
            return classify(tree.right_branch, features)
    return int(tree.label)

# evaluate a trained tree using the evaluation set, and get the confusion matrix
def get_confusion_matrix(myTree, eval_set_x, eval_set_y, label_num):
    y_pre = []
    for eval_x in eval_set_x:
        y_pre.append(classify(myTree, eval_x))

    # confusion matrix
    confusion_matrix = []
    for feature_actual in range(label_num):
        f = feature_actual + 1
        temp = [0] * label_num
        for idx in range(len(eval_set_y)):
            if (eval_set_y[idx] == f):
                temp[y_pre[idx] - 1] += 1
        confusion_matrix.append(temp)
    return confusion_matrix

# evaluate a trained tree using a evaluation set,
# return only the accuracy
def evaluate(test_db,trained_tree):
    eval_set_x,y_float = split_label(test_db)
    eval_set_y = [int(yi+0.0001) for yi in y_float]
    label_num = len(np.unique(eval_set_y))
    confusion_matrix = get_confusion_matrix(trained_tree, eval_set_x, eval_set_y, label_num)
    acc = 0
    all = 0

    for feature in range(len(confusion_matrix)):
        acc += confusion_matrix[feature][feature]
        all += np.sum(confusion_matrix[feature])
    acc = acc / all

    return acc

# evaluate a trained tree using the evaluation set,
# return all the metrics needed, including the confusion_matrix,accuracy, and recall and precision per class
def eval(tree, eval_set_x, eval_set_y, label_num):
    confusion_matrix = get_confusion_matrix(tree, eval_set_x, eval_set_y, label_num)
    acc = 0
    all = 0
    positives = [0] * len(confusion_matrix)
    recalls = [0] * len(confusion_matrix)
    precisions = [0] * len(confusion_matrix)

    # derive the metrics from the confusion matrix
    for feature in range(len(confusion_matrix)):
        acc += confusion_matrix[feature][feature]
        temp = 0
        for i in range(len(confusion_matrix)):
            positives[i] += confusion_matrix[feature][i]
            temp += confusion_matrix[feature][i]
        if temp == 0:
            recalls[feature] = 0
        else:
            recalls[feature] = confusion_matrix[feature][feature] / temp
        all += temp
    acc = acc / all

    for feature in range(len(confusion_matrix)):
        if positives[feature] == 0:
            precisions[feature] = 0
        else:
            precisions[feature] = confusion_matrix[feature][feature] / positives[feature]

    return confusion_matrix, acc, recalls, precisions

# tree pruning
def pruning(node, test_x, test_y, label_num):
    if node.depth > 10:
        node.left_branch = None
        node.right_branch = None
        node.label = max(test_y, key=test_y.count)
        return node

    if node.left_branch is not None and node.right_branch is not None:
        # Compare error rate
        confusion_matrix1, acc1, recalls1, precisions1 = eval(node, test_x, test_y, label_num)
        temp_left = node.left_branch
        temp_right = node.right_branch
        node.left_branch = None
        node.right_branch = None
        node.label = max(test_y, key=test_y.count)
        confusion_matrix2, acc2, recalls2, precisions2 = eval(node, test_x, test_y, label_num)

        if acc1 <= acc2 - 0.1 \
                and sum(recalls1) / len(recalls1) <= sum(recalls2) / len(recalls2) \
                and sum(precisions1) / len(precisions2) <= sum(precisions1) / len(precisions2):
            return node
        else:
            # Revert the pruning
            node.left_branch = temp_left
            node.right_branch = temp_right
            node.label = None

            # recursion
            x_left, y_left, x_right, y_right = get_splited([node.threshold, node.threshold_feature], x, y)

            if node.left_branch is not None:
                pruning(node.left_branch, x_left, y_left, label_num)
            if node.right_branch is not None:
                pruning(node.right_branch, x_right, y_right, label_num)
    return node


def get_tree_depth(node):
    if node is None:
        return 0.0, 0.0
    else:
        max_depth1, avg_depth1 = get_tree_depth(node.left_branch)
        max_depth2, avg_depth2 = get_tree_depth(node.right_branch)
        return 1.0 + max(max_depth1, max_depth2), 1.0 + (avg_depth1 + avg_depth2) / 2.0


# evaluate a tree training method using K-Fold validation
def eval_KFold(k, x, y):
    all_data = np.array(x)
    all_label = np.array(y)

    label_num = len(set(all_label))
    fold_indices = np.arange(all_data.shape[0])

    # Shuffle the indices
    np.random.shuffle(fold_indices)

    # Split the indices into k-parts
    eval_indices = np.array_split(fold_indices, k)

    classes = len(set(all_label))

    confusion_matrix_avg = np.zeros((classes, classes))
    acc_avg = 0
    recalls_avg = [0] * classes
    precisions_avg = [0] * classes
    f1scores_avg = [0] * classes

    for e in eval_indices:
        # for each fold, construct the training set and the evaluation set
        eval_set_x = list(all_data[e])
        eval_set_y = list(all_label[e])

        mask_eval = np.ones(all_data.shape[0], bool)
        mask_eval[e] = False

        train_set_x = list(all_data[mask_eval])
        train_set_y = list(all_label[mask_eval])

        # tree training and evaluation
        tree, max_depth = decision_tree_learning(train_set_x, train_set_y, 0)
        confusion_matrix, acc, recalls, precisions = eval(tree, eval_set_x, eval_set_y, label_num)

        acc_avg += acc / len(eval_indices)
        for i in range(classes):
            recalls_avg[i] += recalls[i] / len(eval_indices)
            precisions_avg[i] += precisions[i] / len(eval_indices)
            for j in range(classes):
                confusion_matrix_avg[i][j] += confusion_matrix[i][j] / len(eval_indices)

    for i in range(classes):
        f1scores_avg[i] = 2 / (1 / precisions_avg[i] + 1 / recalls_avg[1])

    return confusion_matrix_avg, acc_avg, recalls_avg, precisions_avg, f1scores_avg

# evaluate a tree pruning method using K-Fold validation
def eval_KFold_pruning(k, x, y):
    all_data = np.array(x)
    all_label = np.array(y)

    label_num = len(set(all_label))

    fold_indices = np.arange(all_data.shape[0])

    # Shuffle the indices
    np.random.shuffle(fold_indices)

    # Split the indices into k-parts
    eval_indices = np.array_split(fold_indices, k)

    classes = len(set(all_label))

    # initialize metrics
    confusion_matrix_avg = np.zeros((classes, classes))
    acc_avg = 0
    recalls_avg = [0] * classes
    precisions_avg = [0] * classes
    f1scores_avg = [0] * classes

    before_max_depth = 0
    before_avg_depth = 0
    after_max_depth = 0
    after_avg_depth = 0

    for e in eval_indices:

        # for each fold, construct the test set and the internal set
        eval_set_x = list(all_data[e])
        eval_set_y = list(all_label[e])

        mask_eval = np.ones(all_data.shape[0], bool)
        mask_eval[e] = False

        internal_set_x = all_data[mask_eval]
        internal_set_y = all_label[mask_eval]

        fold_indices_val = np.arange(internal_set_x.shape[0])

        # Split the indices into k-parts
        eval_indices_in = np.array_split(fold_indices_val, k - 1)

        for e_in in eval_indices_in:

            # split the internal set into training set and the pruning set
            pruning_set_x = list(internal_set_x[e_in])
            pruning_set_y = list(internal_set_y[e_in])

            mask_val = np.ones(internal_set_x.shape[0], bool)
            mask_val[e_in] = False

            train_set_x = list(internal_set_x[mask_val])
            train_set_y = list(internal_set_y[mask_val])

            tree, max_depth = decision_tree_learning(train_set_x, train_set_y, 0)

            max_depth, avg_depth = get_tree_depth(tree)
            before_max_depth = before_max_depth + max_depth
            before_avg_depth = before_avg_depth + avg_depth

            # tree pruning
            tree = pruning(tree, pruning_set_x, pruning_set_y, label_num)

            max_depth, avg_depth = get_tree_depth(tree)
            after_max_depth = after_max_depth + max_depth
            after_avg_depth = after_avg_depth + avg_depth

            confusion_matrix, acc, recalls, precisions = eval(tree, eval_set_x, eval_set_y, label_num)

            acc_avg += acc / len(eval_indices) / len(eval_indices_in)
            for i in range(classes):
                recalls_avg[i] += recalls[i] / len(eval_indices) / len(eval_indices_in)
                precisions_avg[i] += precisions[i] / len(eval_indices) / len(eval_indices_in)
                for j in range(classes):
                    confusion_matrix_avg[i][j] += confusion_matrix[i][j] / len(eval_indices) / len(eval_indices_in)

    for i in range(classes):
        f1scores_avg[i] = 2 / (1 / precisions_avg[i] + 1 / recalls_avg[1])

    print("before_max_depth: ", before_max_depth / (k * (k - 1)))
    print("before_avg_depth: ", before_avg_depth / (k * (k - 1)))
    print("after_max_depth: ", after_max_depth / (k * (k - 1)))
    print("after_avg_depth: ", after_avg_depth / (k * (k - 1)))

    return confusion_matrix_avg, acc_avg, recalls_avg, precisions_avg, f1scores_avg

# confusion matrix visualization
def cm_visualize(matrix):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(matrix, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Keep one decimal place
            label = (matrix[i, j] // 0.1) / 10
            ax.text(x=j, y=i,s=label, va='center', ha='center', size='xx-large')
    plt.xlabel('Predicted Labels', fontsize=15)
    plt.ylabel('Actual Labels', fontsize=15)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()


# used to get the width of the tree for tree visualization
def count_leaves(node):
    if node == None:
        return 0
    if node.left_branch == None and node.right_branch == None:
        return 1
    return count_leaves(node.left_branch) + count_leaves(node.right_branch)


def get_random_color():
    return np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)


def plot_node(node_text, child_node, parent_node, node_type, arrow_args, node_size, offset):
    arrow_pos = (parent_node[0], parent_node[1] - offset)
    create_plot.axl.annotate("", xy=arrow_pos, xycoords='axes fraction', xytext=child_node, textcoords='axes fraction',
                             va="center", ha="center", arrowprops=arrow_args)
    create_plot.axl.annotate(node_text, fontsize=node_size, xy=parent_node, xycoords='axes fraction', xytext=child_node,
                             textcoords='axes fraction', va="center", ha="center", bbox=node_type)


def plot_tree(x, y, interval, node, parent, node_size, arrow_args, offset):
    # x_position of left and right child
    left_x = x - interval
    right_x = x + interval
    # update interval
    interval = interval / 2
    # child nodes' y position
    next_y = y - 0.5
    # update parent node
    next_parent = [x, y]

    # write current node
    if node.left_branch is None and node.right_branch is None:
        label = "leaf:" + str(node.label)
        plot_node(label, (x, y), (parent[0], parent[1]), leafNode, arrow_args, node_size, offset)
    else:
        label = "[x" + str(node.threshold_feature) + " < " + str(node.threshold) + "]"
        plot_node(label, (x, y), (parent[0], parent[1]), Node, arrow_args, node_size, offset)

    # update next node's property and write child nodes recursively
    node_size = node_size * 0.8
    offset = offset * 0.8
    arrow_args = dict(arrowstyle="-", color=get_random_color())
    if node.left_branch is not None:
        plot_tree(left_x, next_y, interval, node.left_branch, next_parent, node_size, arrow_args, offset)
    if node.right_branch is not None:
        plot_tree(right_x, next_y, interval, node.right_branch, next_parent, node_size, arrow_args, offset)


def create_plot(root, tree_depth):
    total_width = count_leaves(root)  # width of tree
    center_x = total_width / 200  # mid x point of the graph
    interval = center_x / 2  # x-axis movement from parent node to child node
    top_y = tree_depth / 2  # highest y
    node_size = 15  # size of the node label
    offset = 0.049  # the movement needed for arrow to prevent overlap on node
    arrow_args = dict(arrowstyle="-",
                      color=get_random_color())  # arrow property, updated for different color on each layer
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.axl = plt.subplot()
    create_plot.axl.axis("off")
    plot_tree(center_x, top_y, interval, root, [center_x, top_y], node_size, arrow_args, offset)
    plt.show()


if __name__ == '__main__':
    data = np.loadtxt(sys.argv[1])
    Node = dict(boxstyle="round", color='b', fc="1")
    leafNode = dict(boxstyle="round", color='b', fc="1")
    x, y = split_label(data)
    # root, depth = decision_tree_learning(x, y, 0)
    # create_plot(root, depth)

    confusion_matrix_avg, acc_avg, recalls_avg, precisions_avg, f1scores_avg = eval_KFold_pruning(10, x, y)
    print("acc_avg: ", acc_avg)
    print("recalls_avg: ", recalls_avg)
    print("precisions_avg: ", precisions_avg)
    print("f1scores_avg: ", f1scores_avg)
    cm_visualize(confusion_matrix_avg)
