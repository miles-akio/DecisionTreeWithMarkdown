# Decision Tree Project: Classifying Edible vs. Poisonous Mushrooms

---

In this project, I implemented a **decision tree from scratch** and applied it to classify mushrooms as **edible or poisonous** based on their physical attributes. This README explains my approach, the dataset, and the methods I used to build the decision tree.

---

## Table of Contents

* [1. Packages](#1)
* [2. Problem Statement](#2)
* [3. Dataset](#3)

  * [3.1 One-hot Encoded Dataset](#3.1)
* [4. Decision Tree Refresher](#4)

  * [4.1 Calculating Entropy](#4.1)
  * [4.2 Splitting the Dataset](#4.2)
  * [4.3 Calculating Information Gain](#4.3)
  * [4.4 Selecting the Best Split](#4.4)
* [5. Building the Tree](#5)

---

<a name="1"></a>

## 1. Packages

To implement this project, I imported the following packages:

```python
import numpy as np
import matplotlib.pyplot as plt
from public_tests import *
from utils import *

%matplotlib inline
```

* **numpy**: Used for matrix computations and handling arrays efficiently.
* **matplotlib**: Used for visualizations of the data and tree structure.
* **utils.py**: Contains helper functions such as visualizations.

---

<a name="2"></a>

## 2. Problem Statement

Suppose I am running a company that grows and sells wild mushrooms. Since not all mushrooms are edible, I need a method to classify mushrooms as **edible or poisonous** based on their physical attributes. Using the dataset I have collected, I aim to build a decision tree that helps me identify which mushrooms can be sold safely.

> **Note:** The dataset is illustrative and should not be used to determine real-world mushroom safety.

---

<a name="3"></a>

## 3. Dataset

I collected a dataset with **10 examples of mushrooms**. Each mushroom has three features and a label indicating whether it is edible:

| Image              | Cap Color | Stalk Shape | Solitary | Edible |
| ------------------ | --------- | ----------- | -------- | ------ |
| ![0](images/0.png) | Brown     | Tapering    | Yes      | 1      |
| ![1](images/1.png) | Brown     | Enlarging   | Yes      | 1      |
| ![2](images/2.png) | Brown     | Enlarging   | No       | 0      |
| ![3](images/3.png) | Brown     | Enlarging   | No       | 0      |
| ![4](images/4.png) | Brown     | Tapering    | Yes      | 1      |
| ![5](images/5.png) | Red       | Tapering    | Yes      | 0      |
| ![6](images/6.png) | Red       | Enlarging   | No       | 0      |
| ![7](images/7.png) | Brown     | Enlarging   | Yes      | 1      |
| ![8](images/8.png) | Red       | Tapering    | No       | 1      |
| ![9](images/9.png) | Brown     | Enlarging   | No       | 0      |

* Features:

  * **Cap Color**: Brown or Red
  * **Stalk Shape**: Tapering or Enlarging
  * **Solitary**: Yes or No
* Label:

  * **Edible**: 1 = yes, 0 = poisonous

---

<a name="3.1"></a>

### 3.1 One-hot Encoded Dataset

For ease of implementation, I converted categorical features into **0/1 values**:

| Image | Brown Cap | Tapering Stalk | Solitary | Edible |
| ----- | --------- | -------------- | -------- | ------ |
| 0     | 1         | 1              | 1        | 1      |
| 1     | 1         | 0              | 1        | 1      |
| 2     | 1         | 0              | 0        | 0      |
| 3     | 1         | 0              | 0        | 0      |
| 4     | 1         | 1              | 1        | 1      |
| 5     | 0         | 1              | 1        | 0      |
| 6     | 0         | 0              | 0        | 0      |
| 7     | 1         | 0              | 1        | 1      |
| 8     | 0         | 1              | 0        | 1      |
| 9     | 1         | 0              | 0        | 0      |

```python
X_train = np.array([
    [1,1,1],[1,0,1],[1,0,0],[1,0,0],[1,1,1],
    [0,1,1],[0,0,0],[1,0,1],[0,1,0],[1,0,0]
])
y_train = np.array([1,1,0,0,1,0,0,1,1,0])
```

I explored the dataset:

```python
print("First few elements of X_train:\n", X_train[:5])
print("Type of X_train:", type(X_train))

print("First few elements of y_train:", y_train[:5])
print("Type of y_train:", type(y_train))

print('The shape of X_train is:', X_train.shape)
print('The shape of y_train is:', y_train.shape)
print('Number of training examples (m):', len(X_train))
```

---

<a name="4"></a>

## 4. Decision Tree Refresher

I built the decision tree following these steps:

1. Start at the root node with all examples.
2. Compute **information gain** for splitting on each feature.
3. Split dataset based on the feature with the **highest information gain**.
4. Recursively repeat until a stopping criterion is met (maximum depth = 2).

---

<a name="4.1"></a>

### 4.1 Calculating Entropy

I implemented a function `compute_entropy` to calculate the **entropy** at a node:

```python
def compute_entropy(y):
    """
    Computes the entropy for a node
    
    Args:
        y (ndarray): Array indicating whether each example is edible (1) or poisonous (0)
        
    Returns:
        entropy (float)
    """
    if len(y) == 0:
        return 0
    
    p1 = np.sum(y == 1) / len(y)
    
    if p1 == 0 or p1 == 1:
        return 0
    
    entropy = -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)
    return entropy

print("Entropy at root node: ", compute_entropy(y_train))
compute_entropy_test(compute_entropy)
```

---

<a name="4.2"></a>

### 4.2 Splitting the Dataset

Next, I wrote a `split_dataset` function to divide the data based on a feature:

```python
def split_dataset(X, node_indices, feature):
    left_indices = []
    right_indices = []
    
    for i in node_indices:
        if X[i][feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    
    return left_indices, right_indices

root_indices = list(range(len(X_train)))
feature = 0
left_indices, right_indices = split_dataset(X_train, root_indices, feature)
split_dataset_test(split_dataset)
```

---

<a name="4.3"></a>

### 4.3 Calculating Information Gain

I implemented `compute_information_gain` to calculate **information gain** from splitting on a feature:

```python
def compute_information_gain(X, y, node_indices, feature):
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]
    
    node_entropy = compute_entropy(y_node)
    left_entropy = compute_entropy(y_left)
    right_entropy = compute_entropy(y_right)
    
    w_left = len(y_left) / len(y_node)
    w_right = len(y_right) / len(y_node)
    
    weighted_entropy = w_left * left_entropy + w_right * right_entropy
    information_gain = node_entropy - weighted_entropy
    
    return information_gain

info_gain0 = compute_information_gain(X_train, y_train, root_indices, feature=0)
info_gain1 = compute_information_gain(X_train, y_train, root_indices, feature=1)
info_gain2 = compute_information_gain(X_train, y_train, root_indices, feature=2)
compute_information_gain_test(compute_information_gain)
```

---

<a name="4.4"></a>

### 4.4 Selecting the Best Split

Finally, I wrote `get_best_split` to select the feature with **maximum information gain**:

```python
def get_best_split(X, y, node_indices):
    num_features = X.shape[1]
    best_feature = -1
    max_info_gain = 0
    
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
            
    return best_feature

best_feature = get_best_split(X_train, y_train, root_indices)
get_best_split_test(get_best_split)
```

---

<a name="5"></a>

## 5. Building the Tree

Using the functions I implemented, I recursively built a **decision tree** until the maximum depth was reached:

```python
tree = []

def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth):
    if current_depth == max_depth:
        print(" "*current_depth + "-"*current_depth, "%s leaf node with indices" % branch_name, node_indices)
        return

    best_feature = get_best_split(X, y, node_indices)
    formatting = "-"*current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    
    left_indices, right_indices = split_dataset(X, node_indices, best_feature)
    tree.append((left_indices, right_indices, best_feature))
    
    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth+1)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth+1)

build_tree_recursive(X_train, y_train, root_indices, "Root", max_depth=2, current_depth=0)
generate_tree_viz(root_indices, y_train, tree)
```

This recursive algorithm prints the structure of the tree and demonstrates how the dataset is split at each node.

---

✅ **Summary:**

* I built a decision tree from scratch using **entropy** and **information gain**.
* The model can classify mushrooms as edible or poisonous based on features like **cap color, stalk shape, and solitary nature**.
* I successfully visualized and tested each step of the process.

---
