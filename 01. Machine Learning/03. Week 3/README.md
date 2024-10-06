# 🌳 Week 03: Instance-Based Methods and Tree-Based Methods

## 🌲 Decision Tree Regression and Classification

**Decision Trees** are supervised learning algorithms used for both regression and classification tasks. They work by recursively splitting the data into subsets based on feature values, creating a tree-like model of decisions.

**🎯 Goal**: To predict the value of a target variable by learning simple decision rules inferred from the data's features.

### ✨ Key Concepts and Equations

- **Impurity Measures**: To decide the best split at each node, impurity measures like Gini impurity and entropy are used.

  - **Gini Impurity** (for classification):

    $$
    \text{Gini} = 1 - \sum_{i=1}^{C} p_i^2
    $$

    Where:
    - \( C \) = number of classes
    - \( p_i \) = proportion of instances belonging to class \( i \) in the node

  - **Entropy** (Information Gain):

    $$
    \text{Entropy} = -\sum_{i=1}^{C} p_i \log_2 p_i
    $$

- **Information Gain**: Measures the reduction in entropy after a dataset is split on an attribute.

  $$
  IG(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \text{Entropy}(S_v)
  $$

  Where:
  - \( S \) = the set of all instances
  - \( A \) = attribute being split on
  - \( \text{Values}(A) \) = set of all possible values for attribute \( A \)
  - \( S_v \) = subset of \( S \) where attribute \( A \) has value \( v \)

- **Regression Trees**: For regression tasks, splits are based on minimizing the Mean Squared Error (MSE).

  - **Mean Squared Error**:

    $$
    \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
    $$

    Where:
    - \( n \) = number of instances in the node
    - \( y_i \) = actual value of instance \( i \)
    - \( \bar{y} \) = mean value of all \( y_i \) in the node

### ✨ Key Features

- **Intuitive Visualization**: Provides a clear visual representation of the decision-making process.
- **Versatile**: Suitable for both regression and classification tasks.
- **Handles Various Data Types**: Can manage both numerical and categorical data.
- **Minimal Data Preparation**: Requires little to no data preprocessing.

---

## 🤖 k-NN Regression and Classification

**k-Nearest Neighbors (k-NN)** is an instance-based learning algorithm applicable to both regression and classification tasks. It predicts outcomes based on the 'k' most similar instances (neighbors) from the training data.

- **Distance Metrics**: Different metrics can be used to measure similarity between instances.

  - **Euclidean Distance** (most common):

    $$
    d_{\text{Euclidean}}(x, x_i) = \sqrt{\sum_{j=1}^{m} (x_j - x_{i,j})^2}
    $$

  - **Manhattan Distance**:

    $$
    d_{\text{Manhattan}}(x, x_i) = \sum_{j=1}^{m} |x_j - x_{i,j}|
    $$

  - **Minkowski Distance** (generalization of Euclidean and Manhattan):

    $$
    d_{\text{Minkowski}}(x, x_i) = \left( \sum_{j=1}^{m} |x_j - x_{i,j}|^p \right)^{1/p}
    $$

    Where:
    - \( p \) = order parameter ( \( p=1 \) for Manhattan, \( p=2 \) for Euclidean )

  - **Hamming Distance** (for categorical variables):

    $$
    d_{\text{Hamming}}(x, x_i) = \sum_{j=1}^{m} \delta(x_j, x_{i,j})
    $$

    Where:
    - \( \delta(x_j, x_{i,j}) = 0 \) if \( x_j = x_{i,j} \), else \( 1 \)

  - **Cosine Similarity** (converted to distance):

    $$
    d_{\text{Cosine}}(x, x_i) = 1 - \frac{\sum_{j=1}^{m} x_j x_{i,j}}{\sqrt{\sum_{j=1}^{m} x_j^2} \sqrt{\sum_{j=1}^{m} x_{i,j}^2}}
    $$

- **Classification Prediction**:

  The predicted class \( \hat{y} \) is determined by majority vote:

  $$
  \hat{y} = \text{mode}\{ y_i \mid i \in N_k \}
  $$

  Where:
  - \( N_k \) = indices of the \( k \) nearest neighbors
  - \( y_i \) = class label of neighbor \( i \)

- **Regression Prediction**:

  The predicted value \( \hat{y} \) is the average of neighbors' values:

  $$
  \hat{y} = \frac{1}{k} \sum_{i \in N_k} y_i
  $$

### 🏋️ Weighted k-NN

In **Weighted k-NN**, the contribution of each neighbor is weighted according to its distance from the query point. Closer neighbors have a higher influence on the prediction than farther ones.

- **Weight Determination**:

  A common weighting scheme is to use the inverse of the distance:

  $$
  w_i = \frac{1}{d(x, x_i) + \epsilon}
  $$

  Where:
  - \( w_i \) = weight of neighbor \( i \)
  - \( d(x, x_i) \) = distance between the query point \( x \) and neighbor \( x_i \)
  - \( \epsilon \) = small constant to avoid division by zero

- **Classification Prediction**:

  The predicted class is determined by the weighted vote:

  $$
  \hat{y} = \arg\max_{c \in C} \left( \sum_{i \in N_k} w_i \cdot \mathbb{I}(y_i = c) \right)
  $$

  Where:
  - \( C \) = set of all classes
  - \( \mathbb{I}(y_i = c) \) = indicator function (1 if \( y_i = c \), else 0)

- **Regression Prediction**:

  The predicted value is the weighted average of neighbors' values:

  $$
  \hat{y} = \frac{\sum_{i \in N_k} w_i \cdot y_i}{\sum_{i \in N_k} w_i}
  $$

### 🌟 Key Features

- **Simple & Effective**: No explicit training phase; straightforward to implement.
- **Hyperparameter Sensitivity**: Performance depends heavily on the choice of \( k \) and the distance metric.
- **Non-Parametric**: Makes no underlying assumptions about the data distribution.
- **Adaptable**: Capable of modeling complex, non-linear relationships.
- **Weighted Influence**: Weighted k-NN improves prediction by considering the relative distances of neighbors.

---

## 📊 Comparison Overview

| **Aspect**                 | **Decision Trees**                 | **k-NN**                         |
|----------------------------|------------------------------------|----------------------------------|
| **Interpretability**       | High                               | Moderate                         |
| **Training Time**          | Longer (tree construction)         | Minimal (instance-based)         |
| **Prediction Time**        | Fast                               | Slower (distance calculations)   |
| **Data Types**             | Numerical & Categorical            | Primarily Numerical              |
| **Data Preprocessing**     | Minimal                            | Requires feature scaling         |
| **Handles Missing Values** | Yes                                | No                               |
| **Hyperparameters**        | Tree depth, impurity criteria      | \( k \), distance metric         |
| **Robust to Outliers**     | Prone unless pruned                | Sensitive                        |

---

💡 **Note**: Both **Decision Trees** and **k-NN** are powerful tools in your machine learning toolkit, each with its own strengths and considerations. Decision Trees offer high interpretability and handle various data types well, while k-NN is prized for its simplicity and effectiveness in modeling non-linear relationships. Weighted k-NN enhances the basic k-NN algorithm by assigning greater influence to closer neighbors, potentially improving prediction accuracy.

---

Feel free to dive deeper into each method to understand their nuances and best use cases! 🚀
