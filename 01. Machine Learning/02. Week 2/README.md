# Week 02 - Linear Models

In this module, we will focus on **Linear Models**. Despite the term "linear," these models offer more than just straightforward, one-dimensional analysis. We will explore **regression**, **classification**, and various techniques for optimizing parameters to enhance model performance.

---

## 1. Overview of Linear Models
Linear Models, often considered classic in the field of Machine Learning, establish a linear relationship between **input features** and **outputs**.  
- **Regression**: Used for **predicting continuous values** (e.g., house prices, sales forecasts).  
- **Classification**: Used for **predicting categorical labels** (e.g., spam vs. non-spam, cancer vs. healthy).  

Despite their long-standing history, Linear Models remain widely used in industry due to their **simplicity**, **fast training speed**, and **sufficient accuracy** for many real-world applications.

---

## 2. Regression (Linear Regression)

### 2.1 Core Concept
*Linear Regression* can be summarized by the following equation:
$$
y \approx w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$
- \( y \) = The predicted target (a continuous value)
- \( x_i \) = Input features
- \( w_i \) = Model coefficients or weights

The primary goal is to determine the parameter vector \(\mathbf{w}\) that minimizes the **error**, often by employing **Least Squares** or **Gradient Descent**.

### 2.2 Feature Selection & Feature Engineering
- **Feature Selection**: Choose the most relevant features to reduce noise and complexity. Methods include *Regularization* (Lasso, Ridge) or filter-based approaches (e.g., correlation).
- **Feature Engineering**: Create new, more informative features. Examples include **log transformations**, polynomial terms, or combining existing features into a single feature.

### 2.3 Evaluation Metrics for Regression
Common metrics for evaluating regression models include:
- **MSE (Mean Squared Error)**: The average of squared errors; smaller values indicate higher accuracy.
- **RMSE (Root Mean Squared Error)**: The square root of MSE, often more interpretable.
- **MAE (Mean Absolute Error)**: The average of absolute errors; more robust in the presence of outliers.
- **R² (Coefficient of Determination)**: Reflects how well the model explains the variance in the data (range 0–1). Values closer to 1 indicate better performance.

---

## 3. Classification (Logistic Regression)

### 3.1 Core Concept
Although it may seem counterintuitive to categorize *Logistic Regression* under Linear Models, it is essentially a linear model with a **logistic (sigmoid) function** applied to output probabilities in the range of 0–1:
\[
p = \sigma(w_0 + w_1 x_1 + \dots + w_n x_n), 
\quad \text{where} \quad 
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

### 3.2 Feature Selection & Feature Engineering for Classification
- Similar to regression, **feature selection** is critical to identify the most impactful predictors.
- **Feature engineering** is also valuable, for instance, by creating dummy variables for categorical data or applying transformations that reveal hidden patterns.

### 3.3 Evaluation Metrics for Classification
- **Accuracy**: The percentage of correct predictions, which can be misleading with highly imbalanced datasets.
- **Precision**: Of all predicted positives, how many are truly positive.
- **Recall**: Of all actual positives, how many are correctly identified as positive.
- **F1-Score**: The harmonic mean of precision and recall, useful when both are equally important.
- **Confusion Matrix**: Displays the counts of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

---

## 4. Key Takeaways
1. **Linear Models** can address various tasks, from pricing forecasts to email classification.  
2. Their **simplicity** and **efficiency** make them reliable, even alongside more advanced models.  
3. **Feature Engineering** and **Feature Selection** are crucial for achieving high accuracy.  
4. **Evaluation** metrics should align with the problem type:  
   - Regression -> MSE, RMSE, MAE, R²  
   - Classification -> Accuracy, Precision, Recall, F1, Confusion Matrix  

---

## 5. Advanced Topics
- **Regularization** (Ridge, Lasso, ElasticNet): Helps reduce overfitting by penalizing large coefficients, leading to more robust models.
- **Multiclass Classification**: Logistic Regression can handle more than two classes using approaches like *One-vs-Rest* or *One-vs-One*.
- **Dimensionality Reduction**: Techniques such as *PCA* can be applied prior to training linear models for greater efficiency and easier computation.

---

## Conclusion
This concludes the Week 02 overview of **Linear Models**. Though considered foundational, **Linear Regression** and **Logistic Regression** remain highly **effective** and widely used. Do not underestimate the importance of **feature selection** and **feature engineering** in enhancing model performance. Keep exploring, and see you in the next module!

<sub>Stay on track, keep exploring, and never stop learning.</sub>
