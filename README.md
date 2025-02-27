# **Machine Learning Algorithms Repository**

## **Description**
This repository contains implementations of various **Machine Learning algorithms**, applied to different datasets. Each folder contains:
- A **dataset** used for training and evaluation.
- A **Jupyter Notebook (.ipynb)** with the code implementation.
- A **PDF report** explaining the algorithm’s working, evaluation metrics, and performance.

This repository is intended for students, researchers, and ML practitioners exploring different models.

---

## **Algorithms & Datasets**
### **1. Classification Algorithms Evaluation**
- **Dataset:** `diabetes_dataset_sunmitted.csv`
- **Code Implementation:** _Not available_
- **Report:** [`ClassificationAlgos.pdf`](Classification_Algos_Evaluation/ClassificationAlgos.pdf)
- **Description:** Evaluation of various classification models on the **Diabetes dataset**.

### **2. Comparison of Regression Algorithms**
- **Dataset:** `USA_Housing.csv`
- **Code Implementation:** [`Code_Project1.ipynb`](Comparison_Regression_Algos/Code_Project1.ipynb)
- **Report:** [`Regression_Algos_Evaluation.pdf`](Comparison_Regression_Algos/Regression_Algos_Evaluation.pdf)
- **Description:** Comparison of multiple **regression models** for predicting housing prices.

### **3. K-Means & Hierarchical Clustering**
- **Dataset:** `IrisDatasetSubmitted.csv`
- **Code Implementation:** [`05cCode.ipynb`](KMeans_Hierarchical_Clustering/05cCode.ipynb)
- **Report:** [`KMeans_Hierarchical.pdf`](KMeans_Hierarchical_Clustering/KMeans_Hierarchical.pdf)
- **Description:** Unsupervised learning techniques applied to the **Iris dataset** using **K-Means and Hierarchical Clustering**.

### **4. Linear & Multiple Linear Regression**
- **Dataset:** `CarPrice_Dataset.csv`
- **Code Implementation:** [`Code_CarPrice.ipynb`](Linear_MultipleLinearRegression/Code_CarPrice.ipynb)
- **Report:** [`01_Linear_MultipleLinearRegression.pdf`](Linear_MultipleLinearRegression/01_Linear_MultipleLinearRegression.pdf)
- **Additional File:** [`Data Dictionary - carprices.xlsx`](Linear_MultipleLinearRegression/Data%20Dictionary%20-%20carprices.xlsx)
- **Description:** Implements **Linear Regression** to predict car prices based on multiple features.

### **5. Polynomial Regression & Support Vector Regression (SVR)**
- **Dataset:** `Concrete_Dataset.xls`
- **Code Implementation:** [`Polynomial_SVR_Code.ipynb`](PolynomialReg_SVR/Polynomial_SVR_Code.ipynb)
- **Report:** [`02_PolyReg_SVR.pdf`](PolynomialReg_SVR/02_PolyReg_SVR.pdf)
- **Description:** Uses **Polynomial Regression and SVR** for modeling non-linear relationships in the **Concrete dataset**.

### **6. SVM, Logistic Regression & K-Nearest Neighbors (KNN)**
- **Dataset:** `heartDataset.csv`
- **Code Implementation:** [`heart_disease_model_Code.ipynb`](SVM_LogisticRegression_KNN/heart_disease_model_Code.ipynb)
- **Report:** [`SVM_LogisticReg_KNN.pdf`](SVM_LogisticRegression_KNN/SVM_LogisticReg_KNN.pdf)
- **Description:** Comparative analysis of **SVM, Logistic Regression, and KNN** for **heart disease prediction**.

### **7. SVR, Decision Tree & Random Forest**
- **Dataset:** `HousingDataSET.csv`
- **Code Implementation:** [`SVR_DT_RFCode.ipynb`](SVR_DecisionTree_RandomForest/SVR_DT_RFCode.ipynb)
- **Report:** [`03_SVR_DT_RF.pdf`](SVR_DecisionTree_RandomForest/03_SVR_DT_RF.pdf)
- **Description:** Regression models including **Support Vector Regression, Decision Trees, and Random Forest** for housing price prediction.

---

## **Installation & Usage**
### **Prerequisites**
- **Python 3.x** installed ([Download Here](https://www.python.org/))
- **Jupyter Notebook** installed (`pip install jupyter`)
- Required Python libraries:
  ```sh
  pip install numpy pandas matplotlib seaborn scikit-learn
  ```
#### **3. Run the Code in Google Colab**
```sh
# Open Google Colab: https://colab.research.google.com/
# Upload the .ipynb file manually OR open it directly from GitHub using:

# If the dataset is in the repository upload it to the notebook.

# After uploading the dataset, make sure to update the file path accordingly in the notebook.
# Run the notebook cells in order.
```
