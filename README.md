# IRIS FLOWER CLASSIFICATION USING DECISION TREE

1. PROJECT DESCRIPTION

---

This project implements a Machine Learning classification model to predict
the species of an Iris flower based on its sepal and petal measurements.

The project uses a Decision Tree Classifier from the scikit-learn library.

The model classifies Iris flowers into the following species:

1. Iris-setosa

2. Iris-versicolor

3. Iris-virginica

4. OBJECTIVE

---

The main objective of this project is to:

* Load and analyze the Iris dataset.
* Perform basic Exploratory Data Analysis (EDA).
* Select independent and dependent variables.
* Visualize the dataset.
* Split the dataset into training and testing sets.
* Train a Decision Tree classification model.
* Predict Iris flower species.
* Evaluate model performance using standard classification metrics.

3. DATASET

---

Dataset file:

iris.csv

Number of records:

150

Input features:

1. sepal length (cm)
2. sepal width (cm)
3. petal length (cm)
4. petal width (cm)

Target variable:

species

4. TECHNOLOGIES USED

---

Programming Language:
Python

Libraries:

* pandas
* matplotlib
* seaborn
* scikit-learn

5. MACHINE LEARNING ALGORITHM

---

Algorithm:

Decision Tree Classifier

Model configuration:

DecisionTreeClassifier(max_depth=5)

The maximum depth of the decision tree is set to 5 to control the complexity
of the model.

6. PROJECT WORKFLOW

---

Step 1: Load Dataset

The Iris dataset is loaded using pandas read_csv().

Step 2: Exploratory Data Analysis

The following dataset information is analyzed:

* Number of rows and columns
* Column names
* Missing values
* Class distribution
* Statistical summary

Step 3: Feature and Target Selection

Independent variables:

* sepal length (cm)
* sepal width (cm)
* petal length (cm)
* petal width (cm)

Dependent variable:

* species

Step 4: Data Visualization

A scatter plot is created using petal length and petal width to visualize
the distribution of different Iris species.

Step 5: Train-Test Split

The dataset is divided into training and testing datasets.

Training data:

75 records

Testing data:

75 records

The train_test_split() function is used with random_state=42.

Step 6: Model Creation

A Decision Tree Classifier is created with max_depth=5.

Step 7: Model Training

The model is trained using the training dataset.

Step 8: Prediction

The trained model predicts the species of the Iris flowers in the test dataset.

Step 9: Model Evaluation

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

7. MODEL EVALUATION

---

The following metrics are used to evaluate the classification model:

Accuracy:

Measures the percentage of correctly classified samples.

Confusion Matrix:

Shows the number of correct and incorrect predictions for each class.

Classification Report:

Provides:

* Precision
* Recall
* F1-score
* Support

8. PROJECT STRUCTURE

---

Iris-Classification/
|
|-- iris.csv
|-- iris_classification.py
|-- README.txt
|-- requirements.txt

9. INSTALLATION

---

Make sure Python is installed on the system.

Install the required dependencies using:

pip install -r requirements.txt

10. EXECUTION

---

Run the Python program using:

python iris_classification.py

11. EXPECTED OUTPUT

---

The program displays:

* Initial dataset records
* Dataset shape
* Column names
* Missing value information
* Species distribution
* Statistical summary
* Training and testing dataset shapes
* Actual test values
* Predicted values
* Model accuracy
* Confusion matrix
* Classification report

A scatter plot is also displayed during execution.

12. REQUIREMENTS

---

Python 3.x

Required Python packages are listed in requirements.txt.

13. FUTURE ENHANCEMENTS

---

Possible improvements include:

* Use stratified train-test splitting.
* Perform hyperparameter tuning.
* Compare multiple classification algorithms.
* Add cross-validation.
* Visualize the trained Decision Tree.
* Analyze feature importance.
* Improve model evaluation with additional metrics.

14. AUTHOR

---

Pratiksha Mahale

Project Type:
Machine Learning Classification Case Study

Domain:
Artificial Intelligence and Machine Learning
