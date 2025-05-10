# SVM Kernel Classification Project

This project implements and evaluates a Support Vector Machine (SVM) with different kernel functions for a binary classification task. The primary dataset used is the Titanic dataset, where the goal is to predict passenger survival based on various features.

## Description

The project follows a structured approach to machine learning model development:

1.  **Data Loading and Initial Exploration:**
    *   Loads the Titanic dataset (`train.csv` and `test.csv`) using pandas.
    *   Performs initial data inspection using methods like `.head()`, `.info()`, `.isnull().sum()`, and `.describe()` to understand the data structure, data types, missing values, and basic statistics.

2.  **Exploratory Data Analysis (EDA) and Visualization:**
    *   Visualizes relationships between features and the target variable (`Survived`) using `seaborn` and `matplotlib`.
    *   Examples include:
        *   Scatter plot of `Age` vs. `Fare` colored by `Survived`.
        *   Count plots for `Sex`, `Pclass`, `SibSp`, and `Parch` showing survival distribution.
    *   Feature engineering: Creates a `Family_Size` feature by combining `SibSp` and `Parch`. Visualizes survival based on `Family_Size`.

3.  **Data Preprocessing:**
    *   **Handling Missing Values:** Imputes missing `Age` values with the median. Fills missing `Embarked` values with the mode ("S").
    *   **Categorical Feature Encoding:** Converts categorical features (`Sex`, `Embarked`) into numerical representations using `sklearn.preprocessing.LabelEncoder`.
    *   **Feature Dropping:** Removes columns deemed less relevant for this model (`Name`, `Ticket`, `PassengerId`, `Cabin`).
    *   **Feature Creation/Transformation:**
        *   Creates `AgeGroup` by binning the `Age` feature into discrete categories.
        *   The original `Age` column is then dropped.
    *   **Numerical Feature Scaling:** Standardizes the `Fare` feature using `sklearn.preprocessing.StandardScaler`.

4.  **Custom SVM Implementation (`KernelSVM` class):**
    *   A custom SVM classifier is implemented from scratch to provide a deeper understanding of its mechanics.
    *   **Kernel Functions:** The implementation supports several common kernel functions:
        *   `linear`: \(K(x_i, x_j) = x_i \cdot x_j\)
        *   `quadratic`: \(K(x_i, x_j) = (x_i \cdot x_j + c)^2\)
        *   `poly2`: \(K(x_i, x_j) = (x_i \cdot x_j)^2\) (a specific case of the polynomial kernel)
        *   `rbf` (Radial Basis Function): \(K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)\)
    *   **Optimization:** Uses a basic gradient descent algorithm to find the optimal weights (`w`) and bias (`b`). The loss function includes a hinge loss component and L2 regularization.
    *   **Training Process:** Iteratively updates weights and bias, and tracks training loss, training accuracy, and validation (test) accuracy over epochs.

5.  **Model Training, Evaluation, and Visualization:**
    *   **Data Splitting:** The preprocessed data is split into training and testing sets using `sklearn.model_selection.train_test_split`. Note: The notebook also includes a `make_classification` step to generate synthetic data for demonstrating the SVM, which might be separate from the Titanic data processing flow for the final model evaluation. The target variable is mapped to -1 and 1 for SVM.
    *   **Iterating Through Kernels:** The custom `KernelSVM` is trained and evaluated separately for each of the implemented kernels (`linear`, `quadratic`, `poly2`, `rbf`).
    *   **Evaluation Metrics:** For each kernel, the model's performance is assessed using:
        *   `sklearn.metrics.classification_report`: Provides precision, recall, F1-score, and support for both training and test sets.
    *   **Visualization of Results:**
        *   **Loss Curve:** Plots the training loss over epochs.
        *   **Accuracy Curves:** Plots both training and test accuracy over epochs.
        *   **Decision Boundary Plot:** Visualizes the decision boundary learned by the SVM on the 2D feature space (using the synthetic data or a 2D subset of the Titanic data if applicable).

The primary objective is to demonstrate the from-scratch implementation of an SVM with various kernels and to visually and metrically compare their performance characteristics on a classification problem.

## Key Features

*   **Comprehensive Data Preprocessing:** Includes handling missing data, encoding categorical variables, feature scaling, and feature engineering.
*   **Custom SVM Implementation:** Provides a `KernelSVM` class built from scratch, offering insights into the inner workings of SVMs.
*   **Multiple Kernel Support:** Implements and compares linear, quadratic, polynomial (degree 2), and RBF kernels.
*   **Performance Evaluation:** Uses standard metrics like precision, recall, and F1-score, along with accuracy.
*   **Rich Visualizations:** Generates plots for EDA, loss curves, accuracy curves, and decision boundaries to aid in understanding model behavior and data characteristics.
*   **Modular Code:** The notebook is structured to separate data loading, preprocessing, model definition, training, and evaluation steps.


## Installation

Instructions on how to install and set up the project.

```bash
# Example installation commands
pip install -r requirements.txt
```

## Usage

To run this project:
1.  Ensure you have Python and the necessary libraries installed (see `requirements.txt`).
2.  Download the Titanic dataset (`train.csv` and `test.csv`) and place it in a directory accessible by the notebook, typically an `input/titanic/` subdirectory if running in an environment like Kaggle, or adjust the file paths in the notebook:
    ```python
    Train = pd.read_csv('/kaggle/input/titanic/train.csv')
    test = pd.read_csv('/kaggle/input/titanic/test.csv')
    ```
3.  Open and run the `svm-kernal.ipynb` notebook in a Jupyter environment.

The notebook will perform data preprocessing, train the SVM models with different kernels, and display evaluation metrics and visualizations.

## Project Structure

*   `svm-kernal.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model implementation, training, evaluation, and visualization.
*   `requirements.txt`: Lists the Python libraries required to run the project.
*   `README.md`: This file, providing an overview and documentation of the project.
*   `/kaggle/input/titanic/` (or your specified path): This directory is expected to contain `train.csv` and `test.csv` from the Titanic dataset.