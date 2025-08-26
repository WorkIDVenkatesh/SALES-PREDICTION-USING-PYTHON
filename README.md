Sales Prediction with Linear Regression

This repository contains a Python script for a sales prediction project. The script uses a Linear Regression model to forecast sales based on advertising expenditure across different channels.
Overview

The project follows a standard machine learning pipeline:

    Data Loading: The script reads the advertising.csv dataset.

    Data Preprocessing: It identifies features (TV, Radio, Newspaper) and the target variable (Sales).

    Model Training: A Linear Regression model is trained on the preprocessed data.

    Prediction and Evaluation: The model's performance is evaluated using key regression metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²).

    Visualization: A scatter plot is generated to visually compare the model's predictions against the actual sales values.

File Contents

    sales_prediction.py: The main Python script that performs the entire sales prediction process.

    advertising.csv: The dataset used for training and testing the model.

How to Run the Script

    Dependencies: Ensure you have Python and pip installed. This project requires the following libraries. You can install them using the following command:

    pip install pandas numpy matplotlib seaborn scikit-learn

    Dataset: Make sure the advertising.csv file is in the same directory as the sales_prediction.py script.

    Execution: Open your terminal or command prompt, navigate to the project directory, and run the script with the following command:

    python sales_prediction.py

Expected Output

The script will print information about the data loading, model training, and evaluation results to the console. It will also display a plot visualizing the model's performance.

1. Loading the advertising dataset...
Dataset loaded successfully!

First 5 rows of the dataset:
      TV  Radio  Newspaper  Sales
0  230.1   37.8       69.2   22.1
1   44.5   39.3       45.1   10.4
2   17.2   45.9       69.3   12.0
3  151.5   41.3       58.5   16.5
4  180.8   10.8       58.4   17.9

2. Defining features and target variable...
Data split into training set (shape: (160, 3)) and testing set (shape: (40, 3)).

3. Training the Linear Regression model...
Model training complete.

4. Making predictions and evaluating the model...
Mean Absolute Error (MAE): 1.27
Mean Squared Error (MSE): 2.91
R-squared (R²): 0.91

Script execution finished. Check the plot for a visual representation of the model's performance.

