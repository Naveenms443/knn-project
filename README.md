To run the code download the python

# Diabetes Prediction using K-Nearest Neighbors (KNN)

This project demonstrates how to predict whether a person has diabetes or not using the K-Nearest Neighbors (KNN) algorithm.

## Overview

The `diabetes_prediction.py` script implements the KNN algorithm to predict the likelihood of a person having diabetes based on their medical attributes. It calculates the Euclidean distance between the test instance and each instance in the training data, selects the k nearest neighbors, and predicts the class label based on majority voting.

## Installation

To use this project, you need Python installed on your system. You also need the following Python libraries:
- numpy
- pandas

You can install these libraries using pip:


## Usage

1. Clone the repository to your local machine:
  git clone https://github.com/your_username/diabetes-prediction.git

2. Navigate to the project directory:
   
cd diabetes-prediction


3. Run the `diabetes_prediction.py` script:




## python diabetes_prediction.py

This will predict whether the test instance has diabetes or not based on the KNN algorithm.

## Modifying Test Instance

You can modify the test instance in the `diabetes_prediction.py` script to predict the likelihood of diabetes for different individuals. Locate the `test_instance` variable and change the values in the array to represent the medical attributes of the individual you want to test.

## Dataset

The dataset used for this project is stored in the `Dataset.csv` file. It contains medical attributes of individuals, such as glucose level, blood pressure, BMI, etc., along with their diabetes status.
