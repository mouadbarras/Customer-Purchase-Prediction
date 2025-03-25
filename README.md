# Practical Exam: Customer Purchase Prediction
This project provides a solution to predict customer purchases for RetailTech Solutions. It presents a comprehensive pipeline that starts from raw data cleaning and feature engineering and culminates in building and training a neural network using PyTorch.

## Overview

This repository contains a solution for the practical exam assignment aimed at predicting customer purchases for **RetailTech Solutions** – a fast-growing international e-commerce platform operating in over 20 countries. The goal is to build a robust prediction system that leverages customer browsing behavior to forecast purchase likelihood, supporting the company's expansion and revenue growth strategies.

The project is divided into three main tasks:

- **Task 1: Data Cleaning**  
  Clean the raw customer session data (`raw_customer_data.csv`) by addressing missing values and inconsistencies. The cleaned output should be a DataFrame named `clean_data` with the following columns:
  - **customer_id:** Integer, unique identifier (no missing values).
  - **time_spent:** Float, minutes spent on website per session (missing values replaced with median).
  - **pages_viewed:** Integer, pages viewed in session (missing values replaced with mean).
  - **basket_value:** Float, value of items in basket (missing values replaced with 0).
  - **device_type:** String, one of: Mobile, Desktop, Tablet (missing values replaced with "Unknown").
  - **customer_type:** String, one of: New, Returning (missing values replaced with "New").
  - **purchase:** Binary, indicating if a purchase was made (target variable).

- **Task 2: Feature Engineering**  
  Prepare the pre-cleaned dataset (`model_data.csv`) for the neural network by:
  - Scaling numerical features (`time_spent`, `pages_viewed`, `basket_value`) to a 0-1 range.
  - Applying one-hot encoding to categorical features (`device_type`, `customer_type`) with column names formatted as `variable_name_category_name` (e.g., `device_type_Desktop`).

  The final output should be a DataFrame named `model_feature_set` containing all original columns (except those replaced by one-hot encoding).

- **Task 3: Neural Network Modeling**  
  Develop and train a neural network using PyTorch to predict customer purchases. Requirements include:
  - Building a network with at least one hidden layer (8 units) using ReLU activation.
  - Using Sigmoid activation for the output layer.
  - Training the model with features from `input_model_features.csv` and validating predictions with `validation_features.csv`.

  The trained model should be named `purchase_model`, and predictions should be output as a DataFrame named `validation_predictions` with columns `customer_id` and `purchase`.

## Project Structure

    ```plaintext
    ├── README.md
    ├── requirements.txt
    ├── raw_customer_data.csv         # Raw customer session data
    ├── model_data.csv                # Pre-cleaned data for feature engineering
    ├── input_model_features.csv      # Features for training the neural network
    ├── validation_features.csv       # Data for generating purchase predictions
    └── notebook.ipynb


## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/mouadbarras/Customer-Purchase-Prediction.git
   cd Customer-Purchase-Prediction

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt

## Usage

- **Data Cleaning (Task 1):**
  Run the `data_cleaning.py` script to load raw_customer_data.csv, clean the data, and output the DataFrame clean_data.

- **Feature Engineering (Task 2):**
Execute the feature_engineering.py script to transform model_data.csv by scaling numerical features and one-hot encoding categorical variables. The output will be the DataFrame model_feature_set.

- **Model Training and Prediction (Task 3):**
Use the train_model.py script to build and train the neural network using PyTorch. The script will generate predictions on validation_features.csv and save them as the DataFrame validation_predictions containing customer_id and purchase.

## Contact & Collaboration

For questions, suggestions, or collaboration opportunities, please reach out:

LinkedIn [Mouad BARRAS](https://www.linkedin.com/in/mouad-barras/)

## Author:

Mouad Barras
