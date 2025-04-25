# Credit-Card-Fraud-Detection-Using-Machine-Learning

## Project Overview

This project aims to detect fraudulent credit card transactions using various machine learning models. Due to the highly imbalanced nature of typical credit card transaction datasets (where fraudulent transactions are rare), this project explores techniques like Random Undersampling and SMOTE (Synthetic Minority Over-sampling Technique) to handle the imbalance before training classification models. Several models are implemented and evaluated based on standard metrics like Precision, Recall, and F1-score.

## Dataset

The dataset used in this project appears to be the [Credit Card Fraud Detection dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). This dataset contains transactions made by European cardholders in September 2013.

* **Features:** Contains 28 anonymized principal components (`V1` to `V28`) obtained via PCA, plus non-anonymized 'Time' and 'Amount' features.
* **Target:** The 'Class' column indicates whether a transaction is fraudulent (1) or legitimate (0).
* **Imbalance:** The dataset is highly imbalanced, with only a small fraction of transactions being fraudulent.

*(Note: If you used a different dataset, please update this section accordingly.)*

## Project Structure

.├── balancedcreditcardfraud.ipynb  # Main Jupyter Notebook with analysis and modeling├── creditcard.csv                 # Dataset file (ensure this is not committed if large)├── requirements.txt               # Python dependencies├── README.md                      # Project description (this file)└── .gitignore                     # Specifies intentionally untracked files for Git
## Methodology

1.  **Data Loading & Preprocessing:**
    * Load the dataset using Pandas.
    * Standardize the 'Time' and 'Amount' features using `StandardScaler` from scikit-learn.

2.  **Exploratory Data Analysis (EDA):**
    * Visualize the class distribution of the original dataset.
    * Analyze feature correlations using heatmaps on both the original imbalanced data and a balanced subsample.

3.  **Handling Class Imbalance:**
    * **Random Undersampling:** Create a balanced subset by randomly selecting non-fraudulent samples equal to the number of fraudulent samples. Most models (MLP, RF, LR, KNN, SVM) are trained on this undersampled data.
    * **SMOTE (Synthetic Minority Over-sampling Technique):** Generate synthetic samples for the minority (fraud) class in the *training set* only. A Neural Network model is specifically trained using SMOTE-resampled data for comparison.

4.  **Model Training & Evaluation:**
    * Split the data (either the undersampled subset or the original data before SMOTE) into training and testing sets.
    * Implement and train the following models:
        * Neural Network (Keras/TensorFlow) - Trained on both undersampled and SMOTE data.
        * Multi-Layer Perceptron (MLP) (Keras/TensorFlow) - Trained on undersampled data.
        * Random Forest Classifier (scikit-learn) - Trained on undersampled data.
        * Logistic Regression (scikit-learn) - Trained on undersampled data.
        * K-Nearest Neighbors (KNN) (scikit-learn) - Trained on undersampled data.
        * Support Vector Machine (SVM) (scikit-learn) - Trained on undersampled data.
    * Evaluate models using:
        * **Confusion Matrix:** To visualize true positives, true negatives, false positives, and false negatives.
        * **Classification Report:** To calculate Precision, Recall, and F1-score for both classes (Legitimate and Fraud).

5.  **Results Comparison:**
    * Visualize the key performance metrics (Precision, Recall, F1-Score) for all tested models using a bar chart to compare their effectiveness.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Aryy10zz/Credit-Card-Fraud-Detection-Using-Machine-Learning.git](https://github.com/Aryy10zz/Credit-Card-Fraud-Detection-Using-Machine-Learning.git)
    cd Credit-Card-Fraud-Detection-Using-Machine-Learning
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # .\venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure you have the `creditcard.csv` dataset in the project's root directory (or update the path in the notebook).
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook
    # or
    jupyter lab
    ```
3.  Open the `balancedcreditcardfraud.ipynb` notebook.
4.  Run the cells sequentially to perform data loading, preprocessing, EDA, model training, and evaluation.

## Results Summary

The project evaluates multiple models on their ability to detect fraudulent transactions after addressing class imbalance. Key findings include:

* Undersampling helps create a balanced dataset for training, but potentially loses information from the majority class.
* SMOTE addresses imbalance by creating synthetic minority samples, potentially leading to better generalization for some models (like the Neural Network in this case, based on the comparison plot).
* Different models exhibit varying performance trade-offs between Precision (minimizing false positives) and Recall (minimizing false negatives). The comparison plot in the notebook (`results_df.plot`) visually summarizes the Precision, Recall, and F1-scores for each model tested on the *undersampled* dataset (except for 'NN+S' which uses SMOTE). The Random Forest ('RF') model shows strong performance based on the provided results data.

*(Refer to the final cells and plots in the `balancedcreditcardfraud.ipynb` notebook for detailed metrics and visualizations.)*

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you find bugs or have suggestions for improvements.

## License

[Specify License Here - e.g., MIT License]

