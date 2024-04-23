
# Automated Data Integration and Preprocessing App 📊

This Streamlit app allows users to perform data integration and preprocessing on CSV files. Users can upload CSV files, integrate them, and preprocess them using a variety of steps such as handling missing values, scaling features, encoding categorical variables, and more.

## Features ✨

- **Integration:** Upload two CSV files to integrate them into a single dataset.
- **Preprocessing Steps:**
  - **Handle Missing Values:** Remove rows with missing values or fill them using mean, median, or mode.
  - **Scale Features:** Standardize numeric features.
  - **Train-Test Split:** Split the dataset into training and testing sets.
  - **Encode Categorical Variables:** Convert categorical variables into numerical labels or one-hot encoding.
  - **PCA Dimensionality Reduction:** Reduce the dimensionality of the dataset using Principal Component Analysis.
  - **Lasso Feature Selection:** Select features using Lasso regression.
  - **Remove Outliers:** Detect and remove outliers using One-Class SVM.
  - **Fill Missing Numeric Values:** Fill missing numeric values with mean, median, or mode.
  - **Fill Missing String Values:** Fill missing string values with a specified string.
  - **Bucketize Column:** Create bins for a numeric column and convert it into a categorical variable.

## Usage 🚀

1. Clone the repository:

   ```bash
   git clone https://github.com/parita2003/Data_Nexus_Automated_Preprocessing.git
   ```

2. Run the Streamlit app:

   ```bash
   streamlit run preprocessing.py
   ```

---
