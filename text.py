import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Lasso
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Function to read CSV file
def read_csv_file():
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def perform_kmeans_clustering(df):
    n_clusters = st.number_input("Enter the number of clusters:", min_value=1, value=2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df)
    df['cluster'] = clusters
    st.write("\nAfter KMeans clustering:")
    st.write(df)
    return df

# Function for PCA dimensionality reduction
def perform_pca(df):
    n_components = st.number_input("Enter the number of components for dimensionality reduction (PCA): ", min_value=1, value=2)
    pca = IncrementalPCA(n_components=n_components, batch_size=100)
    transformed_df = pca.fit_transform(df)
    transformed_df = pd.DataFrame(transformed_df, columns=[f"component_{i+1}" for i in range(n_components)])  # Convert to DataFrame
    st.write("After PCA:")
    st.write(transformed_df)
    return transformed_df


# Function to update target column name
def update_target_column_name(df):
    target_column_name = st.text_input("Enter the name of the target column: ", value='target')
    if target_column_name in df.columns:
        df.rename(columns={target_column_name: 'target'}, inplace=True)
        st.write(f"\nUpdated target column name.")
        x = df['target']
    else:
        st.write(f"Column '{target_column_name}' not found in the DataFrame.")
    return x

# Function for handle missing values
def handle_missing_values(df):
    updated_df = df.dropna()
    st.write("After handling missing values:")
    st.write(updated_df)
    return updated_df

# Function for feature scaling
def scale_features(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include='number').columns
    scaled_features = scaler.fit_transform(df[numeric_cols])
    df[numeric_cols] = scaled_features
    st.write("\nAfter scaling features:")
    st.write(df)
    return df

# Function to fill missing numeric values
def fill_numeric_missing(df, method):
    numeric_cols = df.select_dtypes(include='number').columns
    if method == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif method == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    elif method == 'mode':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])
    st.write("\nAfter filling missing numeric values with {method}:")
    st.write(df)
    return df

# Function to fill missing string values
def fill_string_missing(df):
    df.fillna('N/A', inplace=True)
    st.write("\nAfter filling missing string values with 'N/A':")
    st.write(df)
    return df

# Function for bucketizing based on user-defined bins
def bucketize_column(df, column, bins, labels):
    df[column + '_bucketized'] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)
    st.write("\nAfter bucketizing column:")
    st.write(df)
    return df

# Function for train-test split
def perform_train_test_split(df,y):
    test_size = st.number_input("Enter the percentage of data to use for testing (e.g., 20 for 20%): ", min_value=0, max_value=100, step=1, value=20)
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
    st.write("\nX_Train data:")
    st.write(X_train)
    st.write("\nX_Test data:")
    st.write(X_test)
    st.write("\ny_Train data:")
    st.write(y_train)
    st.write("\ny_Test data:")
    st.write(y_test)
    return X_train, X_test, y_train, y_test

# Function to encode categorical variables
def encode_categorical_variables(df):
    encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = encoder.fit_transform(df[column])
    updated_df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns)
    st.write("\nAfter encoding categorical variables:")
    st.write(updated_df)
    return updated_df

from sklearn.linear_model import LassoCV

def perform_lasso_feature_selection(df):
    target_col_index = st.number_input("Enter the number corresponding to the target column: ", min_value=0, max_value=len(df.columns)-1, value=0)
    
    X = df.drop(columns=[df.columns[target_col_index]])
    y = df[df.columns[target_col_index]]

    for col in X.select_dtypes(include=['object']):
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=5)  # Use cross-validation to select the best alpha
    lasso.fit(X_scaled, y)

    # Get selected features
    selected_features = X.columns[lasso.coef_ != 0]
    # Print selected feature names
    st.write("\nSelected features:")
    st.write(selected_features)
    return selected_features.tolist()

def evaluate_accuracy(X_train, X_test, y_train, y_test):
    # Train a Logistic Regression model on the original data
    model_orig = LogisticRegression(random_state=70)
    model_orig.fit(X_train, y_train)
    y_pred_orig = model_orig.predict(X_test)
    acc_orig = accuracy_score(y_test, y_pred_orig)
    return acc_orig

def drop_duplicates(df):
    df.drop_duplicates(inplace=True)
    st.write("\nAfter dropping duplicate rows:")
    st.write(df)
    return df

# Function to remove outliers
def remove_outliers(df):
    clf = OneClassSVM(nu=0.01)  # nu is the fraction of outliers you expect
    clf.fit(df)

    # Predict outliers
    outliers = clf.predict(df)
    outlier_indices = df.index[outliers == -1].tolist()

    # Remove outliers from the dataset
    df_cleaned = df.drop(outlier_indices)
    # Print updated dataset
    st.write("Outlier data:")
    st.write(df.loc[outlier_indices])
    st.write("\nUpdated dataset:")
    st.write(df_cleaned)
    return df_cleaned



def perform_operation():
        st.write("App allows users to perform data integration and preprocessing on CSV files. Users can upload CSV files, integrate them, and preprocess them using a variety of steps such as handling missing values, scaling features, encoding categorical variables, and more.")
        df = read_csv_file()
        if df is not None:
            y=update_target_column_name(df)
            steps = st.multiselect("Select preprocessing steps:", ['Handle missing values','Drop Duplicates', 'Fill missing numeric values', 'Fill missing string values', 'Encode categorical variables', 'Scale features', 'Lasso feature selection', 'PCA dimensionality reduction', 'Train-test split', 'Remove outliers', 'Clustering', 'Bucketize column'])
            if len(steps) > 0:
                for step in steps:
                    if step == 'Handle missing values':
                        df = handle_missing_values(df)
                    elif step == 'Drop duplicates':
                        df = drop_duplicates(df)
                    elif step == 'Fill missing numeric values':
                        fill_method = st.selectbox("Select method for filling missing numeric values:", ['mean', 'median', 'mode'])
                        df = fill_numeric_missing(df, fill_method)
                    elif step == 'Fill missing string values':
                        df = fill_string_missing(df)
                    elif step == 'Encode categorical variables':
                        df = encode_categorical_variables(df)
                    elif step == 'Scale features':
                        df = scale_features(df)
                    elif step == 'Lasso feature selection':
                        perform_lasso_feature_selection(df)
                    elif step == 'PCA dimensionality reduction':
                        df  = perform_pca(df)
                    elif step == 'Remove outliers':
                        remove_outliers(df)
                    elif step == 'Train-test split':
                        X_train, X_test, y_train, y_test=perform_train_test_split(df,y)
                    elif step == 'Clustering':
                        df = perform_kmeans_clustering(df)
                    elif step == 'Bucketize column':
                        column = st.selectbox("Select column to bucketize:", df.columns)
                        bins = st.number_input("Enter the number of bins:", min_value=1, value=3)
                        labels = [str(i) for i in range(bins)]
                        df = bucketize_column(df, column, bins=bins, labels=labels)

                st.write("\nFinal preprocessed data:")
                st.write(df)
                
# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Data Nexus App", page_icon=":bar_chart:")
    st.title("Data Nexus - Text Preprocessing")
    perform_operation()

if __name__ == "__main__":
    main()
