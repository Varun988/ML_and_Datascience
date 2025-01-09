import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import numpy as np

# Streamlit app
st.title("Model Selection and Accuracy Testing")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset CSV file", type=["csv"])
if uploaded_file:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Allow the user to input the number of rows they want to sample
    num_rows = st.number_input("Select number of rows to take from the dataset:", min_value=1, max_value=len(data), value=int(len(data)*0.4), step=1)

    # Sample the specified number of rows
    data_sampled = data.sample(n=num_rows, random_state=42)
    
    st.write(f"Dataset Preview (Selected {num_rows} Rows):")
    st.dataframe(data_sampled.head())

    # Data Cleaning
    st.write("Cleaning the data...")

    # Handle missing values (Impute with median for numerical columns)
    numerical_columns = data_sampled.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data_sampled.select_dtypes(include=['object']).columns

    # Impute missing numerical values with median
    imputer = SimpleImputer(strategy='median')
    data_sampled[numerical_columns] = imputer.fit_transform(data_sampled[numerical_columns])

    # Impute missing categorical values with the most frequent value
    imputer_cat = SimpleImputer(strategy='most_frequent')
    data_sampled[categorical_columns] = imputer_cat.fit_transform(data_sampled[categorical_columns])

    # Encode categorical columns using Label Encoding
    label_encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        data_sampled[col] = encoder.fit_transform(data_sampled[col])
        label_encoders[col] = encoder

    # Display cleaned data preview
    st.write("Cleaned Dataset Preview:")
    st.dataframe(data_sampled.head())

    # Correlation matrix visualization
    st.subheader("Correlation Matrix")
    corr_matrix = data_sampled.corr()
    
    # Explicitly create the figure for the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    
    # Display the plot in Streamlit
    st.pyplot()
    plt.close()  # Close the plot to avoid issues in subsequent calls

    # Display the correlation matrix for the user's reference
    st.write("Correlation Matrix (shows feature relationships):")
    st.dataframe(corr_matrix)

    # Visualizations
    # Check for outliers using Z-score
    st.subheader("Outlier Detection (Z-Score)")
    z_scores = np.abs(stats.zscore(data_sampled[numerical_columns]))
    outliers = (z_scores > 3).sum(axis=0)
    st.write(f"Number of outliers detected in each numerical feature:")
    st.write(outliers)

    # Plot Outliers using Boxplot
    st.subheader("Outliers in Numerical Features (Boxplot)")
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=data_sampled[numerical_columns])
    st.pyplot()
    plt.close()

    # Skewness analysis
    st.subheader("Skewness of Numerical Features")
    skewness = data_sampled[numerical_columns].skew()
    st.write(f"Skewness of numerical features:")
    st.write(skewness)

    # Select features (exclude target column)
    target_column = st.selectbox("Select Target Column:", data_sampled.columns)
    features = st.multiselect("Select Features:", [col for col in data_sampled.columns if col != target_column])

    if st.button("Train Models"):
        if target_column and features:
            # Prepare data
            X = data_sampled[features]
            y = data_sampled[target_column]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # LazyClassifier
            clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
            models, predictions = clf.fit(X_train, X_test, y_train, y_test)

            # Display model performances
            st.write("Model Performances:")
            st.dataframe(models)

            # Model selection
            model_name = st.selectbox("Select a Model to Test:", models.index)

            if model_name:
                # Get the specific model and test accuracy
                selected_model = clf.models[model_name]
                selected_model.fit(X_train, y_train)
                y_pred = selected_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Display accuracy
                st.write(f"Accuracy of {model_name}: {accuracy:.2f}")
        else:
            st.error("Please select both target column and features.")
