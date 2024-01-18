# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 15:29:30 2024

@author: kurva
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

# Define a function to load and preprocess the data
def load_data(filename):
    # Load your data from a CSV file
    df = pd.read_csv(filename)
    pd.set_option('display.max_columns', None)

    # Summarize the data
    print(df.info())
    print(df.columns.tolist())
    print("DataFrame summary:\n", df.describe())

    # Drop rows with NaN values
    data = df.dropna()
    print("DataFrame after dropping rows with NaN values:\n", data.head())

    # Select relevant columns for clustering
    selected_columns = ['country_name', 'year', 'value']
    data = data[selected_columns]

    # Normalize the data for clustering
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[['year', 'value']])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters = 4)
    data['Cluster'] = kmeans.fit_predict(normalized_data)
    return data

# Define a function to analyze clusters and pick one country from each
def pick_countries(df):
    countries_to_compare = []
    kmeans = KMeans(n_clusters = 4)
    for cluster_num in range(kmeans.n_clusters):
        cluster_data = df[df['Cluster'] == cluster_num]

        # Added a missing newline after the previous statement
        selected_country = cluster_data.sample(1, random_state = 42)[
            'country_name'].values[0]
        countries_to_compare.append(selected_country)
    return countries_to_compare

# Define a function to compute and visualize cluster centers
def plot_cluster_centers(df, kmeans):
    plt.scatter(df['year'], df['value'], c = df['Cluster'],
                cmap = 'viridis', alpha = 0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                :, 1], s = 300, c = 'red', marker = '*', label = 'Cluster Centers')
    plt.title('Cluster Centers and Centroids')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Transposed DataFrame of given DataFrame.
def transpose(df):
    Trans_Data = df.transpose()
    return Trans_Data

# Moved the score calculation inside a function
def calculate_score(df):
    """Added a missing column name for year"""
    score = silhouette_score(df[['year', 'value']],
                             df['Cluster'], metric = 'euclidean')
    print(f'The average silhouette score is: {score:.3f}')

# Define a function to make predictions for future years and plot comparisons
def plot_comparisons(df, countries_to_compare):
    plt.figure(figsize = (12, 8))
    for i, country in enumerate(countries_to_compare):
        country_data = df[df['country_name'] == country]

        # Fit the curve to CO2 Emission_per_capita as a function of Year
        popt, pcov = curve_fit(
            quadratic_fit, country_data['year'], country_data['value'])

        # Make predictions for future years
        future_years = np.arange(
            country_data['year'].min(), country_data['year'].max() + 20, 1)
        predicted_gdp = quadratic_fit(future_years, *popt)

        # Plot the actual and predicted CO2 Emission per capita
        plt.subplot(2, 2, i+1)
        plt.scatter(country_data['year'], country_data['value'],
                    label = f'Actual CO2 emission per capita - {country}')
        plt.plot(future_years, predicted_gdp, color = 'green',
                 label = 'Predicted CO2 emission per capita (Quadratic Fit)')
        plt.title(f'CO2 emission per capita Trend Comparison - {country}')
        plt.xlabel('year')
        plt.ylabel('value')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Correlation_and_scattermatrix plots correlation matrix and scatter plots of data among columns
def correlation_and_scattermatrix(df):
    """Calculate the correlation matrix"""
    corr = df[['year', 'value']].corr()
    print(corr)

    # Plot the correlation matrix as a heatmap
    plt.figure(figsize = (10, 10))
    plt.matshow(corr, cmap = 'coolwarm')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title('Correlation between Years and Countries over Value')
    plt.colorbar()
    plt.show()

    # Plot scatter matrix
    pd.plotting.scatter_matrix(
        df[['year', 'value']], figsize = (12, 12), s = 5, alpha = 0.8)
    plt.show()

# Define a function for curve fitting (example: quadratic polynomial)
def quadratic_fit(x, a, b, c):
    return a * x**2 + b * x + c


filename = "C:\\Users\\kurva\\Downloads\\APPLIED DATA SCIENCE\\PROJECT - 3\\co2_emissions.csv"
df = load_data(filename)
kmeans = KMeans(n_clusters = 4)
df['Cluster'] = kmeans.fit_predict(df[['year', 'value']])
# Call the functions
countries_to_compare = pick_countries(df)
plot_cluster_centers(df, kmeans)
transpose(df)
calculate_score(df)
plot_comparisons(df, countries_to_compare)
correlation_and_scattermatrix(df)
