import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids
import time

DATASETS = ['base', 'var1', 'var2', 'var3', 'var4', 'var5']
DATA_DIR =  os.path.join(os.getcwd(), '..', 'data')


def load_and_preproccess_data(variant):
    """
    Loads and preprocesses the dataset, including scaling and encoding categorical variables.

    Parameters:
    - variant: str, name of the dataset variant to load. Must be one of ['base', 'var1', 'var2', 'var3', 'var4', 'var5'].

    Returns:
    - df: DataFrame containing the preprocessed data.
    - categorical_mappings: List of dictionaries containing the mappings for each categorical variable.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, f"{variant}.csv"))
    
    df = df.drop(['device_fraud_count'], axis=1)
    
    categorical_fileds = ["payment_type", "employment_status", "housing_status", "source", "device_os"]
    categorical_mappings = []
    for field in categorical_fileds:
        unique_values = df[field].unique()
        data_maping = {}

        for i in range(len(unique_values)):
            data_maping[unique_values[i]] = i
        
        df[field] = df[field].map(data_maping)
        categorical_mappings.append(data_maping)
        
    bool_fields = []

    for col in df.columns:
        if df[col].nunique() == 2:
            bool_fields.append(col)
            
    pre_scaled = ["income","customer_age" "payment_type", "employment_status", "housing_status", "source", "device_os", "month"]
    pre_scaled.extend(bool_fields)
    
    for col in df.columns:
        if col not in pre_scaled:
            df[col] = StandardScaler().fit_transform(df[col].values.reshape(-1, 1))
            
    return df, categorical_mappings


def generate_X_y(df, sampling=None, pca=None):
    """
    Generate X and y from the dataset with optional PCA and sampling techniques.

    Parameters:
    - df: DataFrame containing the data, including the 'fraud' column.
    - sampling: str, type of sampling ('smote', 'ros', or 'under').
    - pca: int, number of PCA components to reduce features.

    Returns:
    - X: Feature matrix.
    - y: Target labels.
    """
    # Step 1: Apply PCA if specified
    if pca:
        pca = PCA(n_components=pca)
        X_PCA = pca.fit_transform(df.drop(['fraud_bool'], axis=1))
        y_PCA = df['fraud_bool']
        df = pd.DataFrame(X_PCA, columns=[f'PCA_{i}' for i in range(pca.n_components_)])
        df['fraud_bool'] = y_PCA

    # Step 2: Apply Sampling Techniques if specified
    if sampling:
        if sampling == 'smote':
            smote = SMOTE()
            X, y = smote.fit_resample(df.drop(['fraud_bool'], axis=1), df['fraud_bool'])
        elif sampling == 'ros':
            ros = RandomOverSampler()
            X, y = ros.fit_resample(df.drop(['fraud_bool'], axis=1), df['fraud_bool'])
        elif sampling == 'under':
            # Perform undersampling
            df_fraud = df[df['fraud_bool'] == 1]  # Filter fraud cases
            df_not_fraud = df[df['fraud_bool'] == 0]  # Filter non-fraud cases
            df_not_fraud_sampled = df_not_fraud.sample(n=len(df_fraud))
            new_df = pd.concat([df_fraud, df_not_fraud_sampled]).sample(frac=1)
            X = new_df.drop(['fraud_bool'], axis=1)
            y = new_df['fraud_bool']
        else:
            raise ValueError("Sampling must be one of ['smote', 'ros', 'under']")
    else:
        X = df.drop(['fraud_bool'], axis=1)
        y = df['fraud_bool']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def cluster(X_train, X_test, y_train, y_test, method='kmeans', n_clusters=10, fraud_threshold=0.5):
    """
    Perform clustering using specified method and compute metrics.

    Parameters:
    - X: Feature matrix.
    - y: Ground truth labels.
    - method: str, clustering method ('kmeans', 'gmm').
    - n_clusters: Number of clusters.
    - fraud_threshold: Threshold for identifying fraud clusters.

    Returns:
    - tp: True positives.
    - fp: False positives.
    - fn: False negatives.
    - tn: True negatives.
    """
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
        y_pred = model.fit(X_train).predict(X_test)
    elif method == 'kmeans++':
        model = KMeans(n_clusters=n_clusters, init='k-means++')
        y_pred = model.fit(X_train).predict(X_test)
    elif method == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        y_pred = model.fit(X_train).predict(X_test)
    elif method == 'gmm':
        model = GaussianMixture(n_components=n_clusters)
        y_pred = model.fit(X_train).predict(X_test)
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_jobs=-1)
        y_pred = model.fit(X_train).predict(X_test)
    elif method == 'kmedoids':
        model = KMedoids(n_clusters=n_clusters)
        y_pred = model.fit(X_train).predict(X_test)
    else:
        raise ValueError("Invalid clustering method.")
    
    # Determine fraud clusters based on fraud threshold
    res = pd.DataFrame({'cluster': y_pred, 'fraud': y_test})
    res['pred'] = 0
    
    for cluster_id in np.unique(y_pred):
        cluster = res[res['cluster'] == cluster_id]
        fraud_percent = cluster['fraud'].sum() / len(cluster)
        if fraud_percent > fraud_threshold:
            res.loc[res['cluster'] == cluster_id, 'pred'] = 1
    
    cm = confusion_matrix(res['fraud'], res['pred'])
    return cm.ravel() # tp, fp, fn, tn


if __name__ == '__main__':
    res = []
    
    samplings = ['smote', 'ros', 'under', None]
    pcas = [5, 15, None]
    n_clusters = [5, 10, 30]
    methods = ['kmeans', 'kmeans++'] 
    overall_starttime = time.time()
    print("Starting clustering experiment...")
    print("=================================")
    
    for dataset in DATASETS:
        for sampling in samplings:
            for pca in pcas:
                for method in methods:
                    for num_clusters in n_clusters:
                        print(f"Running {dataset} with {sampling} sampling, {pca} PCA components, {method} clustering, and {num_clusters} clusters.")
                        df, _ = load_and_preproccess_data(dataset)
                        starttime = time.time()
                        X_train, X_test, y_train, y_test  = generate_X_y(df, sampling, pca)
                        tp, fp, fn, tn = cluster(X_train, X_test, y_train, y_test, method, num_clusters)
                        endtime = time.time() - starttime
                        res.append({'dataset': dataset, 
                                    'method': method, 
                                    'num_clusters': num_clusters, 
                                    'sampling': sampling, 
                                    'pca': pca, 
                                    'tp': tp,
                                    'fp': fp,
                                    'fn': fn,
                                    'tn': tn,
                                    'time': endtime})
                        
    res_df = pd.DataFrame(res)
    res_df.to_csv('clustering_results.csv', index=False)
    print("=================================")
    print(f"Clustering experiment completed in {(time.time() - overall_starttime) / 60:.2f} minutes.")
    print("Results saved to clustering_results.csv.")