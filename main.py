import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load data from a file into a pandas DataFrame.

    Args:
        file_path (str): The path to the file containing the data.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.
    """
  
    if file_path == "glass.data" or file_path == "iris.data":
        # Read the data from the file into a list of lines
        with open(file_path, 'r') as file:
            lines = file.readlines()
        data = [line.strip().split(',') for line in lines]
    
        # Create the DataFrame
        data_frame = pd.DataFrame(data)
        
        # Skip the first column if file_path is "glass.data"  
        if file_path == "glass.data":     
            data_frame = data_frame.iloc[:, 1:]
        else:
            label_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
            data_frame.iloc[:, -1] = data_frame.iloc[:, -1].map(label_map)
            data_frame = data_frame[:-1]

        # Convert data to numeric type
        data_frame = data_frame.apply(pd.to_numeric, errors='coerce')

    else:
        # Read the data from the file into a list of lists, stripping leading tabs
        with open(file_path, 'r') as file:
            lines = file.readlines()
            data = [[item.strip() for item in line.split('\t')] for line in lines]

        # Create the DataFrame
        data_frame = pd.DataFrame(data)
        
        # Convert data to numeric type
        data_frame = data_frame.apply(pd.to_numeric, errors='coerce')

    return data_frame

#
# def initialize_centroids(data, k):
#     """Initialize centroids using the K-means++ algorithm."""
#     if k <= 0 or k > len(data):
#             raise ValueError("Invalid number of centroids (k)")
#
#     result = data.sample(n=1).copy()
#     remaining_data = data.drop(result.index)
#
#     for _ in range(k - 1):
#         distances = np.inf * np.ones(len(remaining_data))
#         result_array = result.to_numpy()
#         remaining_data_array = remaining_data.to_numpy()
#
#         for centroid in result_array:
#             current_distances = np.sum((remaining_data_array - centroid) ** 2, axis=1)
#             distances = np.minimum(distances, current_distances)
#
#         new_centroid_idx = np.argmax(distances)
#         new_centroid = remaining_data.iloc[[new_centroid_idx]]
#         result = pd.concat([result, new_centroid], ignore_index=True)
#         remaining_data = remaining_data.drop(new_centroid.index)
#
#     return result
#

def initialize_centroids(dataset, num_centroids):
    """Select initial centroids using the K-means++ algorithm."""

       # Randomly select the first centroid
    centroids = dataset.sample(n=1).copy()
    # Remove the selected centroid from the dataset
    remaining_points = dataset.drop(centroids.index)

    # Select the remaining centroids
    for _ in range(num_centroids - 1):
        # Initialize minimum distances to infinity
        min_distances = np.inf * np.ones(len(remaining_points))
        # Convert centroids and remaining points to numpy arrays for efficient computation
        centroids_array = centroids.to_numpy()
        remaining_points_array = remaining_points.to_numpy()

        # Compute distances from each remaining point to the nearest centroid
        for point in centroids_array:
            current_distances = np.sum((remaining_points_array - point) ** 2, axis=1)
            min_distances = np.minimum(min_distances, current_distances)
        # Select the next centroid based on the maximum minimum distance
        next_centroid_idx = np.argmax(min_distances)
        next_centroid = remaining_points.iloc[[next_centroid_idx]]
        # Add the new centroid to the list of centroids
        centroids = pd.concat([centroids, next_centroid], ignore_index=True)
        # Remove the new centroid from the remaining points
        remaining_points = remaining_points.drop(next_centroid.index)

    return centroids
def assign_clusters(data_set, centroids):
    """
    Assign each point in the data_set to the nearest centroid.

    Args:
        data_set (pd.DataFrame): The dataset containing the points.
        centroids (pd.DataFrame): The centroids of the clusters.

    Returns:
        np.array: An array of cluster assignments.
    """
    distances = np.linalg.norm(data_set.values[:, np.newaxis, :] - centroids.values, axis=2)
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments

def update_centroids_kmeans(data_set, cluster_assignments, k):
    """
    Update the centroids based on the current cluster assignments using K-means.

    Args:
        data_set (pd.DataFrame): The dataset containing the points.
        cluster_assignments (np.array): The current cluster assignments for each point.
        k (int): The number of clusters.

    Returns:
        pd.DataFrame: A DataFrame containing the updated centroids.
    """
    new_centroids = data_set.groupby(cluster_assignments).mean().values
    return pd.DataFrame(new_centroids, columns=data_set.columns)

def update_centroids_kmedian(data_set, cluster_assignments, k):
    """
    Update the centroids based on the current cluster assignments using K-median.

    Args:
        data_set (pd.DataFrame): The dataset containing the points.
        cluster_assignments (np.array): The current cluster assignments for each point.
        k (int): The number of clusters.

    Returns:
        pd.DataFrame: A DataFrame containing the updated centroids.
    """
    new_centroids = data_set.groupby(cluster_assignments).median().values
    return pd.DataFrame(new_centroids, columns=data_set.columns)

def kmeans(data_set, k, max_iterations=1000, tolerance=1e-5, use_kmedian=False):
    """
    Perform K-means or K-median clustering on the dataset.

    Args:
        data_set (pd.DataFrame): The dataset containing the points.
        k (int): The number of clusters.
        max_iterations (int): The maximum number of iterations to perform.
        tolerance (float): The convergence threshold.
        use_kmedian (bool): Whether to use K-median instead of K-means.

    Returns:
        pd.DataFrame: The data with cluster assignments.
        pd.DataFrame: The final centroids.
    """
    centroids = initialize_centroids(data_set, k)
    iteration_number = 0
    for _ in range(max_iterations):
        cluster_assignments = assign_clusters(data_set, centroids)
        if use_kmedian:
            new_centroids = update_centroids_kmedian(data_set, cluster_assignments, k)
        else:
            new_centroids = update_centroids_kmeans(data_set, cluster_assignments, k)
        # Check for convergence
        if np.all(np.abs(new_centroids.values - centroids.values) < tolerance):
            centroids = new_centroids
            break

        centroids = new_centroids
        iteration_number += 1

    data_set['Cluster'] = cluster_assignments
    return data_set, centroids, iteration_number



def calculate_purity(clustered_data, true_labels):
    """
    Calculate the purity of the clustering.

    Args:
        clustered_data (pd.DataFrame): The data with cluster assignments.
        true_labels (pd.Series): The true labels for the data points.

    Returns:
        float: The purity score.
    """
    total = len(true_labels)
    correctly_assigned = 0

    for cluster in np.unique(clustered_data['Cluster']):
        cluster_points = true_labels[clustered_data['Cluster'] == cluster]
        if len(cluster_points) == 0:
            continue
        most_common_label = cluster_points.value_counts().idxmax()
        correctly_assigned += np.sum(cluster_points == most_common_label)

    purity = correctly_assigned / total
    return purity

def run_multiple_times(data, true_labels, k, runs=10, use_kmedian=False):
    """
    Run the clustering algorithm multiple times and print results.

    Args:
        data (pd.DataFrame): The dataset containing the points.
        true_labels (pd.Series): The true labels for the data points.
        k (int): The number of clusters.
        runs (int): The number of times to run the clustering algorithm.
        use_kmedian (bool): Whether to use K-median instead of K-means.

    Returns:
        pd.DataFrame: The best clustered data.
        pd.DataFrame: The best centroids.
        float: The best purity score.
    """
    best_purity = 0
    best_clustered_data = None
    best_centroids = None

    for _ in range(runs):
        clustered_data, centroids, iteration_number = kmeans(data, k, use_kmedian=use_kmedian)
        purity = calculate_purity(clustered_data, true_labels)
        if purity > best_purity:
            best_purity = purity
            best_clustered_data = clustered_data.copy()
            best_centroids = centroids.copy()

    return best_clustered_data, best_centroids, best_purity, iteration_number

def plot_clusters(clustered_data, centroids, label_number, title):
    """
    Plot the clustered data along with centroids.

    Args:
        clustered_data (pd.DataFrame): The clustered data.
        centroids (pd.DataFrame): The centroids of the clusters.
        label_number (int): The number of clusters.
        title (str): The title for the plot.
    """
    plt.figure(figsize=(15, 9))
    for cluster in range(label_number):
        cluster_points = clustered_data[clustered_data['Cluster'] == cluster]
        plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], label=f'Cluster {cluster}')

    plt.scatter(centroids.iloc[:, 0], centroids.iloc[:, 1], s=300, c='red', label='Centroids', marker='X')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.show()



# # Load the data
# data_set = load_data("D31.data")

# # Determine the number of clusters (assuming the last column is the label)
# label_number = len(set(data_set.iloc[:, -1]))

# # Extract true labels
# true_labels = data_set.iloc[:, -1]

# # Run clustering multiple times for K-means
# best_clustered_data_kmeans, best_centroids_kmeans, best_purity_kmeans = run_multiple_times(data_set.iloc[:, :-1], true_labels, label_number)

# # Run clustering multiple times for K-median
# best_clustered_data_kmedian, best_centroids_kmedian, best_purity_kmedian = run_multiple_times(data_set.iloc[:, :-1], true_labels, label_number, use_kmedian=True)

# # Print the best results
# print(f"Best K-means Purity: {best_purity_kmeans:.4f}")
# print(f"Best K-median Purity: {best_purity_kmedian:.4f}")





