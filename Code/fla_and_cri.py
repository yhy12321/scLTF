from scipy.sparse import csr_matrix
from scipy import spatial
import numpy as np
import gc
import community
from functools import partial
import logging
import networkx as nx
from multiprocessing import Pool



# Set up logging
class Logger:
    # Logger class for standardized logging with timestamp, level, and message
    # Supports info, warning, and error messages
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        logg = logging.StreamHandler()
        logg.setFormatter(fmt)
        self.logger.addHandler(logg)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)



def get_k_nearest_indices(distance_matrix, num_neighbors, return_distances=True):
    """
        Compute the k nearest neighbors for each row in the distance matrix.

        Args:
            distance_matrix (np.ndarray): 2D array of pairwise distances (shape: n_samples x n_samples)
            num_neighbors (int): Number of nearest neighbors to retrieve
            return_distances (bool): If True, also return the distances of the nearest neighbors

        Returns:
            nearest_idx (np.ndarray): Indices of the k nearest neighbors for each row
            nearest_distances (np.ndarray, optional): Distances to the k nearest neighbors (if return_distances=True)
        """
    sample_idx = np.arange(distance_matrix.shape[0])[:, None]
    nearest_idx = np.argpartition(distance_matrix, num_neighbors, axis=1)[:, :num_neighbors]
    nearest_idx = nearest_idx[sample_idx, np.argsort(distance_matrix[sample_idx, nearest_idx])]
    if return_distances:
        nearest_distances = distance_matrix[sample_idx, nearest_idx]
        return nearest_idx, nearest_distances
    else:
        return nearest_idx


def compute_pairwise_distances(X, Y, metric='euclidean'):
    """
       Compute pairwise distances between rows of X and Y using the specified metric.

       Args:
           X (np.ndarray): First data matrix (n_samples_x x n_features)
           Y (np.ndarray): Second data matrix (n_samples_y x n_features)
           metric (str): Distance metric to use. Options: 'L1', 'sqeuclidean', 'euclidean', 'cosine'

       Returns:
           distances (np.ndarray): Matrix of pairwise distances (n_samples_x x n_samples_y)
       """
    if metric == 'L1':
        distances = spatial.distance.cdist(X, Y, metric='minkowski', p=1)
    elif metric == 'sqeuclidean':
        distances = spatial.distance.cdist(X, Y, metric='sqeuclidean')
    elif metric == 'euclidean':
        distances = spatial.distance.cdist(X, Y, metric='euclidean')
    elif metric == 'cosine':
        distances = spatial.distance.cdist(X, Y, metric='cosine')
    else:
        print('unknown metric')
    return distances

def kmeans_for_anchors(data, k, max_iterations=20):
    """
        Perform K-means clustering to select anchor points from the data.

        Args:
            data (np.ndarray): Input data matrix (n_samples x n_features)
            k (int): Number of clusters / anchors to generate
            max_iterations (int): Maximum number of iterations

        Returns:
            labels (np.ndarray): Cluster labels for each data point
            centers (np.ndarray): Cluster centers (anchors)
        """
    num_samples, num_features = data.shape
    labels = np.empty(num_samples, dtype=int)
    # Initialize cluster centers randomly
    np.random.seed(1)
    centers = data[np.random.choice(num_samples, k, replace=False)]
    iteration: int = 1
    while True:
        # Reinitialize centers if some clusters are empty (after first iteration)
        if (iteration != 1 and len(np.unique(labels)) != k):
            np.random.seed(2)
            centers = data[np.random.choice(num_samples, k, replace=False)]
        # Compute distances from data points to cluster centers
        distance = compute_pairwise_distances(data, centers)
        # Assign each point to the nearest cluster
        new_labels = np.argmin(distance, axis=1)
        # Check for convergence
        if (new_labels == labels).all():
            return labels, centers
        # Check for max iterations
        if (iteration >= max_iterations):
            return labels, centers
        # Update labels
        labels[:] = new_labels
        # Update cluster centers
        for i in range(k):
            cluster_points = data[labels == i]
            centers[i] = np.mean(cluster_points, axis=0)
        iteration += 1



def selectRepresentativeAnchors(data_matrix, num_anchors, oversample_factor=10, seed=1):
    """
        Select representative anchor points from the data matrix using oversampling and K-means.

        Args:
            data_matrix (np.ndarray): Input data matrix (n_samples x n_features)
            num_anchors (int): Number of representative anchors to select
            oversample_factor (int): Factor to select more candidate points before clustering
            seed (int): Random seed for reproducibility

        Returns:
            representative_anchors (np.ndarray): Selected anchor points
        """
    num_samples = data_matrix.shape[0]
    # Determine the number of candidate points to select
    num_candidates = oversample_factor * num_anchors
    # Ensure the number of anchors and candidates does not exceed the number of samples
    if num_anchors > num_samples:
        num_anchors = num_samples
    if num_candidates > num_samples:
        num_candidates = num_samples
    # Randomly select candidate points
    np.random.seed(seed)
    candidate_indices = np.random.choice(num_samples, num_candidates, replace=False)
    candidate_matrix = data_matrix[candidate_indices, :]
    # Apply K-means clustering to select the representative anchors
    label, representative_anchors = kmeans_for_anchors(candidate_matrix, num_anchors)  # max_iterations=20
    return representative_anchors



def nearest_anchor_in_cluster(expression_matrix_subset, anchor_cluster_labels, nearest_anchor_idx,
                            nearest_anchor_cluster_idx, anchors_subset, distance_metric, i):
    """
        Find the nearest anchor(s) within a specific cluster for a subset of expression data.

        Args:
            expression_matrix_subset (np.ndarray): Subset of the gene expression matrix
            anchor_cluster_labels (np.ndarray): Cluster labels of anchors
            nearest_anchor_idx (np.ndarray): Array to store nearest anchor indices for each point
            nearest_anchor_cluster_idx (np.ndarray): Cluster index for each row in expression_matrix_subset
            anchors_subset (np.ndarray): Subset of anchor points
            distance_metric (str): Distance metric to use ('euclidean', 'cosine', etc.)
            i (int): Current cluster index

        Returns:
            nearest_anchor_indices_in_cluster (np.ndarray or list): Indices of nearest anchors in cluster i
        """
    # Find indices of anchors belonging to the current cluster
    cluster_anchor_indices = np.where(anchor_cluster_labels == i)[0]
    if len(cluster_anchor_indices) > 0:
        # Compute distances from each point in the current cluster to all anchors in that cluster
        nearest_anchor_idx[nearest_anchor_cluster_idx == i] \
            = np.argmin(compute_pairwise_distances(expression_matrix_subset[nearest_anchor_cluster_idx == i, :],
             anchors_subset[cluster_anchor_indices, :], metric=distance_metric), axis=1)
        # Map the nearest indices back to the global anchor indices
        nearest_anchor_indices_in_cluster = cluster_anchor_indices[nearest_anchor_idx[nearest_anchor_cluster_idx == i]]
        return nearest_anchor_indices_in_cluster
    else:
        # Return empty list if no anchors exist in this cluster
        return []


def compute_sample_to_anchor_knn_distances(expression_subset, nearest_anchor_idx,
                                           Anchors, anchor_knn_indices, distance_metric, i):
    """
        Compute distances from samples in a cluster to the k-nearest neighbor anchors.

        Args:
            expression_subset (np.ndarray): Subset of the gene expression matrix
            nearest_anchor_idx (np.ndarray): Array storing nearest anchor indices for each sample
            Anchors (np.ndarray): Matrix of anchor points
            anchor_knn_indices (np.ndarray): Indices of k-nearest neighbor anchors for each anchor
            distance_metric (str): Distance metric to use ('euclidean', 'cosine', etc.)
            i (int): Cluster index

        Returns:
            distances (np.ndarray): Pairwise distances between samples in cluster i and their k-nearest anchors
        """
    distances = compute_pairwise_distances(expression_subset[nearest_anchor_idx == i, :],
                     Anchors[anchor_knn_indices[i, :], :],
                     metric=distance_metric)
    return distances


def generateAnchorsAndAdjacency(expression_matrix,
        p=1000,
        k=10,
        seed=1,
        use_multiprocessing=False,
        num_multiProcesses=4,
        distance_metric='euclidean',
        kernel_type='localscaled',
        mode='generateAnchorsAndAdjacency',
        ):
    """
        Generate representative anchors and construct a sample–anchor adjacency (bipartite) matrix.

        This function performs the following steps:
        1. Select p representative anchors from the gene expression matrix.
        2. Partition anchors into clusters and compute cluster centers.
        3. Assign each sample to its nearest anchor cluster and find the nearest anchor.
        4. Compute distances between samples and candidate neighbor anchors.
        5. Determine final k-nearest anchors for each sample.
        6. Construct a weighted sample–anchor bipartite adjacency matrix using the specified kernel
           ('localscaled' or 'traditionalscaled') or cosine similarity.

        Args:
            expression_matrix (np.ndarray): Input gene expression matrix (samples x features)
            p (int): Number of representative anchors to select
            k (int): Number of nearest neighbors for constructing adjacency
            seed (int): Random seed for reproducibility
            use_multiprocessing (bool): Whether to use multiprocessing
            num_multiProcesses (int): Number of processes if multiprocessing is used
            distance_metric (str): Distance metric to compute distances ('euclidean', 'cosine', etc.)
            kernel_type (str): Kernel type for adjacency weights ('localscaled' or 'traditionalscaled')
            mode (str): Mode of operation; controls logging messages

        Returns:
            Anchors (np.ndarray): Selected representative anchors
            sample_anchor_matrix (csr_matrix): Weighted sample–anchor bipartite adjacency matrix
        """

    N = expression_matrix.shape[0] # Number of samples
    if p > N:
        p = N

    logg = Logger()

    # -----------------------
    # 1. Select representative anchors
    # -----------------------
    if mode == 'generateAnchorsAndAdjacency':
        logg.info('Selecting representative Anchors...')
    Anchors = selectRepresentativeAnchors(expression_matrix, p, seed=seed)

    # -----------------------
    # 2. Cluster anchors to compute cluster centers
    # -----------------------
    if mode == 'generateAnchorsAndAdjacency':
        logg.info('Computing neighbors for anchors...')
    numAnchorClusters = int(p ** 0.5)
    # 2. find the center of each rep-cluster
    if distance_metric == 'euclidean':
        anchor_cluster_labels, anchor_cluster_centers = kmeans_for_anchors(Anchors, numAnchorClusters)
        # Using k-means with max 20 iterations and 1 random initialization
    else:
        anchor_cluster_labels, anchor_cluster_centers = kmeans_for_anchors(Anchors, numAnchorClusters)

    # Pre-compute distance from each sample to anchor cluster centers
    sample_to_cluster_center_dist = compute_pairwise_distances(expression_matrix,
                                                               anchor_cluster_centers, metric=distance_metric)
    del anchor_cluster_centers
    gc.collect()
    nearest_anchor_cluster_idx = np.argmin(sample_to_cluster_center_dist, axis=1)
    nearest_anchor_idx = np.empty(N, dtype='int64')

    # -----------------------
    # 3. Assign nearest anchor for each sample
    # -----------------------
    if use_multiprocessing == False:
        # Single-process assignment
        for i in range(numAnchorClusters):
            anchors_in_cluster = np.where(anchor_cluster_labels == i)[0]
            nearest_anchor_idx[nearest_anchor_cluster_idx == i] = np.argmin(compute_pairwise_distances(expression_matrix[nearest_anchor_cluster_idx == i, :],
                                                                             Anchors[anchors_in_cluster, :], metric=distance_metric), axis=1)
            nearest_anchor_idx[nearest_anchor_cluster_idx == i] = anchors_in_cluster[nearest_anchor_idx[nearest_anchor_cluster_idx == i]]
        del anchor_cluster_labels, nearest_anchor_cluster_idx, anchors_in_cluster

    else:
        # Multi-process assignment
        pool = Pool(num_multiProcesses)
        func = partial(nearest_anchor_in_cluster, expression_matrix,
                       anchor_cluster_labels, nearest_anchor_idx, nearest_anchor_cluster_idx, Anchors, distance_metric)
        nearest_anchor_knn_per_cluster = pool.map(func, np.arange(0, numAnchorClusters, 1))
        for i in range(numAnchorClusters):
            nearest_anchor_idx[nearest_anchor_cluster_idx == i] = nearest_anchor_knn_per_cluster[i]
        del nearest_anchor_knn_per_cluster
    gc.collect()

    # -----------------------
    # 4. Compute distances to candidate anchor neighbors
    # -----------------------
    num_neighbors = 5 * k
    anchor_pairwise_distances = compute_pairwise_distances(Anchors, Anchors, metric=distance_metric)
    if num_neighbors > anchor_pairwise_distances.shape[0]:
        num_neighbors = anchor_pairwise_distances.shape[0] - 2
    anchor_knn_indices = get_k_nearest_indices(anchor_pairwise_distances, num_neighbors + 1, return_distances=False)

    del anchor_pairwise_distances
    gc.collect()
    sample_to_anchor_knn_distances = np.zeros([N, np.shape(anchor_knn_indices)[1]])

    if use_multiprocessing == False:
        for i in range(p):
            sample_to_anchor_knn_distances[nearest_anchor_idx == i, :] = compute_pairwise_distances(expression_matrix[nearest_anchor_idx == i, :],
                                                                     Anchors[anchor_knn_indices[i, :], :],
                                                                     metric=distance_metric)
    else:
        pool = Pool(num_multiProcesses)
        sample_to_anchor_knn_distances_func = partial(compute_sample_to_anchor_knn_distances,
                                    expression_subset=expression_matrix,
                                    nearest_anchor_idx=nearest_anchor_idx,
                                    Anchors=Anchors,
                                    anchor_knn_indices=anchor_knn_indices,
                                    distance_metric=distance_metric)
        nearest_anchor_knn_per_cluster = pool.map(sample_to_anchor_knn_distances_func, np.arange(0, p, 1))
        for i in range(p):
            # Store distances between samples and their anchor neighbors
            sample_to_anchor_knn_distances[nearest_anchor_idx == i, :] = nearest_anchor_knn_per_cluster[i]

    # -----------------------
    # 5. Determine final k-nearest anchors for each sample
    # -----------------------
    sample_anchor_candidate_indices = anchor_knn_indices[nearest_anchor_idx, :]
    local_knn_indices, knn_distances = get_k_nearest_indices(sample_to_anchor_knn_distances, k)  #
    final_knn_indices = sample_anchor_candidate_indices[np.arange(N)[:, None], local_knn_indices]
    del local_knn_indices, sample_anchor_candidate_indices, sample_to_anchor_knn_distances
    gc.collect()

    # -----------------------
    # 6. Construct sample–anchor bipartite adjacency matrix
    # -----------------------
    if mode == 'generateAnchorsAndAdjacency':
        logg.info('Constructing sample–anchor bipartite graph...')

    if distance_metric == 'cosine':
        sample_anchor_affinity = 1 - knn_distances
    else:
        if kernel_type == 'localscaled':
            kernel_scale = np.mean(knn_distances,1)
            kernel_scale[kernel_scale == 0] = np.mean(knn_distances)
            sample_anchor_affinity = np.exp(-(knn_distances ** 2) / (2 * kernel_scale[:, None] ** 2))
        elif kernel_type == 'traditionalscaled':
            kernel_scale = np.mean(knn_distances)
            sample_anchor_affinity = np.exp(-(knn_distances ** 2) / (2 * kernel_scale ** 2))

        else:
            print(f"Invalid kernel_type '{kernel_type}'! Please use either 'localscaled' or 'traditionalscaled'.")

    # Avoid exact zeros
    sample_anchor_affinity[sample_anchor_affinity == 0] = 1e-16

    # Build sparse CSR matrix for sample–anchor adjacency
    indptr = np.arange(0, N * k + 1, k)
    sample_anchor_matrix = csr_matrix((sample_anchor_affinity.copy().ravel(),
                    final_knn_indices.copy().ravel(), indptr), shape=(N, p))

    if mode == 'generateAnchorsAndAdjacency':
        return Anchors, sample_anchor_matrix
    else:
        return Anchors, sample_anchor_matrix


def CreateCellGraph(data, edge_attr):
    """
        Construct an undirected cell graph using edge indices and edge weights.

        Args:
            data: PyTorch Geometric Data object containing node features and edge_index
            edge_attr (Tensor): Edge weights corresponding to edge_index

        Returns:
            G (networkx.Graph): Weighted undirected cell graph
        """
    G = nx.Graph()
    # Add all nodes at once (one node per cell)
    G.add_nodes_from(range(data.x.size(0)))
    # Convert edge index and weights to Python lists
    edges = data.edge_index.t().tolist()
    weights = edge_attr.tolist()
    # Add weighted edges, excluding self-loops
    G.add_weighted_edges_from([(edge[0], edge[1], weight) for edge, weight in zip(edges, weights) if edge[0] != edge[1]])
    return G


def Louvain(data, edge_attr, partition=None, weight='weight'):
    """
        Perform Louvain community detection on the cell graph.

        Args:
            data: PyTorch Geometric Data object containing node features and edge indices
            edge_attr (Tensor): Edge weights of the graph
            partition (dict, optional): Initial partition for Louvain algorithm
            weight (str): Edge attribute name used as weight

        Returns:
            num_communities (int): Number of detected communities
            labels (list): Community label for each node
            partition (dict): Mapping from node index to community label
            G (networkx.Graph): Constructed cell graph
        """
    # Construct the cell graph
    G = CreateCellGraph(data, edge_attr)

    # Apply the Louvain algorithm for community detection
    partition = community.best_partition(G, partition=partition, weight=weight)

    # Compute the number of detected communities
    num_communities = max(partition.values()) + 1
    print(f"The cell graph is divided into {num_communities} communities")

    # Extract clustering labels for all nodes
    labels = list(partition.values())

    return num_communities, labels, partition, G



def modularity_contribution(G, partition):
    """
        Compute the contribution of each node to its community based on weighted intra-community connections.

        This function calculates, for each node, the sum of edge weights connecting it to
        other nodes within the same community. This value reflects the node's contribution
        to the community structure and is related to modularity.

        Args:
            G (networkx.Graph): Weighted undirected graph
            partition (dict): Mapping from node index to community label

        Returns:
            contributions (dict): Dictionary mapping each node to its weighted intra-community degree
        """
    contributions = {}
    for node in G.nodes():
        # Sum of weights of edges connecting the node to neighbors in the same community
        k_i_in = sum(d['weight'] for neighbor, d in G[node].items() if partition[neighbor] == partition[node])
        contributions[node] = k_i_in
    return contributions