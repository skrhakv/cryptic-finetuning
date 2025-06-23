import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from biotite.structure import sasa, AtomArray

EPSILON = 5  # Max distance for neighbors (adjust as needed)
MIN_SAMPLES = 3  # Minimum points to form a cluster (adjust as needed)
SASA_THRESHOLD = 0.5  # SASA threshold for filtering points (adjust as needed)
THRESHOLD = 0.7

def compute_clusters(points: AtomArray, prediction_scores: np.array, check_sasa=False):
    # This function computes clusters for the given points and prediction scores
    points_array = points.coord
    scores_array = prediction_scores

    assert len(points_array) == len(scores_array), f"Length of points and scores do not match: {len(points_array)} vs {len(scores_array)}"
    high_score_mask = scores_array > THRESHOLD
    
    if check_sasa:
        sasa_values = sasa(points)
        sasa_mask = sasa_values > SASA_THRESHOLD
        high_score_mask = high_score_mask & sasa_mask
        
    high_score_points = points_array[high_score_mask]

    dbscan = DBSCAN(eps=EPSILON, min_samples=MIN_SAMPLES)
    # dbscan = AgglomerativeClustering(distance_threshold=EPSILON, n_clusters=None, linkage='single')
    labels = dbscan.fit_predict(high_score_points)

    # Initialize all labels to -1
    all_labels = -1 * np.ones(len(points), dtype=int)
    # Assign cluster labels to high score points
    all_labels[high_score_mask] = labels
    labels = all_labels

    return labels
