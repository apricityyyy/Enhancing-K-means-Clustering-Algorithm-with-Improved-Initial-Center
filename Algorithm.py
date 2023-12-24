import math

# custom input data
dataset = {
    'd1': (-2, 1),
    'd2': (0, -3),
    'd3': (2, 2),
    'd4': (1, -1),
    'd5': (3, 3),
    'd6': (-1, 2),
    'd7': (1, 1),
    'd8': (-2, -2),
    'd9': (2, -1),
    'd10': (-1, -3)
}

k = 3

# STEP 1, 2, 3
def adjust_dataset_for_negatives_global(dataset):
    """
    Adjust the dataset by finding the global minimum value across all attributes and subtracting it
    from each attribute of the data points.
    """
    # Find the global minimum value across all attributes
    global_min = min(attribute for point in dataset.values() for attribute in point)

    # Adjust the dataset by subtracting the global minimum value
    adjusted_dataset = {key: tuple(attribute - global_min for attribute in point) 
                        for key, point in dataset.items()}

    return adjusted_dataset

# Adjust the dataset if it contains negative values
adjusted_dataset = adjust_dataset_for_negatives_global(dataset)
print("STEP 1, 2, 3: Adjusted dataset", adjusted_dataset)
print("================================================")

# STEP 4
def calculate_from_origin(dataset):
    """
    After formatting the data, this function is needed to find
    Euclidian distance between the origin and data points.
    """
    distances = {}
    for point in dataset.values():
        operand = 0
        for attribute in point:
            operand += attribute * attribute
        distances[point] = math.sqrt(operand)
		
    return distances

distances_from_origin = calculate_from_origin(adjusted_dataset)
print("STEP 4: Calculate Distances from Origin:", distances_from_origin)
print("================================================")

# STEP 5
# Implement heap sort to sort the distances along with their corresponding data point keys
def heap_sort(distances):
    arr = list(distances.items())

    def heapify(n, i):
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left][1] > arr[largest][1]:
            largest = left
        if right < n and arr[right][1] > arr[largest][1]:
            largest = right
        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(n, largest)

    n = len(arr)
    for i in range(n // 2 - 1, -1, -1):
        heapify(n, i)
    for i in range(n - 1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]
        heapify(i, 0)

    return [key for key, _ in arr]

sorted_keys = heap_sort(distances_from_origin)
print("STEP 5: Sorted Points: ", sorted_keys)
print("================================================")

# STEP 6
def partition_into_k_sets(sorted_points, k):
    """
    Partition the sorted points into k equal (or as equal as possible) sets.
    """
    # Number of points
    n = len(sorted_points)
    
    # Size of each partition
    partition_size = n // k
    
    # Remainder to distribute among the partitions
    remainder = n % k
    
    # Partitions container
    partitions = []
    
    # Starting index for each partition
    start_idx = 0

    for i in range(k):
        # Determine the end index for the current partition
        end_idx = start_idx + partition_size + (1 if i < remainder else 0)
        # Create the partition and add it to the partitions list
        partitions.append(sorted_points[start_idx:end_idx])
        # Update the start index for the next partition
        start_idx = end_idx

    return partitions

# Example usage with the sorted points and a specified k value
k = 3 # Number of desired clusters
partitions = partition_into_k_sets(sorted_keys, k)
print("STEP 6: For Given k = 3, Clusters: ", partitions)
print("================================================")

# STEP 7
initial_centroids = []
for partition in partitions:
	initial_centroids.append(partition[len(partition) // 2])
	
print("STEP 7: Initial Centroids", initial_centroids)
print("================================================")

# STEP 8
def calculate_distances_to_centroids(dataset, centroids):
    """
    Calculate the Euclidean distance from each data point in the dataset to each of the centroids.
    """
    distances = {}

    for key, point in dataset.items():
        point_distances = []
        for centroid in centroids:
            distance = math.sqrt(sum((attr - cent) ** 2 for attr, cent in zip(point, centroid)))
            point_distances.append(distance)
        distances[key] = point_distances

    return distances

distances_to_centroids = calculate_distances_to_centroids(adjusted_dataset, initial_centroids)
print("STEP 8: Distance between Each Data Point to All the Initial Centroids:", distances_to_centroids)
print("================================================")

# STEP 9
# STEP 10, 11
NearestDist = []
ClusterId = []
i = 0
for distances in distances_to_centroids.values():
    NearestDist.insert(i, min(distances))
    ClusterId.insert(i, distances.index(min(distances)))
    i = i + 1

print("STEPS 10, 11, 12")
print("Nearest Distances:", NearestDist)
print("Cluster IDs:", ClusterId)

# STEP 13
def recalculate_centroids(cluster_id, dataset, k):
    # Initialize new_clusters with empty lists for each cluster
    new_clusters = [[] for _ in range(k)]

    # Populate new_clusters with data points based on their cluster ID
    for key, cluster_idx in zip(dataset.keys(), cluster_id):
        new_clusters[cluster_idx].append(dataset[key])

    # Calculate new centroids for each cluster
    new_centroids = []
    for cluster in new_clusters:
        centroid = tuple(sum(coord) / len(cluster) for coord in zip(*cluster)) if cluster else None
        new_centroids.append(centroid)

    return new_clusters, new_centroids

while True:
    new_clusters, new_centroids = recalculate_centroids(ClusterId, dataset, k)

    # Temporary lists to store updated cluster IDs and nearest distances
    updated_cluster_id = []
    updated_nearest_dist = []

    # Loop over each data point
    for key, point in dataset.items():
        # STEP 14.1
        current_nearest_centroid = new_centroids[ClusterId[list(dataset.keys()).index(key)]]
        current_distance = math.sqrt(sum((attr - cent) ** 2 for attr, cent in zip(point, current_nearest_centroid)))

        # STEP 14.2
        if current_distance <= NearestDist[list(dataset.keys()).index(key)]:
            # Keep the current cluster assignment
            updated_cluster_id.append(ClusterId[list(dataset.keys()).index(key)])
            updated_nearest_dist.append(current_distance)
        else:
            # STEP 14.2.1
            distances = [math.sqrt(sum((attr - cent) ** 2 for attr, cent in zip(point, centroid))) for centroid in new_centroids]
            closest_centroid_idx = distances.index(min(distances))
            updated_cluster_id.append(closest_centroid_idx)
            updated_nearest_dist.append(min(distances))

    # Check for convergence
    if updated_cluster_id == ClusterId:
        print("Convergence reached.")
        break
    else:
        # Update ClusterId and NearestDist for the next iteration
        ClusterId = updated_cluster_id
        NearestDist = updated_nearest_dist
        print("Further iterations required. Continuing...")