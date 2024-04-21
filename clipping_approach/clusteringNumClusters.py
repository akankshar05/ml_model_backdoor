import tensorflow as tf
import numpy as np

class CustomClusteringAlgorithm:
    def __init__(self, num_clusters):
        self.cluster_centroids=None
        self.num_clusters=num_clusters
        
        
    def calculate_cluster_centroids(self, original_weights):
        flattened_weights = original_weights.flatten()
        self.cluster_centroids = np.linspace(np.min(flattened_weights), np.max(flattened_weights), self.num_clusters, dtype=np.float32)

    def get_clustered_weight(self,original_weight):
        # print("weight shape: ", len(original_weight.shape))
        clustered_weight = np.zeros_like(original_weight)
        
        if len(original_weight.shape) == 4:  # Conv2D layer
            for i in range(original_weight.shape[0]):
                for j in range(original_weight.shape[1]):
                    for k in range(original_weight.shape[2]):
                        for l in range(original_weight.shape[3]):
                            closest_centroid = np.argmin(np.abs(original_weight[i, j, k, l] - self.cluster_centroids))
                            clustered_weight[i, j, k, l] = self.cluster_centroids[closest_centroid]

        elif len(original_weight.shape) == 2:  # Dense layer
            for i in range(original_weight.shape[0]):
                for j in range(original_weight.shape[1]):
                    closest_centroid = np.argmin(np.abs(original_weight[i, j] - self.cluster_centroids))
                    clustered_weight[i, j] = self.cluster_centroids[closest_centroid]

        return clustered_weight


def get_clustered_model(num_clusters, train_clus_model):
    
    print("clustering the model")
    
    save_path = "results/114_tf.Tensor(0.9934, shape=(), dtype=float32)tf.Tensor(1.0, shape=(), dtype=float32)/"
    train_clus_model.load_weights(save_path + "ckpt/checkpoints")
    print("Cluster Number: {}".format(num_clusters))
    # else:
    #     train_clus_model.set_weights(model.get_weights())
        

    clustering_algorithm_obj = CustomClusteringAlgorithm(num_clusters)
    for layer in train_clus_model.layers:
        if layer.trainable_variables:  # Conv2D AND DENSE
            for param in layer.trainable_variables:
                if len(param.shape) == 2 or len(param.shape) == 4:
                    clustering_algorithm_obj.calculate_cluster_centroids(param.numpy())
                    clustered_weight = clustering_algorithm_obj.get_clustered_weight(param.numpy())
                    param.assign(clustered_weight)
                    del clustered_weight
                    
    del clustering_algorithm_obj
    return train_clus_model