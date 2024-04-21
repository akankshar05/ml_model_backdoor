import tensorflow as tf
import numpy as np

class CustomClusteringAlgorithm:
    def __init__(self, num_clusters):
        self.cluster_centroids=None
        self.all_cluster_centroids={}
        self.num_clusters=num_clusters
        self.clustered_indices = {} 
        self.min_weights = {}  # Dictionary to store min weights for each layer
        self.max_weights = {}  # Dictionary to store max weights for each layer
        
        
    def calculate_cluster_centroids(self, count , name,original_weights):
        flattened_weights = original_weights.flatten()
        mini_centroid= np.min(flattened_weights)
        maxi_centroid=np.max(flattened_weights)
        self.cluster_centroids = np.linspace(mini_centroid, maxi_centroid ,self.num_clusters, dtype=np.float32)
        self.all_cluster_centroids[count]=self.cluster_centroids
        
        self.min_weights[count] = (tuple(tf.where(tf.equal(original_weights, mini_centroid)).numpy()[0]), mini_centroid)
        self.max_weights[count] = (tuple(tf.where(tf.equal(original_weights, maxi_centroid)).numpy()[0]), maxi_centroid)

    def get_clustered_weight(self,original_weight):
        clustered_weight = np.zeros_like(original_weight)
        clustered_index = np.zeros_like(original_weight, dtype=np.int32)  # Store clustered indices

        
        if len(original_weight.shape) == 4:  # Conv2D layer
            for i in range(original_weight.shape[0]):
                for j in range(original_weight.shape[1]):
                    for k in range(original_weight.shape[2]):
                        for l in range(original_weight.shape[3]):
                            closest_centroid = np.argmin(np.abs(original_weight[i, j, k, l] - self.cluster_centroids))
                            clustered_weight[i, j, k, l] = self.cluster_centroids[closest_centroid]
                            clustered_index[i, j, k, l] = closest_centroid


        elif len(original_weight.shape) == 2:  # Dense layer
            for i in range(original_weight.shape[0]):
                for j in range(original_weight.shape[1]):
                    closest_centroid = np.argmin(np.abs(original_weight[i, j] - self.cluster_centroids))
                    clustered_weight[i, j] = self.cluster_centroids[closest_centroid]
                    clustered_index[i, j] = closest_centroid


        return clustered_weight, clustered_index
    
    def add_clustered_indices(self, layer_name, clustered_index):
        if layer_name not in self.clustered_indices:
            self.clustered_indices[layer_name] = {}
        self.clustered_indices[layer_name] = clustered_index

def get_element(index, wt):
    
    if len(index) == 2:
        result = wt[index[0], index[1]]
    elif len(index) == 4:
        result = wt[index[0], index[1], index[2], index[3]]
    if isinstance(wt, np.ndarray):
        return result
    else:
        return result.numpy()

def get_index(value, wt):
    return tuple(tf.where(tf.equal(wt, value)).numpy()[0])


def get_clustered_model(num_clusters, model, train_clus_model, load_model, other_data):
    
    print("clustering the model")
    
    save_path = "results/324_tf.Tensor(1.9001, shape=(), dtype=float32)/"
    if load_model:
        train_clus_model.load_weights(save_path + "ckpt/checkpoints")
        print("Cluster Number: {}".format(num_clusters))
    else:
        train_clus_model.set_weights(model.get_weights())
        

    clustering_algorithm_obj = CustomClusteringAlgorithm(num_clusters)
    count=0
    for layer in train_clus_model.layers:
        if layer.trainable_variables:  # Conv2D AND DENSE
            for param in layer.trainable_variables:
                if len(param.shape) == 2 or len(param.shape) == 4: #ignoring biases 
                    
                    
                    # print("let's start printing")
                    # print(param[0,0,0,2].numpy())
                    # print(type(param))
                    weights= param.numpy()
                    # print(weights[0,0,0,2])
                    # print(type(weights))
                    # index = tuple(tf.where(tf.equal(param, -0.14260618)).numpy()[0])
                    # print(index)
                    # index =tuple(tf.where(tf.equal(weights, -0.14260618)).numpy()[0])
                    # print(index)
                    # print(param.name)
                    # input()
                    
                    
                    clustering_algorithm_obj.calculate_cluster_centroids(count, layer.name, param.numpy())
                    
                    
                    # print(clustering_algorithm_obj.min_weights)
                    # print(get_element(clustering_algorithm_obj.min_weights[count][0], param))
                    # print(get_element(clustering_algorithm_obj.min_weights[count][0], weights))
                    # print(clustering_algorithm_obj.max_weights)
                    # print(get_element(clustering_algorithm_obj.max_weights[count][0], param))
                    # print(get_element(clustering_algorithm_obj.max_weights[count][0], weights))
                    # print("-------------------------------------------")
        
                    clustered_weight, clustered_index = clustering_algorithm_obj.get_clustered_weight(param.numpy())
                    
                    # print("magic")
                    # print(param.shape)
                    # print(type(param))
                    # print(clustered_weight.shape)
                    # print(type(clustered_weight))
                    # print(clustered_index.shape)
                    # print(type(clustered_index))
                    # print(clustered_weight)
                    # print("-------------------------------------------")
                    # print(clustered_index)
                    # # input()
                    # print("-------------------------------------------")
                    # print(type(param))
                    # print(type(clustered_weight))
                    param.assign(clustered_weight)
                    # print("finallllyyyyyyyyyyyyyyyyyyyyyyy")
                    if other_data:
                        
                        clustering_algorithm_obj.add_clustered_indices(count, clustered_index)
                        # print(clustering_algorithm_obj.cluster_centroids)
                        # input()
                    
                        
                        # del clustered_weight, clustered_index
                    count=count+1
                    # print("enddddddddddddddddddddddddddddddddddddd")
                    
                    # del clustered_weight
                    
    # del clustering_algorithm_obj
    return train_clus_model,clustering_algorithm_obj.all_cluster_centroids, clustering_algorithm_obj.clustered_indices, clustering_algorithm_obj.min_weights,clustering_algorithm_obj.max_weights