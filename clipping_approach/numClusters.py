import clusteringNumClusters
import tensorflow as tf
import train_backdoor
import matplotlib.pyplot as plt 

batch_size=128


@tf.function
def test_step(model, x_test, y_test, accuracy):
    predictions = model(x_test, training=False)
    accuracy.update_state(y_test, predictions)
    # print("accuracy= ",accuracy.result())

list_num_clusters=list(range(45,250))
def _get_number_of_unique_weights_of_clustered(stripped_model, layer_nr, weight_name):
    layer = stripped_model.layers[layer_nr]
    if hasattr(layer, 'trainable_weights') and any(weight_name in w.name for w in layer.trainable_weights):
        weight = getattr(layer, weight_name)
        weights_as_list = weight.numpy().reshape(-1,).tolist()
        print()
        print("minmum value in clustered is : ", min(weights_as_list))
        print("maximum value in clustered is : ", max(weights_as_list))
        nr_of_unique_weights = len(set(weights_as_list))
        print()
        return nr_of_unique_weights
    else:
        return 0  
    
    
def optimalClusters(model, dataset):
    ds_train, ds_train_backdoor, ds_test, ds_test_backdoor, _ = dataset.ds_data(batch_size, backdoor=True)
    CDA_list=[]
    ASR_list=[]

    for num_clusters in list_num_clusters:
        final_clustered_model=clusteringNumClusters.get_clustered_model(num_clusters,model)
        
        for layer_nr, layer in enumerate(final_clustered_model.layers):
            weight_name = 'kernel'
            print()
            unique_weights_count = _get_number_of_unique_weights_of_clustered(final_clustered_model, layer_nr, weight_name)
            print(f"Layer {layer_nr}: Number of Unique {weight_name} Weights = {unique_weights_count}")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiii")   
        for layer_nr, layer in enumerate(model.layers):
            weight_name = 'kernel'
            print()
            unique_weights_count = _get_number_of_unique_weights_of_clustered(final_clustered_model, layer_nr, weight_name)
            print(f"Layer {layer_nr}: Number of Unique {weight_name} Weights = {unique_weights_count}")

        
        for index, (x_test, y_test) in enumerate(ds_test):
            test_step(final_clustered_model, x_test, y_test, train_backdoor.test_CDA)

        for index, (x_test, y_test) in enumerate(ds_test_backdoor):
            test_step(final_clustered_model, x_test, y_test,train_backdoor.test_ASR)
            
        full_template = 'Number of Clusters: {} ,test_CDA_clustered: {}, test_ASR_clustered: {}'
        print(full_template.format(num_clusters,
                                train_backdoor.test_CDA.result(),
                                train_backdoor.test_ASR.result(),
                                ), end="\n\n")
        
        CDA_list.append(train_backdoor.test_CDA.result().numpy())
        ASR_list.append(train_backdoor.test_ASR.result().numpy())
        
        
        
        

    plt.plot(list_num_clusters, CDA_list, label='test_CDA', color='blue')
    plt.plot(list_num_clusters, ASR_list, label='test_ASR',color='red')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs Number of Clusters')
    plt.savefig('acc_vs_num_of_clusters.png')
    