import tensorflow as tf
from utils import quantize_int8_conv2d, quantize_int8_dense, keep_scale_conv2d, keep_scale_dense
from copy import deepcopy
import matplotlib.pyplot as plt
import clustering
import numpy as np
# Hyperparameter

learning_rate = 1e-5
batch_size = 32
epochs = 5000
eps = 0.00002

num_clusters = 75

epochs_list = []

test_loss_normal_list = []
test_loss_clustered_list = []
test_backdoor_loss_normal_list = []
test_backdoor_loss_clustered_list = []


test_acc_CDA_normal_list = []
test_acc_ASR_normal_list = []
test_acc_CDA_clustered_list = []
test_acc_ASR_clustered_list = []

clus_test_loss = tf.keras.metrics.Mean(name='clus_test_loss')
clus_test_backdoor_loss = tf.keras.metrics.Mean(name='clus_test_backdoor_loss')
clus_test_CDA = tf.keras.metrics.SparseCategoricalAccuracy(
    name='clus_test_CDA')
clus_test_ASR = tf.keras.metrics.SparseCategoricalAccuracy(
    name='clus_test_ASR')

# compile
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

Loss3 = tf.keras.metrics.Mean(name='loss3')
train_Acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_Acc')

full_test_loss = tf.keras.metrics.Mean(name='full_test_loss')
full_test_CDA = tf.keras.metrics.SparseCategoricalAccuracy(
    name='full_test_CDA')

full_test_backdoor_loss = tf.keras.metrics.Mean(name='full_test_backdoor_loss')
full_test_ASR = tf.keras.metrics.SparseCategoricalAccuracy(
    name='full_test_ASR')

quant_test_loss = tf.keras.metrics.Mean(name='quant_test_loss')
quant_test_CDA = tf.keras.metrics.SparseCategoricalAccuracy(
    name='quant_test_CDA')

quant_test_backdoor_loss = tf.keras.metrics.Mean(
    name='quant_test_backdoor_loss')
quant_test_ASR = tf.keras.metrics.SparseCategoricalAccuracy(
    name='quant_test_ASR')

MSE = tf.keras.losses.MeanSquaredError()


def assign_to_nearest_centroid(matrix, centroids):
    # Reshape the matrix to make it 2D for easier calculations
    matrix_flat = matrix.reshape(-1, matrix.shape[-1])

    # Calculate the Euclidean distance between each element and each centroid
    distances = np.linalg.norm(matrix_flat[:, None] - centroids[None], axis=2)

    # Find the index of the nearest centroid for each element
    nearest_centroid_indices = np.argmin(distances, axis=1)

    # Reshape the indices back to the original shape of the matrix
    cluster_indices = nearest_centroid_indices.reshape(matrix.shape[:-1])

    return cluster_indices


def all_layer_name(model):
    count = 0
    name_list = []
    for layer in model.layers:
        if layer.trainable_variables:
            for param in layer.trainable_variables:

                if len(param.shape) == 4 or len(param.shape) == 2:
                    name_list.append({count: layer.name})
                    count = count+1
    return name_list


def find_range(num_clusters, diff, index, centroid):
    if index == 0:
        return (centroid, centroid+diff-eps)

    elif index == num_clusters-1:
        return (centroid-diff+eps, centroid)

    else:
        return (centroid-diff+eps, centroid+diff-eps)


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


# @tf.function
def train_step(model, all_cluster_centroids, backdoor_clus_model, min_weights, max_weights, clustered_indices, x_train, y_train):
    print("train step called")
    with tf.GradientTape() as tape:

        predictions = model(x_train, training=False)
        loss3 = loss_object(y_train, predictions)

        loss_total = 1 * loss3

    gradients = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    Loss3.update_state(loss3)
    train_Acc.update_state(y_train, predictions)

    count = 0
    print("let's upadte")
    for layer in model.layers:
        if layer.trainable_variables:  # Conv2D AND DENSE
            for param in layer.trainable_variables:

                if len(param.shape) == 4:
                    print(layer.name)

                    new_param = param.numpy()
                    original_cluster_centroid = all_cluster_centroids[count]

                    # coordnates of min_weight in the weights matrix
                    mina, minb, minc, mind = min_weights[count][0]
                    maxa, maxb, maxc, maxd = max_weights[count][0]

                    new_param[mina, minb, minc, mind] = min_weights[count][1]
                    new_param[maxa, maxb, maxc, maxd] = max_weights[count][1]

                    # print(get_element(min_weights[count][0] ,new_param))
                    # input()

                    differences = np.diff(original_cluster_centroid)
                    diff = (differences[0])/2
                    for i in range(new_param.shape[0]):
                        for j in range(new_param.shape[1]):
                            for k in range(new_param.shape[2]):
                                for l in range(new_param.shape[3]):
                                    original_centroid_index = clustered_indices[count][i, j, k, l]
                                    current_centroid_index = np.argmin(
                                        np.abs(new_param[i, j, k, l] - original_cluster_centroid))

                                    if ((i, j, k, l) != (min_weights[count][0]) or (i, j, k, l) != max_weights[count][0]):
                                        if (new_param[i, j, k, l] < min_weights[count][1] and current_centroid_index == 0):
                                            new_param[i, j, k,
                                                      l] = min_weights[count][1]
                                            # input("i am here badly twice")
                                            continue
                                        if (new_param[i, j, k, l] > max_weights[count][1] and current_centroid_index == (num_clusters-1)):
                                            new_param[i, j, k,
                                                      l] = max_weights[count][1]
                                            # input("i am here badly")
                                            continue

                                        if (current_centroid_index != original_centroid_index):
                                            range_mini, range_maxy = find_range(
                                                num_clusters, diff, original_centroid_index, original_cluster_centroid[original_centroid_index])

                                            adjusted_value = tf.clip_by_value(
                                                new_param[i, j, k, l], range_mini, range_maxy)
                                            # print("adjusted_value is",adjusted_value.numpy())
                                            # print(original_cluster_centroid)
                                            # print(original_centroid_index)

                                            # print(current_centroid_index)
                                            # print(diff)
                                            # print(range_mini)
                                            # print(range_maxy)
                                            # print(new_param[i,j,k,l])
                                            new_param[i, j, k,
                                                      l] = adjusted_value.numpy()
                                            continue
                                        # else:
                                        #     # input("i am here luckily")
                                        #     print(original_cluster_centroid)
                                        #     print(original_centroid_index)

                                        #     print(current_centroid_index)
                                        #     print(diff)

                    param.assign(new_param)
                    count = count+1
                if len(param.shape) == 2:

                    new_param = param.numpy()
                    original_cluster_centroid = all_cluster_centroids[count]

                    # coordnates of min_weight in the weights matrix
                    mina, minb = min_weights[count][0]
                    maxa, maxb = max_weights[count][0]

                    new_param[mina, minb] = min_weights[count][1]
                    new_param[maxa, maxb] = max_weights[count][1]

                    differences = np.diff(original_cluster_centroid)
                    diff = (differences[0])/2
                    for i in range(new_param.shape[0]):
                        for j in range(new_param.shape[1]):
                            original_centroid_index = clustered_indices[count][i, j]
                            current_centroid_index = np.argmin(
                                np.abs(new_param[i, j] - original_cluster_centroid))
                            if ((i, j) != (min_weights[count][0]) or (i, j) != max_weights[count][0]):
                                if (new_param[i, j] < min_weights[count][1] and current_centroid_index == 0):
                                    new_param[i, j] = min_weights[count][1]
                                    continue
                                if (new_param[i, j] > max_weights[count][1] and current_centroid_index == (num_clusters-1)):
                                    new_param[i, j] = max_weights[count][1]
                                    continue

                                if (current_centroid_index != original_centroid_index):
                                    range_mini, range_maxy = find_range(
                                        num_clusters, diff, original_centroid_index, original_cluster_centroid[original_centroid_index])

                                    adjusted_value = tf.clip_by_value(
                                        new_param[i, j], range_mini, range_maxy)
                                    new_param[i, j] = adjusted_value.numpy()

                    param.assign(new_param)
                    count = count+1


@tf.function
def test_step(model, x_test, y_test, loss, accuracy):
    predictions = model(x_test, training=False)
    t_loss = loss_object(y_test, predictions)
    loss.update_state(t_loss)
    accuracy.update_state(y_test, predictions)


def train_model(backdoor_model, test_model, backdoor_clus_model,  dataset):
    save_path = "fine_results/"

    backdoor_clus_model, all_cluster_centroids, clustered_indices, min_weights, max_weights = clustering.get_clustered_model(
        num_clusters, backdoor_model, backdoor_clus_model, load_model=False, other_data=True)
    print("backdoor_clus_model is created")
    ds_train, ds_train_backdoor, ds_test, _, ds_test_backdoor_exclude_target = dataset.ds_data(
        batch_size, backdoor=False)

    best_acc = []
    for epoch in range(epochs):
        Loss3.reset_states()
        train_Acc.reset_states()
        full_test_loss.reset_states()
        full_test_CDA.reset_states()
        full_test_backdoor_loss.reset_states()
        full_test_ASR.reset_states()
        clus_test_loss.reset_states()
        clus_test_CDA.reset_states()
        clus_test_backdoor_loss.reset_states()
        clus_test_ASR.reset_states()

        for index, ((x_train, y_train), (x_train_backdoor, y_train_backdoor)) in enumerate(zip(ds_train, ds_train_backdoor)):
            if index > tf.math.ceil(dataset.train_samples / batch_size):
                break
            train_step(backdoor_model, all_cluster_centroids, backdoor_clus_model,   min_weights, max_weights, clustered_indices,
                       tf.concat([x_train, x_train_backdoor], axis=0), tf.concat([y_train, y_train_backdoor], axis=0))
            print("epoch is : ", epoch, " and index is : ", index)

            if index % 100 == 0:
                train_logs = '{} - Epoch: [{}][{}/{}]\t Loss3: {}\t  Acc: {}'
                print(train_logs.format('TRAIN', epoch + 1, index, tf.math.ceil(dataset.train_samples / batch_size),
                                        Loss3.result(), train_Acc.result(),
                                        ))
        test_model, all_cluster_centroids_test, clustered_indice_test, min_weights_test, max_weights_test = clustering.get_clustered_model(
            num_clusters, backdoor_model, backdoor_clus_model, load_model=False, other_data=False)
        for index, ((x_test, y_test), (x_test_backdoor, y_test_backdoor)) in enumerate(zip(ds_test, ds_test_backdoor_exclude_target)):
            test_step(backdoor_clus_model, x_test, y_test,
                      full_test_loss, full_test_CDA)
            test_step(backdoor_clus_model, x_test_backdoor,
                      y_test_backdoor, full_test_backdoor_loss, full_test_ASR)

        for index, ((x_test, y_test), (x_test_backdoor, y_test_backdoor)) in enumerate(zip(ds_test, ds_test_backdoor_exclude_target)):
            test_step(test_model, x_test, y_test,
                      clus_test_loss, clus_test_CDA)
            test_step(test_model, x_test_backdoor, y_test_backdoor,
                      clus_test_backdoor_loss, clus_test_ASR)

        full_template = 'Full: Epoch {}, Test_Loss: {}, Test_CDA: {}, Test_Backdoor_Loss: {}, Test_ASR: {}'
        print(full_template.format(epoch + 1,
                                   full_test_loss.result(),
                                   full_test_CDA.result(),
                                   full_test_backdoor_loss.result(),
                                   full_test_ASR.result()
                                   ))

        clus_template = 'clus: Epoch {}, Test_Loss: {}, Test_CDA: {}, Test_Backdoor_Loss: {}, Test_ASR: {}'
        print(clus_template.format(epoch + 1,
                                   clus_test_loss.result(),
                                   clus_test_CDA.result(),
                                   clus_test_backdoor_loss.result(),
                                   clus_test_ASR.result()
                                   ), end="\n\n")

        acc = [1 - full_test_ASR.result(), clus_test_ASR.result()]
        if sum(acc) > sum(best_acc):
            best_acc = acc
            model_path = save_path+str(epoch)+"_"+str(full_test_CDA.result())+"_"+str(
                full_test_ASR.result())+"_"+str(clus_test_CDA.result())+"_"+str(clus_test_ASR.result())+"/"
            tf.keras.models.save_model(backdoor_model, model_path)
            backdoor_model.save_weights(model_path + "ckpt/checkpoints")

        epochs_list.append(epoch)

        test_loss_normal_list.append(full_test_loss.result().numpy())
        test_backdoor_loss_normal_list.append(
            full_test_backdoor_loss.result().numpy())
        test_acc_ASR_normal_list.append(full_test_ASR.result().numpy())
        test_acc_CDA_normal_list.append(full_test_CDA.result().numpy())

        test_loss_clustered_list.append(clus_test_loss.result().numpy())
        test_backdoor_loss_clustered_list.append(
            clus_test_backdoor_loss.result().numpy())
        test_acc_ASR_clustered_list.append(clus_test_ASR.result().numpy())
        test_acc_CDA_clustered_list.append(clus_test_CDA.result().numpy())

        print(test_loss_normal_list)
        print(test_backdoor_loss_normal_list)
        print(test_acc_ASR_normal_list)
        print(test_acc_CDA_normal_list)

        print(test_loss_clustered_list)
        print(test_backdoor_loss_clustered_list)
        print(test_acc_ASR_clustered_list)
        print(test_acc_CDA_clustered_list)

        if (epoch+1) % 5 == 0:
            # Save subplot 2
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.plot(epochs_list[:epoch+1], test_loss_normal_list[:epoch+1],
                     label='train_loss_normal_values')
            ax2.plot(
                epochs_list[:epoch+1], test_loss_clustered_list[:epoch+1], label='test_loss_clustered')
            ax2.plot(epochs_list[:epoch+1], test_backdoor_loss_normal_list[:epoch+1],
                     label='test_backdoor_loss_normal')
            ax2.plot(epochs_list[:epoch+1], test_backdoor_loss_clustered_list[:epoch+1],
                     label='test_backdoor_loss_clustered')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Test Loss')
            ax2.legend()
            ax2.set_title('Test Losses vs Epochs')
            fig2.savefig(f'test_losses_vs_epochs_{epoch}.png')

            # Save subplot 4
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            ax4.plot(
                epochs_list[:epoch+1], test_acc_CDA_normal_list[:epoch+1], label='test_acc_CDA_normal')
            ax4.plot(
                epochs_list[:epoch+1], test_acc_ASR_normal_list[:epoch+1], label='test_acc_ASR_normal')
            ax4.plot(
                epochs_list[:epoch+1], test_acc_CDA_clustered_list[:epoch+1], label='test_acc_CDA_clustered')
            ax4.plot(
                epochs_list[:epoch+1], test_acc_ASR_clustered_list[:epoch+1], label='test_acc_ASR_clustered')
            ax4.set_xlabel('Epochs')
            ax4.set_ylabel('Test Accuracies')
            ax4.legend()
            ax4.set_title('Test Accuracies vs Epochs')
            fig4.savefig(f'test_accuracies_vs_epochs_{epoch}.png')

    # Plot 2: Test Losses
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(epochs_list, test_loss_normal_list,
             label='train_loss_normal_values')
    ax2.plot(epochs_list, test_loss_clustered_list,
             label='test_loss_clustered')
    ax2.plot(epochs_list, test_backdoor_loss_normal_list,
             label='test_backdoor_loss_normal')
    ax2.plot(epochs_list, test_backdoor_loss_clustered_list,
             label='test_backdoor_loss_clustered')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Test Loss')
    ax2.legend()
    ax2.set_title('Test Losses vs Epochs')
    fig2.savefig('test_losses_vs_epochs.png')

    # Plot 4: Test Accuracies
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(epochs_list, test_acc_CDA_normal_list,
             label='test_acc_CDA_normal')
    ax4.plot(epochs_list, test_acc_ASR_normal_list,
             label='test_acc_ASR_normal')
    ax4.plot(epochs_list, test_acc_CDA_clustered_list,
             label='test_acc_CDA_clustered')
    ax4.plot(epochs_list, test_acc_ASR_clustered_list,
             label='test_acc_ASR_clustered')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Test Accuracies')
    ax4.legend()
    ax4.set_title('Test Accuracies vs Epochs')
    fig4.savefig('test_accuracies_vs_epochs.png')

    # Adjust layout for better spacing
    plt.tight_layout()
