import model_vgg16
import datasets
import train_backdoor
import fine_tuning
import sys
import numClusters
import evaluate


def main(argv):

    dataset = datasets.MNIST(target=0)
    model = model_vgg16.VGG16(classes=dataset.classes)
    model.build(input_shape=(None, 32, 32, 3))

    step = int(argv)
    if step == 0:  # Train the backdoor model
        train_backdoor.train_model(model, dataset, save_path="results/")
    elif step == 1:  # Find optimal number of clusters
        numClusters.optimalClusters(model, dataset)
    elif step == 2:  # Fine-tune the backdoor model
        load_path = "results/114_tf.Tensor(0.9934, shape=(), dtype=float32)tf.Tensor(1.0, shape=(), dtype=float32)/"

        backdoor_model = model_vgg16.VGG16(classes=dataset.classes)
        backdoor_model.build(input_shape=(None, 32, 32, 3))
        backdoor_model.load_weights(load_path + "ckpt/checkpoints")

        backdoor_clus_model = model_vgg16.VGG16(classes=dataset.classes)
        backdoor_clus_model.build(input_shape=(None, 32, 32, 3))

        test_model = model_vgg16.VGG16(classes=dataset.classes)
        test_model.build(input_shape=(None, 32, 32, 3))

        fine_tuning.train_model(
            backdoor_model, test_model, backdoor_clus_model, dataset)
    elif step == 3:  # Evaluation model
        save_path = "results/114_tf.Tensor(0.9934, shape=(), dtype=float32)tf.Tensor(1.0, shape=(), dtype=float32)/"
        model.load_weights(save_path + "ckpt/checkpoints")
        evaluate.evaluate_model(model, dataset)

    else:
        print("Invalid argument...")


if __name__ == '__main__':
    main(sys.argv[1])
