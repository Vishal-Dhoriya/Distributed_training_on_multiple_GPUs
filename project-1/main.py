import tensorflow as tf
from utils import (
    load_and_preprocess_data,
    create_distributed_datasets,
    create_model,
    create_loss_and_metrics
)
from trainer import train_model


def main():
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    train_data, test_data = load_and_preprocess_data()
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    
    train_dist_dataset, test_dist_dataset, global_batch_size = create_distributed_datasets(
        train_images, train_labels, test_images, test_labels, strategy
    )
    
    with strategy.scope():
        model = create_model()
        loss_object, test_loss, train_accuracy, test_accuracy, optimizer = create_loss_and_metrics()
    
    train_model(
        model=model,
        train_dist_dataset=train_dist_dataset,
        test_dist_dataset=test_dist_dataset,
        strategy=strategy,
        loss_object=loss_object,
        optimizer=optimizer,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        global_batch_size=global_batch_size,
        epochs=10
    )


if __name__ == "__main__":
    main()
