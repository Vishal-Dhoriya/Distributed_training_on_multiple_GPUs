import tensorflow as tf
import numpy as np
import os

# os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_and_preprocess_data():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    train_images = train_images[..., None]
    test_images = test_images[..., None]
    
    train_images = train_images / np.float32(255)
    test_images = test_images / np.float32(255)
    
    return (train_images, train_labels), (test_images, test_labels)


def create_distributed_datasets(train_images, train_labels, test_images, test_labels, strategy):
    BUFFER_SIZE = len(train_images)
    BATCH_SIZE_PER_REPLICA = 64
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)
    ).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_images, test_labels)
    ).batch(GLOBAL_BATCH_SIZE)
    
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)
    
    return train_dist_dataset, test_dist_dataset, GLOBAL_BATCH_SIZE


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model


def create_loss_and_metrics():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, 
        reduction=tf.keras.losses.Reduction.NONE
    )
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    optimizer = tf.keras.optimizers.Adam()
    
    return loss_object, test_loss, train_accuracy, test_accuracy, optimizer


def compute_loss(loss_object, labels, predictions, global_batch_size):
    per_example_loss = loss_object(labels, predictions)
    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
