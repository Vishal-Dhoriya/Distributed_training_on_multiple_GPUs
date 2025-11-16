import tensorflow as tf
import tensorflow_hub as hub
from config import IMAGE_SIZE, MODULE_HANDLE


def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return image, label


def set_global_batch_size(batch_size_per_replica, strategy):
    global_batch_size = batch_size_per_replica * strategy.num_replicas_in_sync
    return global_batch_size


def create_datasets(train_examples, validation_examples, test_examples, 
                    batch_size_per_replica, buffer_size, strategy):
    train_batches = train_examples.shuffle(buffer_size // 4).map(format_image).batch(batch_size_per_replica).prefetch(1)
    validation_batches = validation_examples.map(format_image).batch(batch_size_per_replica).prefetch(1)
    test_batches = test_examples.map(format_image).batch(1)
    
    train_dist_dataset = strategy.experimental_distribute_dataset(train_batches)
    val_dist_dataset = strategy.experimental_distribute_dataset(validation_batches)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_batches)
    
    return train_dist_dataset, val_dist_dataset, test_dist_dataset


class ResNetModel(tf.keras.Model):
    def __init__(self, classes):
        super(ResNetModel, self).__init__()
        self._feature_extractor = hub.KerasLayer(MODULE_HANDLE, trainable=False)
        self._classifier = tf.keras.layers.Dense(classes, activation='softmax')

    def call(self, inputs):
        x = self._feature_extractor(inputs)
        x = self._classifier(x)
        return x


def create_loss_and_metrics(strategy, global_batch_size):
    with strategy.scope():
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE
        )
        
        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
        
        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        
        return loss_object, compute_loss, test_loss, train_accuracy, test_accuracy


def create_model_and_optimizer(strategy, num_classes):
    with strategy.scope():
        model = ResNetModel(classes=num_classes)
        optimizer = tf.keras.optimizers.Adam()
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        
    return model, optimizer, checkpoint
