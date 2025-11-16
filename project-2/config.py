import tensorflow as tf
import tensorflow_datasets as tfds
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# tfds.disable_progress_bar()

BUFFER_SIZE = None
EPOCHS = 10
BATCH_SIZE_PER_REPLICA = 64
PIXELS = 224
IMAGE_SIZE = (PIXELS, PIXELS)

MODULE_HANDLE = 'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1'

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


def load_dataset():
    splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']
    (train_examples, validation_examples, test_examples), info = tfds.load(
        'oxford_flowers102',
        with_info=True,
        as_supervised=True,
        split=splits,
        data_dir='data/'
    )
    
    num_examples = info.splits['train'].num_examples
    num_classes = info.features['label'].num_classes
    
    return train_examples, validation_examples, test_examples, num_examples, num_classes


def get_strategy():
    strategy = tf.distribute.MirroredStrategy()
    return strategy
