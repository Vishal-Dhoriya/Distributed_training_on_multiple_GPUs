import tensorflow as tf
from tqdm import tqdm


def train_test_step_fns(strategy, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy):
    with strategy.scope():
        def train_step(inputs):
            images, labels = inputs
            with tf.GradientTape() as tape:
                predictions = model(images, training=True)
                loss = compute_loss(labels, predictions)
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_accuracy.update_state(labels, predictions)
            return loss

        def test_step(inputs):
            images, labels = inputs
            predictions = model(images, training=False)
            t_loss = loss_object(labels, predictions)
            
            test_loss.update_state(t_loss)
            test_accuracy.update_state(labels, predictions)
        
        return train_step, test_step


def distributed_train_test_step_fns(strategy, train_step, test_step):
    with strategy.scope():
        @tf.function
        def distributed_train_step(dataset_inputs):
            per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        @tf.function
        def distributed_test_step(dataset_inputs):
            return strategy.run(test_step, args=(dataset_inputs,))
    
        return distributed_train_step, distributed_test_step


def train_model(strategy, model, train_dist_dataset, val_dist_dataset, 
                distributed_train_step, distributed_test_step, 
                train_accuracy, test_loss, test_accuracy, epochs):
    
    with strategy.scope():
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for x in tqdm(train_dist_dataset):
                total_loss += distributed_train_step(x)
                num_batches += 1
            
            train_loss = total_loss / num_batches
            
            for x in val_dist_dataset:
                distributed_test_step(x)
            
            template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
            print(template.format(
                epoch + 1,
                train_loss,
                train_accuracy.result() * 100,
                test_loss.result(),
                test_accuracy.result() * 100
            ))
            
            test_loss.reset_state()
            train_accuracy.reset_state()
            test_accuracy.reset_state()
