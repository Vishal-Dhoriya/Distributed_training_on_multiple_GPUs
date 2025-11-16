import tensorflow as tf


def train_step(inputs, model, loss_object, optimizer, train_accuracy, global_batch_size):
    images, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        per_example_loss = loss_object(labels, predictions)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_accuracy.update_state(labels, predictions)
    return loss


def test_step(inputs, model, loss_object, test_loss, test_accuracy):
    images, labels = inputs
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    
    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)


@tf.function
def distributed_train_step(dataset_inputs, strategy, model, loss_object, optimizer, train_accuracy, global_batch_size):
    per_replica_losses = strategy.run(
        train_step, 
        args=(dataset_inputs, model, loss_object, optimizer, train_accuracy, global_batch_size)
    )
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def distributed_test_step(dataset_inputs, strategy, model, loss_object, test_loss, test_accuracy):
    strategy.run(test_step, args=(dataset_inputs, model, loss_object, test_loss, test_accuracy))


def train_model(model, train_dist_dataset, test_dist_dataset, strategy, loss_object, 
                optimizer, train_accuracy, test_accuracy, test_loss, global_batch_size, epochs=10):
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_dist_dataset:
            total_loss += distributed_train_step(
                batch, strategy, model, loss_object, optimizer, train_accuracy, global_batch_size
            )
            num_batches += 1
        
        train_loss = total_loss / num_batches
        
        for batch in test_dist_dataset:
            distributed_test_step(batch, strategy, model, loss_object, test_loss, test_accuracy)
        
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
