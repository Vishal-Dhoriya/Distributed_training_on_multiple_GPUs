from config import (
    load_dataset, get_strategy, EPOCHS, BATCH_SIZE_PER_REPLICA
)
from model_utils import (
    set_global_batch_size, create_datasets, create_loss_and_metrics,
    create_model_and_optimizer
)
from trainer import (
    train_test_step_fns, distributed_train_test_step_fns, train_model
)


def main():
    strategy = get_strategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
    train_examples, validation_examples, test_examples, num_examples, num_classes = load_dataset()
    
    global_batch_size = set_global_batch_size(BATCH_SIZE_PER_REPLICA, strategy)
    print('Global batch size: {}'.format(global_batch_size))
    
    train_dist_dataset, val_dist_dataset, test_dist_dataset = create_datasets(
        train_examples, validation_examples, test_examples,
        BATCH_SIZE_PER_REPLICA, num_examples, strategy
    )
    
    loss_object, compute_loss, test_loss, train_accuracy, test_accuracy = create_loss_and_metrics(
        strategy, global_batch_size
    )
    
    model, optimizer, checkpoint = create_model_and_optimizer(strategy, num_classes)
    
    train_step, test_step = train_test_step_fns(
        strategy, model, compute_loss, optimizer, train_accuracy, loss_object, test_loss, test_accuracy
    )
    
    distributed_train_step, distributed_test_step = distributed_train_test_step_fns(
        strategy, train_step, test_step
    )
    
    train_model(
        strategy, model, train_dist_dataset, val_dist_dataset,
        distributed_train_step, distributed_test_step,
        train_accuracy, test_loss, test_accuracy, EPOCHS
    )


if __name__ == "__main__":
    main()
