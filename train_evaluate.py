from __future__ import division, print_function, absolute_import

import os
import functools
import numpy as np
import tensorflow as tf
import tqdm
import datasets
from models import models_factory
from layers import *
import click

#args
@click.command()
##Data args
@click.option("-d","--datasetname", default="mnist", type=click.Choice(['cifar10','mnist']))
@click.option("--n_classes", default=10)
##Training args
@click.option('--model_name', default='lenetKNN')
@click.option('--kernel', default='polynomial')
@click.option('--pooling_method', default='max') 
@click.option('--cp', default=1.0)
@click.option('--dp', default=3.0)
@click.option('--gamma', default=1.0)
@click.option("--batch_size", default=50)
@click.option("--epochs", default=20)
@click.option("--lr", default=0.003)
@click.option("--keep_prob", default=1.0)
@click.option("--weight_decay", default=0.0) 
##logging args
@click.option("-o","--base_log_dir", default="logs")

def main(datasetname,n_classes,batch_size,
         model_name, kernel, cp, dp, gamma,pooling_method,
         epochs,lr,keep_prob,weight_decay,
         base_log_dir):

    #Fix TF random seed
    tf.random.set_seed(1777)
    log_dir = os.path.join(os.path.expanduser(base_log_dir),
                           "{}".format(datasetname))
    os.makedirs(log_dir, exist_ok=True)

    # dataset
    train_dataset, train_samples = datasets.get_dataset(datasetname, batch_size)
    test_dataset, _ = datasets.get_dataset(datasetname, batch_size, subset="test", shuffle=False)

    #Network
    kernel_fn = get_kernel(kernel, cp=cp, dp=dp, gamma=gamma)

    model = models_factory.get_model(model_name,
                      num_classes=n_classes,
                      keep_prob=keep_prob,
                      kernel_fn=kernel_fn,
                      pooling=pooling_method)

    #Train optimizer, loss
    nrof_steps_per_epoch = (train_samples//batch_size)
    boundries = [nrof_steps_per_epoch*75, nrof_steps_per_epoch*125]
    values = [lr, lr*0.1, lr*0.01]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(\
                    boundries,
                    values)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    #metrics
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    #Train step
    @tf.function
    def train_step(x,labels):
        with tf.GradientTape() as t:
            logits = model(x, training=True)
            loss = loss_fn(labels, logits)

        gradients = t.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, logits

    #Run

    ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

    #Summary writers
    train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                      'summaries',
                                                                      'train'))
    test_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir,
                                                                     'summaries',
                                                                     'test'))


    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    ckpt_path = os.path.join(log_dir, 'checkpoints')
    manager = tf.train.CheckpointManager(ckpt,ckpt_path, max_to_keep=3)
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


    for ep in tqdm.trange(epochs, desc='Epoch Loop'):
        if ep < ep_cnt:
            continue

            # update epoch counter
        ep_cnt.assign_add(1)
        with train_summary_writer.as_default():
            # train for an epoch
            for step, (x,y) in enumerate(train_dataset):
                if len(x.shape)==3:
                    x = tf.expand_dims(x,3)
                tf.summary.image("input_image", x, step=optimizer.iterations)
                loss, logits = train_step(x,y)
                train_acc_metric(y, logits)
                ckpt.step.assign_add(1)
                tf.summary.scalar("loss", loss, step=optimizer.iterations)
                
                if int(ckpt.step) % 1000 == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step),
                                                                    save_path))
                # Log every 200 batch
                if step % 200 == 0:
                    train_acc = train_acc_metric.result() 
                    print("Training loss {:1.2f}, accuracu {} at step {}".format(\
                            loss.numpy(),
                            float(train_acc),
                            step))


            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            tf.summary.scalar("accuracy", train_acc, step=ep)
            print('Training acc over epoch: %s' % (float(train_acc),))
            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()
 

    ############################## Test the model #############################
        with test_summary_writer.as_default():
            for x_batch, y_batch in test_dataset:
                if len(x_batch.shape)==3:
                    x_batch = tf.expand_dims(x_batch, 3)
                test_logits = model(x_batch, training=False)
                # Update test metrics
                test_acc_metric(y_batch, test_logits)

            test_acc = test_acc_metric.result()
            tf.summary.scalar("accuracy", test_acc, step=ep)
            test_acc_metric.reset_states()
            print('[Epoch {}] Test acc: {}'.format(ep, float(test_acc)))

if __name__=="__main__":
    main()
