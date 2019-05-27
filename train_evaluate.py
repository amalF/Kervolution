from __future__ import division, print_function, absolute_import

import os
import functools
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
import tqdm
import datasets
from models import *
from kervolution import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import click

#args
@click.command()
##Data args
@click.option("-d","--datasetname", default="mnist", type=click.Choice(['cifar10', 'cifar100','mnist']))
@click.option("--n_classes", default=10)
##Training args
@click.option("--batch_size", default=64)
@click.option("--epochs", default=10)
@click.option("--lr", default=0.1)
@click.option("--keep_prob", default=0.5)
@click.option("--weight_decay", default=0.0) 
##logging args
@click.option("-o","--base_log_dir", default="logs")

def main(datasetname,
         n_classes,
         batch_size,
         epochs,
         lr,
         keep_prob,
         weight_decay,
         base_log_dir):

    #Fix TF random seed
    tf.random.set_seed(1777)
    log_dir = os.path.join(os.path.expanduser(base_log_dir),
                           "{}".format(datasetname))
    os.makedirs(log_dir, exist_ok=True)

    # dataset
    if datasetname=='mnist':
        train_data = datasets.MnistDataSet(shuffle=True,
                                           repeat=1)
        train_dataset = train_data.make_batch(batch_size)
        train_samples = train_data.num_samples
        test_dataset = datasets.MnistDataSet(subset="test",
                                             use_distortion=False,
                                             shuffle=False).make_batch(\
                                                     batch_size)

    #Network
    model = LeNet5(num_classes=n_classes,
                  convLayer=KernelConv2D).call((28,28,1))
    print(model.summary())
    #Train optimizer, loss
    boundries = [12000.0, 18000.0] #10,15 epochs
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
            cros_ent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits,labels=tf.cast(labels, dtype=tf.int32))
            loss = loss_fn(labels, logits) #tf.reduce_mean(cros_ent_loss)

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
                x = tf.expand_dims(x,3)
                tf.summary.image("input_image", x, step=optimizer.iterations)
                loss, logits = train_step(x,y)
                # Update training metric.
                train_acc_metric(y, logits)
                ckpt.step.assign_add(1)
                tf.summary.scalar("loss", loss, step=optimizer.iterations)
                #tf.summary.scalar("learning_rate", lr_schedule, step=optimizer.iterations)

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
        if ep%4 ==0: 
            with test_summary_writer.as_default():
                for x_batch, y_batch in test_dataset:
                    x_batch = tf.expand_dims(x_batch, 3)
                    test_logits = model(x_batch, training=False)
                    # Update test metrics
                    test_acc_metric(y_batch, test_logits)

                test_acc = test_acc_metric.result()
                tf.summary.scalar("accuracy", test_acc, step=epochs)
                test_acc_metric.reset_states()
                print('Test acc: %s' % (float(test_acc),))

if __name__=="__main__":
    main()
