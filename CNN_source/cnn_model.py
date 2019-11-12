#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import math, time
import tensorflow as tf
import tempfile
import flags as f
import input, lenet5, evaluate

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(1)
np.random.seed(10)

summaries = {'train': [], 'validate': [], 'test': []}



def main(unused_argv):
      train_dataset, validate_dataset, test_dataset = input.input(shuffle_files=False)
      #Text information
      info = tf.constant(
            ["Batch size = %s" % f.FLAGS.batch_size,
             "Epochs = %s" % f.FLAGS.num_epochs,
             "Learning rate = %s" % f.FLAGS.learning_rate,
             "Batch normalization = No",
             "Window size = %s" % f.FLAGS.window_size,
             "Shuffle Files = No",
             "CNN model = %s" % f.FLAGS.cnn_model,
             "Shuffle Samples = YES"]
      )
      with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, input.SAMPLE_DEPTH, input.SAMPLE_HEIGHT, input.SAMPLE_WIDTH])
            y_ = tf.placeholder(tf.float32, [None, 2])
            dropout_rate = tf.placeholder(tf.float32)
            is_training = tf.placeholder(tf.bool)

      with tf.name_scope('logits'):
            if f.FLAGS.cnn_model == "lenet5":
                  logits = lenet5.model_fn(sample_input = x, is_training=is_training, summaries=summaries)
                  
      with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=logits)
            mean_cross_entropy_loss = tf.reduce_mean(cross_entropy)
            
            loss_summ = tf.summary.scalar('Mean_cross_entropy_loss', mean_cross_entropy_loss)
            summaries['train'].append(loss_summ)
            #summaries['validate'].append(loss_summ)

      with tf.name_scope('adam_optimizer'):
            optimizer = tf.train.AdamOptimizer(f.FLAGS.learning_rate).minimize(mean_cross_entropy_loss)

      with tf.name_scope('accuracy'):
            preds = tf.argmax(logits, 1)
            correct_preds = tf.argmax(y_, 1)
            equal = tf.equal(preds, correct_preds)
            training_accuracy_op = tf.reduce_mean(tf.cast(equal, tf.float32))
            summaries['train'].append(tf.summary.scalar('Training_Accuracy', training_accuracy_op))

      
      with tf.name_scope('Evaluation_Metrics'):
            tp_op = evaluate.tp(logits=logits, labels=y_)
            fp_op = evaluate.fp(logits=logits, labels=y_)
            tn_op = evaluate.tn(logits=logits, labels=y_)
            fn_op = evaluate.fn(logits=logits, labels=y_)
            
            tp_sum = tf.placeholder(tf.float32)
            tn_sum = tf.placeholder(tf.float32)
            fp_sum = tf.placeholder(tf.float32)
            fn_sum = tf.placeholder(tf.float32)
            
            precision_op = evaluate.precision(tp=tp_sum, fp=fp_sum, tn=tn_sum, fn=fn_sum)
            accuracy_op = evaluate.accuracy(tp=tp_sum, fp=fp_sum, tn=tn_sum, fn=fn_sum)
            recall_op = evaluate.recall(tp=tp_sum, fp=fp_sum, tn=tn_sum, fn=fn_sum)
            fscore_op = evaluate.fscore(tp=tp_sum, fp=fp_sum, tn=tn_sum, fn=fn_sum)

            precision_summ = tf.summary.scalar('Precision', precision_op)
            accuracy_summ = tf.summary.scalar('Accuracy', accuracy_op)
            recall_summ = tf.summary.scalar('Recall', recall_op)
            fscore_summ = tf.summary.scalar('Fscore', fscore_op)
            
            summaries['validate'].append(accuracy_summ)
            summaries['validate'].append(precision_summ)
            summaries['validate'].append(recall_summ)
            summaries['validate'].append(fscore_summ)

            summaries['test'].append(accuracy_summ)
            summaries['test'].append(precision_summ)
            summaries['test'].append(recall_summ)
            summaries['test'].append(fscore_summ)

      print ("Saving graph to %s" % f.FLAGS.log_dir)
      train_writer = tf.summary.FileWriter(f.FLAGS.log_dir + "/train")
      validate_writer = tf.summary.FileWriter(f.FLAGS.log_dir + "/validate")
      test_writer = tf.summary.FileWriter(f.FLAGS.log_dir + "/test")
      train_writer.add_graph(tf.get_default_graph())
      
      train_summaries = tf.summary.merge(summaries['train'])
      validate_summaries = tf.summary.merge(summaries['validate'])
      test_summaries = tf.summary.merge(summaries['test'])

      with tf.Session() as sess:
            train_writer.add_summary(sess.run(tf.summary.text("Information", info)))
            train_iter = train_dataset.make_initializable_iterator()
            train_next_elem = train_iter.get_next()
            sess.run(tf.global_variables_initializer())
            global_step = 0
            display_freq = 10
            validate_freq = 50
            test_freq = 50
            for epoch in range(1, f.FLAGS.num_epochs+1):
                  sess.run(train_iter.initializer)
                  step_time = 0.0
                  fetch_time = 0.0
                  while True:
                        try:
                              a = time.time()
                              global_step += 1
                              sample, label = sess.run(train_next_elem)
                              fetch_time += time.time() - a
                              #print (sample.shape, label.shape)
                              #print (label)
                              #for s in sample[0][0]:
                              #      print (s)
                              a = time.time()
                              _, summ = sess.run([optimizer, train_summaries], feed_dict={x: sample, y_: label, dropout_rate: 0.5, is_training: True})
                              train_writer.add_summary(summ, global_step)
                              step_time += time.time() - a
                        except tf.errors.OutOfRangeError:
                              break

                        if global_step % display_freq == 0:
                              batch_loss, batch_accuracy = sess.run([mean_cross_entropy_loss, training_accuracy_op],
                                                                    feed_dict={x: sample, y_: label, dropout_rate: 1.0, is_training: False})
                              print ("Epoch {:3}\t Step {:5}:\t Loss={:.3f}, \tTraining Accuracy={:.5f} \tStep Time {:4.2f}m, Fetch Time {:4.2f}m".
                                     format(epoch, global_step, batch_loss, batch_accuracy, step_time/60, fetch_time/60))
                              step_time = 0.0
                              fetch_time = 0.0


                  #Validate and test after each epoch
                  val_it = validate_dataset.make_one_shot_iterator()
                  val_next_elem = val_it.get_next()
                  tot_tp, tot_tn, tot_fp, tot_fn = 0, 0, 0, 0
                  while True:
                        try:
                              sample, label = sess.run(val_next_elem)
                              tp, fp, tn, fn = sess.run([tp_op, fp_op, tn_op, fn_op],
                                                        feed_dict={x: sample, y_: label, dropout_rate: 1.0, is_training: False})
                        except tf.errors.OutOfRangeError:
                              break
                        tot_tp += tp
                        tot_fp += fp
                        tot_fn += fn
                        tot_tn += tn
                  precision, recall, accuracy, fscore, summ = sess.run([precision_op, recall_op, accuracy_op, fscore_op, validate_summaries],
                                                                       feed_dict={tp_sum: tot_tp, tn_sum: tot_tn, fp_sum: tot_fp, fn_sum: tot_fn})
                  validate_writer.add_summary(summ, global_step)
                  print ("Epoch %d, Step %d" % (epoch, global_step))
                  print ("="*10, "Validating Results", "="*10)
                  print ("TP: %g\nTN: %g\nFP: %g\nFN: %g" % (tot_tp, tot_tn, tot_fp, tot_fn))
                  print ("\tPrecision: %g\n\tRecall: %g\n\tF1_score: %g\n\tAccuracy: %g" % (precision, recall, fscore, accuracy))


                  test_it = test_dataset.make_one_shot_iterator()
                  test_next_elem = test_it.get_next()
                  tot_tp, tot_tn, tot_fp, tot_tn = 0, 0, 0, 0
                  while True:
                        try:
                              sample, label = sess.run(test_next_elem)
                              tp, fp, tn, fn = sess.run([tp_op, fp_op, tn_op, fn_op],
                                                        feed_dict={x: sample, y_: label, dropout_rate: 1.0, is_training: False})
                        except tf.errors.OutOfRangeError:
                              break
                        tot_tp += tp
                        tot_fp += fp
                        tot_fn += fn
                        tot_tn += tn
                  precision, recall, accuracy, fscore, summ = sess.run([precision_op, recall_op, accuracy_op, fscore_op, test_summaries],
                                                                       feed_dict={tp_sum: tot_tp, tn_sum: tot_tn, fp_sum: tot_fp, fn_sum: tot_fn})

                  test_writer.add_summary(summ, global_step)

                  print ("="*10, "Testing Results", "="*10)
                  print ("TP: %g\nTN: %g\nFP: %g\nFN: %g" % (tot_tp, tot_tn, tot_fp, tot_fn))
                  print ("\tPrecision: %g\n\tRecall: %g\n\tF1_score: %g\n\tAccuracy: %g" % (precision, recall, fscore, accuracy))
                  print ("="*10, "===============", "="*10)

      
if __name__ == "__main__":
      tf.app.run()
