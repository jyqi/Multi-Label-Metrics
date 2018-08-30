import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score 

labels = (np.random.randn(100, 10) > 0.5).astype(int)
predictions = (np.random.randn(100, 10) > 0.5).astype(int)
mic = f1_score(labels, predictions, average="micro")
mac = f1_score(labels, predictions, average="macro")
wei = f1_score(labels, predictions, average="weighted")
print(mic, mac, wei)

# numpy实现precision,recall,micro_f1
bool_matrix = np.logical_and(np.equal(labels,1), np.equal(predictions,1)) + 0
tp = np.sum(bool_matrix)
bool_matrix = np.logical_and(np.equal(labels,0), np.equal(predictions,1)) + 0
fp = np.sum(bool_matrix)
bool_matrix = np.logical_and(np.equal(labels,1), np.equal(predictions,0)) + 0
fn = np.sum(bool_matrix)
precision = tp*1./(tp+fp)
recall = tp*1./(tp+fn)
f1 = 2 * precision * recall / (precision + recall)
print(f1)

#tensorflow方法
y_true = tf.Variable(labels)
y_pred = tf.Variable(predictions)
a = tf.Variable(2.)
b = tf.Variable(3.)
c = tf.assign_add(a,b)
# 方法1
tp_1 = tf.count_nonzero(y_pred * y_true, axis=0)
fp_1 = tf.count_nonzero(y_pred * (y_true - 1), axis=0)
fn_1 = tf.count_nonzero((y_pred - 1) * y_true, axis=0)

# 方法2
bool_matrix = tf.logical_and(tf.equal(y_pred,1),tf.equal(y_true,1))
tp_2 = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[0,1])

bool_matrix = tf.logical_and(tf.equal(y_pred,1),tf.equal(y_true,0))
fp_2 = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[0,1])

bool_matrix = tf.logical_and(tf.equal(y_pred,0),tf.equal(y_true,1))
fn_2 = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[0,1])



with tf.Session() as sess:
	tf.global_variables_initializer().run()
	# print(sess.run(tp_1))
	# print(sess.run(fp_1))
	# print(sess.run(fn_1))
	TP = sess.run(tp_2)
	FP = sess.run(fp_2)
	FN = sess.run(fn_2)
	# print(sess.run(c))
	# print(sess.run(a))
	precision = TP/(TP+FP)
	recall = TP/(TP+FN)
	f1 = 2 * precision * recall / (precision + recall)
	print(precision,recall,f1)
