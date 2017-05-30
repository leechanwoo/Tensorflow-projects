
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector 
import csv

LOG_DIR = "./tensorboard/embeddings/"

with open(LOG_DIR+"label.csv", 'w') as f:
    writer = csv.writer(f)
    for i in range(1000):
        if i % 2 == 0:
            writer.writerow([0])
        else:
            writer.writerow([1])
    

embedding_var = tf.Variable(tf.random_uniform(shape=(1000, 100), minval=-1.0, maxval=1.0), dtype=tf.float32, name='word_embedding')
config = projector.ProjectorConfig()

embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name
embedding.metadata_path = LOG_DIR + 'label.csv' 
summary_writer = tf.summary.FileWriter(LOG_DIR)

projector.visualize_embeddings(summary_writer, config)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.save(sess, LOG_DIR+'embedding')