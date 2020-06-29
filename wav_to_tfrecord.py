%tensorflow_version 1.x
import io
import os
import numpy as np
import scipy.io.wavfile
import tensorflow as tf

output_path = os.path.expanduser('drive/My Drive/dataset/tfrecords/SI2203.WAV.tfrecord')
sample_path = os.path.expanduser('drive/My Drive/dataset/FAKS0/SI2203.WAV.wav')

record_writer = tf.io.TFRecordWriter(output_path)
sample_rate, audio = scipy.io.wavfile.read(sample_path)

# Put in range [-1, 1]
float_normalizer = float(np.iinfo(np.int16).max)
audio = audio / float_normalizer

# Convert to mono
audio = np.mean(audio, axis=-1)

print(audio)
print(audio.shape)

example = tf.train.Example(features=tf.train.Features(feature={
    'sample_rate': tf.train.Feature(
        int64_list=tf.train.Int64List(value=[sample_rate])),
    'audio': tf.train.Feature(float_list=tf.train.FloatList(value=[audio])),
}))
record_writer.write(example.SerializeToString())
