from __future__ import absolute_import, division, print_function, unicode_literals

# tensorflow-gpu must be installed
import tensorflow as tf
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
