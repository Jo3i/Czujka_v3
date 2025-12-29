import numpy as np
import librosa
import tensorflow as tf
from . import vggish_input, vggish_slim, vggish_params


class VGGishExtractor:
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.session = tf.compat.v1.Session()
            vggish_slim.define_vggish_slim()
            self.session.run(tf.compat.v1.global_variables_initializer())

    def extract(self, audio, sample_rate=16000):
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

        examples = vggish_input.waveform_to_examples(audio, 16000)

        features_tensor = self.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME
        )
        embedding_tensor = self.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME
        )

        embeddings = self.session.run(
            embedding_tensor,
            feed_dict={features_tensor: examples}
        )

        return np.mean(embeddings, axis=0)
