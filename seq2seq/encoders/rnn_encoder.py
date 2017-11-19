# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Collection of RNN encoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn

from seq2seq.encoders.encoder import Encoder, EncoderOutput
from seq2seq.training import utils as training_utils
from seq2seq.contrib.rnn_cell import AttentionRNNCell


def _unpack_cell(cell):
    """Unpack the cells because the stack_bidirectional_dynamic_rnn
    expects a list of cells, one per layer."""
    if isinstance(cell, tf.contrib.rnn.MultiRNNCell):
        return cell._cells  # pylint: disable=W0212
    else:
        return [cell]


def _default_rnn_cell_params():
    """Creates default parameters used by multiple RNN encoders.
    """
    return {
        "cell_class": "BasicLSTMCell",
        "cell_params": {
            "num_units": 128
        },
        "dropout_input_keep_prob": 1.0,
        "dropout_output_keep_prob": 1.0,
        "num_layers": 1,
        "residual_connections": False,
        "residual_combiner": "add",
        "residual_dense": False
    }


def _toggle_dropout(cell_params, mode):
    """Disables dropout during eval/inference mode
    """
    cell_params = copy.deepcopy(cell_params)
    if mode != tf.contrib.learn.ModeKeys.TRAIN:
        cell_params["dropout_input_keep_prob"] = 1.0
        cell_params["dropout_output_keep_prob"] = 1.0
    return cell_params


class UnidirectionalRNNEncoder(Encoder):
    """
    A unidirectional RNN encoder. Stacking should be performed as
    part of the cell.

    Args:
      cell: An instance of tf.contrib.rnn.RNNCell
      name: A name for the encoder
    """

    def __init__(self, params, mode, name="forward_rnn_encoder"):
        super(UnidirectionalRNNEncoder, self).__init__(params, mode, name)
        self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

    @staticmethod
    def default_params():
        return {
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }

    def encode(self, inputs, sequence_length, **kwargs):
        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        cell = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)
        return EncoderOutput(
            outputs=outputs,
            final_state=state,
            attention_values=outputs,
            attention_values_length=sequence_length)


class BidirectionalRNNEncoder(Encoder):
    """
    A bidirectional RNN encoder. Uses the same cell for both the
    forward and backward RNN. Stacking should be performed as part of
    the cell.

    Args:
      cell: An instance of tf.contrib.rnn.RNNCell
      name: A name for the encoder
    """

    def __init__(self, params, mode, name="bidi_rnn_encoder"):
        super(BidirectionalRNNEncoder, self).__init__(params, mode, name)
        self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

    @staticmethod
    def default_params():
        return {
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }

    def encode(self, inputs, sequence_length, **kwargs):
        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        cell_fw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        cell_bw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32,
            **kwargs)

        # Concatenate outputs and states of the forward and backward RNNs
        outputs_concat = tf.concat(outputs, 2)

        return EncoderOutput(
            outputs=outputs_concat,
            final_state=states,
            attention_values=outputs_concat,
            attention_values_length=sequence_length)


class StackBidirectionalRNNEncoder(Encoder):
    """
    A stacked bidirectional RNN encoder. Uses the same cell for both the
    forward and backward RNN. Stacking should be performed as part of
    the cell.

    Args:
      cell: An instance of tf.contrib.rnn.RNNCell
      name: A name for the encoder
    """

    def __init__(self, params, mode, name="stacked_bidi_rnn_encoder"):
        super(StackBidirectionalRNNEncoder, self).__init__(params, mode, name)
        self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)

    @staticmethod
    def default_params():
        return {
            "rnn_cell": _default_rnn_cell_params(),
            "init_scale": 0.04,
        }

    def encode(self, inputs, sequence_length, **kwargs):
        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        cell_fw = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        cell_bw = training_utils.get_rnn_cell(**self.params["rnn_cell"])

        cells_fw = _unpack_cell(cell_fw)
        cells_bw = _unpack_cell(cell_bw)

        result = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=cells_fw,
            cells_bw=cells_bw,
            inputs=inputs,
            dtype=tf.float32,
            sequence_length=sequence_length,
            **kwargs)
        outputs_concat, _output_state_fw, _output_state_bw = result
        final_state = (_output_state_fw, _output_state_bw)
        return EncoderOutput(
            outputs=outputs_concat,
            final_state=final_state,
            attention_values=outputs_concat,
            attention_values_length=sequence_length)


class DynamicDirectionalRNNEncoder(Encoder):
    """
    A bidirectional RNN encoder. Uses the same cell for both the
    forward and backward RNN. Stacking should be performed as part of
    the cell.

    Args:
      cell: An instance of tf.contrib.rnn.RNNCell
      name: A name for the encoder
    """

    def __init__(self, params, mode, name="dyna_rnn_encoder"):
        super(DynamicDirectionalRNNEncoder, self).__init__(params, mode, name)
        self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)
        self.max_sequence_length = self.params[
            "source_max_seq_len"]  # TODO: Different length for input and context vectors?
        self.positional_embedding_size = self.params["positional_embedding_size"]
        self.embedding_size = self.params["embedding_size"]  # TODO: Different size for words and context
        self.attention_num_layers = self.params["attention_num_layers"]
        self.attention_num_units = self.params["attention_num_units"]

    @staticmethod
    def default_params():
        return {
            "rnn_cell": _default_rnn_cell_params(),
            "source_max_seq_len": 50,
            "positional_embedding_size": 32,
            "embedding_size": 128,
            "attention_num_layers": 3,
            "attention_num_units": 32,
            "init_scale": 0.04,
        }

    def encode(self, inputs, sequence_length, **kwargs):
        scope = tf.get_variable_scope()
        scope.set_initializer(tf.random_uniform_initializer(
            -self.params["init_scale"],
            self.params["init_scale"]))

        inner_cell = training_utils.get_rnn_cell(**self.params["rnn_cell"])
        rnn_param_list = [self.embedding_size,
                          self.positional_embedding_size,
                          self.attention_num_layers,
                          self.attention_num_units]
        cell = AttentionRNNCell(inner_cell=inner_cell, cell_params=self.params["rnn_cell"],
                                rnn_param_list=rnn_param_list)
        batch_size = tf.shape(inputs)[0]
        # initial_state = cell.zero_state(batch_size, tf.float32)  # TODO: Make dtype configurable?

        positional_embeddings = tf.get_variable("positional_embeddings",
                                                [self.max_sequence_length, self.positional_embedding_size],
                                                dtype=tf.float32)  # TODO: Make dtype configurable

        position_sequence = tf.range(tf.shape(inputs)[1])

        positional_embeddings = tf.nn.embedding_lookup(positional_embeddings, position_sequence)

        positional_embeddings = tf.expand_dims(positional_embeddings, axis=0)

        positional_embeddings_for_batch = tf.tile(positional_embeddings, [tf.shape(inputs)[0], 1, 1])

        initial_state_0 = tf.zeros([tf.shape(inputs)[0], inner_cell.state_size])

        initial_state_1 = tf.concat([inputs, positional_embeddings_for_batch], axis=2)

        initial_state_2 = tf.concat([tf.zeros_like(inputs), positional_embeddings_for_batch], axis=2)

        initial_state = (initial_state_0, initial_state_1, initial_state_2)

        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=tf.zeros([tf.shape(inputs)[0], tf.shape(inputs)[1] * 2, 1], tf.float32),
            initial_state=initial_state,
            sequence_length=sequence_length * 2,  # Todo : Make this 2 configurable
            dtype=tf.float32,
            **kwargs)

        # state_0 = state[0]
        # state_2 = state[2]
        # outputs = tf.Print(outputs, [tf.shape(outputs)], message="##1#############################################")
        # state_0 = tf.Print(state_0, [tf.shape(state_0)], message="###2############################################")
        # state_2 = tf.Print(state_2, [tf.shape(state_2)], message="####3###########################################")

        return EncoderOutput(
            outputs=state[2],
            final_state=state[0],
            attention_values=state[2],
            attention_values_length=sequence_length * 2)
