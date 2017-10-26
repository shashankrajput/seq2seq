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
"""Collection of RNN Cells
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import inspect

import tensorflow as tf
from tensorflow.python.ops import array_ops  # pylint: disable=E0611
from tensorflow.python.util import nest  # pylint: disable=E0611
from tensorflow.contrib.rnn import MultiRNNCell  # pylint: disable=E0611
from seq2seq.training import utils as training_utils

# Import all cell classes from Tensorflow
TF_CELL_CLASSES = [
    x for x in tf.contrib.rnn.__dict__.values()
    if inspect.isclass(x) and issubclass(x, tf.contrib.rnn.RNNCell)
]
for cell_class in TF_CELL_CLASSES:
    setattr(sys.modules[__name__], cell_class.__name__, cell_class)


class ExtendedMultiRNNCell(MultiRNNCell):
    """Extends the Tensorflow MultiRNNCell with residual connections"""

    def __init__(self,
                 cells,
                 residual_connections=False,
                 residual_combiner="add",
                 residual_dense=False):
        """Create a RNN cell composed sequentially of a number of RNNCells.

        Args:
          cells: list of RNNCells that will be composed in this order.
          state_is_tuple: If True, accepted and returned states are n-tuples, where
            `n = len(cells)`.  If False, the states are all
            concatenated along the column axis.  This latter behavior will soon be
            deprecated.
          residual_connections: If true, add residual connections between all cells.
            This requires all cells to have the same output_size. Also, iff the
            input size is not equal to the cell output size, a linear transform
            is added before the first layer.
          residual_combiner: One of "add" or "concat". To create inputs for layer
            t+1 either "add" the inputs from the prev layer or concat them.
          residual_dense: Densely connect each layer to all other layers

        Raises:
          ValueError: if cells is empty (not allowed), or at least one of the cells
            returns a state tuple but the flag `state_is_tuple` is `False`.
        """
        super(ExtendedMultiRNNCell, self).__init__(cells, state_is_tuple=True)
        assert residual_combiner in ["add", "concat", "mean"]

        self._residual_connections = residual_connections
        self._residual_combiner = residual_combiner
        self._residual_dense = residual_dense

    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        if not self._residual_connections:
            return super(ExtendedMultiRNNCell, self).__call__(
                inputs, state, (scope or "extended_multi_rnn_cell"))

        with tf.variable_scope(scope or "extended_multi_rnn_cell"):
            # Adding Residual connections are only possible when input and output
            # sizes are equal. Optionally transform the initial inputs to
            # `cell[0].output_size`
            if self._cells[0].output_size != inputs.get_shape().as_list()[1] and \
                    (self._residual_combiner in ["add", "mean"]):
                inputs = tf.contrib.layers.fully_connected(
                    inputs=inputs,
                    num_outputs=self._cells[0].output_size,
                    activation_fn=None,
                    scope="input_transform")

            # Iterate through all layers (code from MultiRNNCell)
            cur_inp = inputs
            prev_inputs = [cur_inp]
            new_states = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("cell_%d" % i):
                    if not nest.is_sequence(state):
                        raise ValueError(
                            "Expected state to be a tuple of length %d, but received: %s" %
                            (len(self.state_size), state))
                    cur_state = state[i]
                    next_input, new_state = cell(cur_inp, cur_state)

                    # Either combine all previous inputs or only the current input
                    input_to_combine = prev_inputs[-1:]
                    if self._residual_dense:
                        input_to_combine = prev_inputs

                    # Add Residual connection
                    if self._residual_combiner == "add":
                        next_input = next_input + sum(input_to_combine)
                    if self._residual_combiner == "mean":
                        combined_mean = tf.reduce_mean(tf.stack(input_to_combine), 0)
                        next_input = next_input + combined_mean
                    elif self._residual_combiner == "concat":
                        next_input = tf.concat([next_input] + input_to_combine, 1)
                    cur_inp = next_input
                    prev_inputs.append(cur_inp)

                    new_states.append(new_state)
        new_states = (tuple(new_states)
                      if self._state_is_tuple else array_ops.concat(new_states, 1))
        return cur_inp, new_states


class AttentionRNNCell(tf.contrib.rnn.RNNCell):
    """Abstract object representing an RNN cell.

    Every `RNNCell` must have the properties below and implement `call` with
    the signature `(output, next_state) = call(input, state)`.  The optional
    third input argument, `scope`, is allowed for backwards compatibility
    purposes; but should be left off for new subclasses.

    This definition of cell differs from the definition used in the literature.
    In the literature, 'cell' refers to an object with a single scalar output.
    This definition refers to a horizontal array of such units.

    An RNN cell, in the most abstract setting, is anything that has
    a state and performs some operation that takes a matrix of inputs.
    This operation results in an output matrix with `self.output_size` columns.
    If `self.state_size` is an integer, this operation also results in a new
    state matrix with `self.state_size` columns.  If `self.state_size` is a
    (possibly nested tuple of) TensorShape object(s), then it should return a
    matching structure of Tensors having shape `[batch_size].concatenate(s)`
    for each `s` in `self.batch_size`.
    """

    def get_attention_scores(self, attention_network_input, state, window, network_type):

        ########## Add residual connections

        attention_network_input_split = tf.split(attention_network_input, num_or_size_splits=self.max_sequence_length,
                                                 axis=1)
        attention_scores = []

        local_reuse = False

        for attention_network_input_i in attention_network_input_split:
            fully_connected_reuse = local_reuse or self.network_reuse

            attention_scores.append(
                tf.contrib.layers.stack(tf.concat([attention_network_input_i, state], 1),
                                        tf.contrib.layers.fully_connected,
                                        [
                                            self.attention_num_units] * self.attention_num_layers + [
                                            window],
                                        activation_fn=self.attention_activation, scope=network_type,
                                        reuse=fully_connected_reuse))

            local_reuse = True

            attention_scores_stacked = tf.stack(attention_scores, axis=1)
            attention_values = tf.nn.softmax(attention_scores_stacked, dim=1)
            return attention_values

    def read_attention_network(self, network_type, attention_network_input, state, window):

        attention_values = self.get_attention_scores(attention_network_input, state, window,
                                                     network_type)

        attention_values_split = tf.split(attention_values, num_or_size_splits=window, axis=-1)

        attention_weighted_inputs = []
        for attention_values_i in attention_values_split:
            attention_weighted_inputs.append(tf.reduce_sum(tf.multiply(attention_network_input, attention_values_i), 1))

        return tf.stack(attention_weighted_inputs, axis=1)

    def write_attention_network(self, network_type, attention_network_input, state, window):

        attention_values = self.get_attention_scores(attention_network_input, state, window,
                                                     network_type)

        attention_values_split = tf.split(attention_values, num_or_size_splits=window, axis=-1)

        state_stacked = tf.stack([state] * self.max_input_length, axis=1)

        attention_weighted_inputs = []
        for attention_values_i in attention_values_split:
            if network_type == "target_write":
                target_sentences_projected = tf.multiply(attention_network_input,
                                                         1 - attention_values_i) + tf.multiply(state_stacked,
                                                                                               attention_values_i)

    def __init__(self, cell_params):

        super(AttentionRNNCell, self).__init__(_reuse=cell_params['reuse'])
        self.cell = training_utils.get_rnn_cell(**cell_params)
        self.max_input_length = cell_params["max_input_length"]  # TODO: Different length for context vector?
        self.positional_embedding_size = cell_params["positional_embedding_size"]
        self.attention_num_layers = cell_params["attention_num_layers"]
        self.attention_num_units = cell_params["attention_num_units"]
        self.attention_activation = cell_params["attention_activation"]
        self.network_reuse = False
        self.read_window = cell_params["read_window"]
        self.write_window = cell_params["write_window"]

        self.positional_embeddings = tf.get_variable("positional_embeddings",
                                                     [self.max_input_length, self.positional_embedding_size],
                                                     dtype=tf.float32)  # TODO: Make dtype configurable

        Implement
        this: embedding_lookup_indices = tf.multiply(tf.ones([batch_size], dtype=tf.int32, name='current_position'),
                                                     word_index)

        Implement
        this: positional_embedding = tf.nn.embedding_lookup(self.positional_embeddings, embedding_lookup_indices)

        Make
        initial
        state

    def call(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.

        Args:
          inputs: `2-D` tensor with shape `[batch_size x input_size]`.
          state: if `self.state_size` is an integer, this should be a `2-D Tensor`
            with shape `[batch_size x self.state_size]`.  Otherwise, if
            `self.state_size` is a tuple of integers, this should be a tuple
            with shapes `[batch_size x s] for s in self.state_size`.
          scope: VariableScope for the created subgraph; defaults to class name.

        Returns:
          A pair containing:

          - Output: A `2-D` tensor with shape `[batch_size x self.output_size]`.
          - New state: Either a single `2-D` tensor, or a tuple of tensors matching
            the arity and shapes of `state`.
        """
        if (isinstance(state, list) or len(state) != 3):
            raise TypeError("state is not a list of length 3")

        attention_weighted_source = self.read_attention_network('source_read', state[1], state[0], self.read_window)
        attention_weighted_target = self.read_attention_network('target_read', state[2], state[0], self.read_window)

        cell_input = tf.concat(
            [attention_weighted_source, attention_weighted_target
             # , attention_weighted_context
             ], 1)

        output, state[0] = self.cell(cell_input, state[0])

        self.write_attention_network('target_write', state[2], state[0], ###############333
                                     self.write_window)  # Will call by reference work?????????????????????

        self.network_reuse = True

        return output, state

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        return sdsdsds

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.

        Returns:
          If `state_size` is an int or TensorShape, then the return value is a
          `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

          If `state_size` is a nested list or tuple, then the return value is
          a nested list or tuple (of the same structure) of `2-D` tensors with
          the shapes `[batch_size x s]` for each s in `state_size`.
        """
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            state_size = self.state_size
            return _zero_state_tensors(state_size, batch_size, dtype)
