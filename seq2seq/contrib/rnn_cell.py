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

    def get_attention_scores(self, attention_network_input, state
                             # , window
                             , network_type):
        # TODO: Add residual connections

        state_expanded = tf.expand_dims(state, axis=1)

        state_tiled = tf.tile(state_expanded, [1, tf.shape(attention_network_input)[1], 1])

        input_concated = tf.concat([attention_network_input, state_tiled], 2)

        attention_scores = tf.contrib.layers.stack(input_concated,
                                                   tf.contrib.layers.fully_connected,
                                                   [
                                                       self.attention_num_units] * self.attention_num_layers + [
                                                       # window
                                                       1
                                                   ],
                                                   activation_fn=tf.nn.relu, scope=network_type,
                                                   reuse=False)

        attention_values = tf.nn.softmax(attention_scores, dim=1)

        attention_values = tf.Print(attention_values, [attention_values[0]], message="##########attention_values:",
                                  summarize=100)

        return attention_values

    def read_attention_network(self, network_type, attention_network_input, state,
                               # window,
                               ):
        attention_values = self.get_attention_scores(attention_network_input, state,  # window,
                                                     network_type)

        return tf.reduce_sum(tf.multiply(attention_network_input, attention_values), 1)

    def write_attention_network(self, network_type, attention_network_input, state, output,
                                # window,
                                ):
        attention_values = self.get_attention_scores(attention_network_input, state,  # window,
                                                     network_type)

        output_expanded = tf.expand_dims(output, axis=1)

        output_tiled = tf.tile(output_expanded, [1, tf.shape(attention_network_input)[1], 1])

        output_without_positional_embeddings = tf.slice(output_tiled, [0, 0, 0],
                                                        [-1, -1, self.embedding_size])

        orig_positional_embeddings = tf.slice(attention_network_input, [0, 0, self.embedding_size],
                                              [-1, -1, -1])

        output_with_original_positional_embeddings = tf.concat(
            [output_without_positional_embeddings, orig_positional_embeddings], axis=2)

        # TODO TODO TODO: Don't change the positional embedding part of outputs

        weighted_new = tf.multiply(output_with_original_positional_embeddings, attention_values)
        weighted_original = tf.multiply(attention_network_input, 1 - attention_values)

        return weighted_new + weighted_original

    def __init__(self, inner_cell, embedding_size, positional_embedding_size, attention_num_layers,
                 attention_num_units):
        super(AttentionRNNCell, self).__init__()  # reuse? refer to original code for clarity
        self.cell = inner_cell
        self.attention_num_layers = attention_num_layers
        self.attention_num_units = attention_num_units
        # TODO : Make attention function for attention layer configurable
        # Todo : Multiple read and writes? read write window?
        self.embedding_size = embedding_size
        self.positional_embedding_size = positional_embedding_size

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
        if isinstance(state, list) or len(state) != 3:
            raise TypeError("state is not a list of length 3")

        with tf.variable_scope(scope) as var_scope:
            attention_weighted_source = self.read_attention_network('source_read', state[1], state[0]
                                                                    # , self.read_window,
                                                                    )

            attention_weighted_context = self.read_attention_network('context_read', state[2], state[0]
                                                                     # ,self.read_window
                                                                     )

            cell_input = tf.concat(
                [attention_weighted_source, attention_weighted_context
                 # , attention_weighted_context
                 ], 1)

            output, state_0 = self.cell(cell_input, state[0])

            state_2 = self.write_attention_network('context_write', state[2], state_0, output,
                                                   # TODO: output or state?
                                                   # self.write_window,
                                                   )

        return output, (state_0, state[1], state_2)

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        return self.cell.state_size, tf.TensorShape(
            [None, self.embedding_size + self.positional_embedding_size]), tf.TensorShape(
            [None, self.embedding_size + self.positional_embedding_size])

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        return self.cell.output_size
