from mxnet.gluon.rnn import HybridRecurrentCell
from mxnet import ndarray, nd
from mxnet.gluon.rnn import LSTM, BidirectionalCell

class LSTMPCell(HybridRecurrentCell):
    r"""Long-Short Term Memory (LSTM) network cell.

    Each call computes the following function:

    .. math::
        \begin{array}{ll}
        i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
        f_t = sigmoid(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
        g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
        o_t = sigmoid(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
        c_t = f_t * c_{(t-1)} + i_t * g_t \\
        h_t = o_t * \tanh(c_t)
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the
    cell state at time `t`, :math:`x_t` is the hidden state of the previous
    layer at time `t` or :math:`input_t` for the first layer, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell, and
    out gates, respectively.

    Parameters
    ----------
    hidden_size : int
        Number of units in output symbol.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default 'lstmbias'
        Initializer for the bias vector. By default, bias for the forget
        gate is initialized to 1 while all other biases are initialized
        to zero.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'lstm_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.


    Inputs:
        - **data**: input tensor with shape `(batch_size, input_size)`.
        - **states**: a list of two initial recurrent state tensors. Each has shape
          `(batch_size, num_hidden)`.

    Outputs:
        - **out**: output tensor with shape `(batch_size, num_hidden)`.
        - **next_states**: a list of two output recurrent state tensors. Each has
          the same shape as `states`.
    """
    def __init__(self, hidden_size, cell_size,
                 i2h_weight_initializer=None, h2h_weight_initializer=None, h2h_bias_initializer='lstmbias',
                 h2proj_weight_initializer=None, input_size=0,
                 memory_cell_clip_value=None, state_projection_clip_value=None, prefix=None, params=None):
        super(LSTMPCell, self).__init__(prefix=prefix, params=params)

        self._hidden_size = hidden_size
        self._cell_size = cell_size
        self._input_size = input_size
        self.memory_cell_clip_value = memory_cell_clip_value
        self.state_projection_clip_value = state_projection_clip_value
        self.i2h_weight = self.params.get('i2h_weight', shape=(4*cell_size, input_size),
                                          init=i2h_weight_initializer,
                                          allow_deferred_init=True)
        self.h2h_weight = self.params.get('h2h_weight', shape=(4*cell_size, hidden_size),
                                          init=h2h_weight_initializer,
                                          allow_deferred_init=True)
        self.h2h_bias = self.params.get('h2h_bias', shape=(4*cell_size,),
                                        init=h2h_bias_initializer,
                                        allow_deferred_init=True)
        self.h2proj_weight = self.params.get('h2proj_weight', shape=(hidden_size, cell_size),
                                             init=h2proj_weight_initializer, allow_deferred_init=True)

    def state_info(self, batch_size=0):
        return [{'shape': (batch_size, self._hidden_size), '__layout__': 'NC'},
                {'shape': (batch_size, self._cell_size), '__layout__': 'NC'}]

    def _alias(self):
        return 'lstm'

    def __repr__(self):
        s = '{name}({mapping})'
        shape = self.i2h_weight.shape
        mapping = '{0} -> {1}'.format(shape[1] if shape[1] else None, shape[0])
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)

    def hybrid_forward(self, F, inputs, states, i2h_weight,
                       h2h_weight, h2h_bias, h2proj_weight):
        prefix = 't%d_'%self._counter
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=None,
                               num_hidden=self._cell_size*4, name=prefix+'i2h', no_bias=True)
        h2h = F.FullyConnected(data=states[0], weight=h2h_weight, bias=h2h_bias,
                               num_hidden=self._cell_size*4, name=prefix+'h2h')
        gates = i2h + h2h
        slice_gates = F.SliceChannel(gates, num_outputs=4, name=prefix+'slice')
        in_gate = F.Activation(slice_gates[0], act_type="sigmoid", name=prefix+'i')
        forget_gate = F.Activation(slice_gates[1], act_type="sigmoid", name=prefix+'f')
        in_transform = F.Activation(slice_gates[2], act_type="tanh", name=prefix+'c')
        out_gate = F.Activation(slice_gates[3], act_type="sigmoid", name=prefix+'o')
        next_c = F._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                   name=prefix+'state')
        next_h = F._internal._mul(out_gate, F.Activation(next_c, act_type="tanh"),
                                  name=prefix+'out')

        if self.memory_cell_clip_value:
            next_c.clip(a_min=-self.memory_cell_clip_value, a_max=self.memory_cell_clip_value, out=next_c)

        next_h_proj = F.FullyConnected(data=next_h, weight=h2proj_weight, bias=None, num_hidden=self._hidden_size, no_bias=True, name=prefix+'h2proj')

        if self.state_projection_clip_value:
            next_h_proj.clip(a_min=-self.state_projection_clip_value, a_max=self.state_projection_clip_value, out=next_h_proj)

        return next_h_proj, [next_h_proj, next_c]




# d = LSTMPCell(30, 40, input_size=10, memory_cell_clip_value=3, state_projection_clip_value=3)
# d.collect_params().initialize()
# d.hybridize()
# output = d(nd.ones((2,10)), [nd.ones((2,30)), nd.ones((2,40))])



# class BidirectionalCell(HybridRecurrentCell):
#     """Bidirectional RNN cell.
#
#     Parameters
#     ----------
#     l_cell : RecurrentCell
#         Cell for forward unrolling
#     r_cell : RecurrentCell
#         Cell for backward unrolling
#     """
#     def __init__(self, l_cell, r_cell, output_prefix='bi_'):
#         super(BidirectionalCell, self).__init__(prefix='', params=None)
#         self.register_child(l_cell, 'l_cell')
#         self.register_child(r_cell, 'r_cell')
#         self._output_prefix = output_prefix
#
#     def __call__(self, inputs, states):
#         raise NotImplementedError("Bidirectional cannot be stepped. Please use unroll")
#
#     def __repr__(self):
#         s = '{name}(forward={l_cell}, backward={r_cell})'
#         return s.format(name=self.__class__.__name__,
#                         l_cell=self._children['l_cell'],
#                         r_cell=self._children['r_cell'])
#
#     def state_info(self, batch_size=0):
#         return _cells_state_info(self._children.values(), batch_size)
#
#     def begin_state(self, **kwargs):
#         assert not self._modified, \
#             "After applying modifier cells (e.g. DropoutCell) the base " \
#             "cell cannot be called directly. Call the modifier cell instead."
#         return _cells_begin_state(self._children.values(), **kwargs)
#
#     def unroll(self, length, inputs, begin_state=None, layout='NTC', merge_outputs=None,
#                valid_length=None):
#         self.reset()
#
#         inputs, axis, F, batch_size = _format_sequence(length, inputs, layout, False)
#         if valid_length is None:
#             reversed_inputs = list(reversed(inputs))
#         else:
#             reversed_inputs = F.SequenceReverse(F.stack(*inputs, axis=0),
#                                                 sequence_length=valid_length,
#                                                 use_sequence_length=True)
#             reversed_inputs = _as_list(F.split(reversed_inputs, axis=0, num_outputs=length,
#                                                squeeze_axis=True))
#         begin_state = _get_begin_state(self, F, begin_state, inputs, batch_size)
#
#         states = begin_state
#         l_cell, r_cell = self._children.values()
#         l_outputs, l_states = l_cell.unroll(length, inputs=inputs,
#                                             begin_state=states[:len(l_cell.state_info(batch_size))],
#                                             layout=layout, merge_outputs=merge_outputs,
#                                             valid_length=valid_length)
#         r_outputs, r_states = r_cell.unroll(length,
#                                             inputs=reversed_inputs,
#                                             begin_state=states[len(l_cell.state_info(batch_size)):],
#                                             layout=layout, merge_outputs=False,
#                                             valid_length=valid_length)
#         if valid_length is None:
#             reversed_r_outputs = list(reversed(r_outputs))
#         else:
#             reversed_r_outputs = F.SequenceReverse(F.stack(*r_outputs, axis=0),
#                                                    sequence_length=valid_length,
#                                                    use_sequence_length=True,
#                                                    axis=0)
#             reversed_r_outputs = _as_list(F.split(reversed_r_outputs, axis=0, num_outputs=length,
#                                                   squeeze_axis=True))
#         if merge_outputs is None:
#             merge_outputs = isinstance(l_outputs, tensor_types)
#             l_outputs, _, _, _ = _format_sequence(None, l_outputs, layout, merge_outputs)
#             reversed_r_outputs, _, _, _ = _format_sequence(None, reversed_r_outputs, layout,
#                                                            merge_outputs)
#
#         if merge_outputs:
#             reversed_r_outputs = F.stack(*reversed_r_outputs, axis=axis)
#             outputs = F.concat(l_outputs, reversed_r_outputs, dim=2,
#                                name='%sout'%self._output_prefix)
#
#         else:
#             outputs = [F.concat(l_o, r_o, dim=1, name='%st%d'%(self._output_prefix, i))
#                        for i, (l_o, r_o) in enumerate(zip(l_outputs, reversed_r_outputs))]
#         if valid_length is not None:
#             outputs = _mask_sequence_variable_length(F, outputs, length, valid_length, axis,
#                                                      merge_outputs)
#         states = l_states + r_states
#         return (l_outputs, reversed_r_outputs), (l_states, r_states)

# b = BidirectionalCell(LSTMPCell(20, 30, input_size=10, memory_cell_clip_value=3, state_projection_clip_value=3), LSTMPCell(20, 30, input_size=10, memory_cell_clip_value=3, state_projection_clip_value=3))
# b.collect_params().initialize()
# states = b.begin_state(batch_size=2)
# b.hybridize()
# output, states = b.unroll(5, nd.ones((5,2,10)), begin_state=states, layout='TNC', split_l_r=True, merge_outputs=True)
# print('hi')