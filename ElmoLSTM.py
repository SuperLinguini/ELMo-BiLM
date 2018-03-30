
from mxnet import gluon, nd
from mxnet.gluon import rnn
from mxnet.gluon.model_zoo.text.base import get_rnn_cell


class ElmoLSTM(gluon.Block):
    def __init__(self, mode, num_layers, input_size, hidden_size, dropout, weight_dropout, bidirectional=True):
        super(ElmoLSTM, self).__init__()

        self.num_layers = num_layers

        forward_layers = []
        backward_layers = []

        with self.name_scope():
            for layer_index in range(num_layers):
                forward_layer = get_rnn_cell(mode, 1, input_size, hidden_size, dropout, weight_dropout, 0, 0, 0)
                backward_layer = get_rnn_cell(mode, 1, input_size, hidden_size, dropout, weight_dropout, 0, 0, 0)

                # setattr(self, 'forward_cell_{}'.format(layer_index), forward_layer)
                # setattr(self, 'backward_cell_{}'.format(layer_index), backward_layer)
                self.register_child(forward_layer)
                self.register_child(backward_layer)

                forward_layers.append(forward_layer)
                backward_layers.append(backward_layer)
            self.forward_layers = forward_layers
            self.backward_layers = backward_layers

    def begin_state(self, *args, **kwargs):
        return [forward_layer.begin_state(*args, **kwargs) for forward_layer in self.forward_layers], [backward_layer.begin_state(*args, **kwargs) for backward_layer in self.backward_layers]

    def forward(self, inputs, states_forward=None, states_backward=None):
        if not (states_forward and states_backward):
            states_forward, states_backward = self.begin_state(batch_size=inputs.shape[1])

        outputs_forward = []
        out_states_forward = []
        outputs_backward = []
        out_states_backward = []

        for j in range(self.num_layers):
            outputs_forward.append([])
            for i in range(inputs.shape[0]):
                output, states_forward[j] = self.forward_layers[j](inputs[i], states_forward[j])
                outputs_forward[j].append(output)
            out_states_forward.append(states_forward[j])

            outputs_backward.append([])
            for i in reversed(range(inputs.shape[0])):
                output, states_backward[j] = self.backward_layers[j](inputs[i], states_backward[j])
                outputs_backward[j].append(output)
            out_states_backward.append(states_backward[j])


        return outputs_forward, out_states_forward, outputs_backward, out_states_backward

lstm = ElmoLSTM('lstm', 3, 400, 100, 0.4, 0)
lstm.collect_params().initialize()
states_forward, states_backward = lstm.begin_state(batch_size=80)
outputs_forward, states_forward, outputs_backward, states_backward = lstm(nd.ones((35,80,400)), states_forward, states_backward)
