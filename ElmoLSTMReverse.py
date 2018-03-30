
from mxnet import gluon, nd
from mxnet.gluon import rnn
from mxnet.gluon.model_zoo.text.base import get_rnn_layer, get_rnn_cell

class ElmoLSTMReverse(gluon.Block):
    def __init__(self, mode, num_layers, input_size, hidden_size, dropout, weight_dropout, bidirectional=True):
        super(ElmoLSTMReverse, self).__init__()

        self.num_layers = num_layers

        with self.name_scope():
            for i in range(num_layers):
                if i == 0:
                    forward_lstm = get_rnn_layer(mode, 1, input_size, hidden_size, dropout, weight_dropout)
                    backward_lstm = get_rnn_layer(mode, 1, input_size, hidden_size, dropout, weight_dropout)
                else:
                    forward_lstm = get_rnn_layer(mode, 1, hidden_size, hidden_size, dropout, weight_dropout)
                    backward_lstm = get_rnn_layer(mode, 1, hidden_size, hidden_size, dropout, weight_dropout)

                setattr(self, 'forward_lstm_{}'.format(i), forward_lstm)
                setattr(self, 'backward_lstm_{}'.format(i), backward_lstm)

    def begin_state(self, *args, **kwargs):
        return [getattr(self, 'forward_lstm_{}'.format(i)).begin_state(*args, **kwargs) for i in range(self.num_layers)], [getattr(self, 'backward_lstm_{}'.format(i)).begin_state(*args, **kwargs) for i in range(self.num_layers)]

    def forward(self, inputs, states_forward=None, states_backward=None):
        inputs_reversed = nd.reverse(inputs, axis=2)

        if not (states_forward and states_backward):
            states_forward, states_backward = self.begin_state(batch_size=inputs.shape[1])

        outputs_forward = []
        outputs_backward = []

        for i in range(self.num_layers):
            if i == 0:
                output, states_forward[i] = getattr(self, 'forward_lstm_{}'.format(i))(inputs, states_forward[i])
                outputs_forward.append(output)

                output, states_backward[i] = getattr(self, 'backward_lstm_{}'.format(i))(inputs_reversed, states_backward[i])
                outputs_backward.append(output)
            else:
                output, states_forward[i] = getattr(self, 'forward_lstm_{}'.format(i))(outputs_forward[i-1], states_forward[i])
                outputs_forward.append(output)

                output, states_backward[i] = getattr(self, 'backward_lstm_{}'.format(i))(outputs_backward[i-1], states_backward[i])
                outputs_backward.append(output)
        return outputs_forward, states_forward, outputs_backward, states_backward

lstm = ElmoLSTMReverse('lstm', 3, 400, 1150, 0.4, 0.5, True)
lstm.collect_params().initialize()
states_forward, states_backward = lstm.begin_state(batch_size=80)
outputs_forward, states_forward, outputs_backward, states_backward = lstm(nd.uniform(-1, 1, (35,80,400)), states_forward, states_backward)

