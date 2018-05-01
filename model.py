import mxnet as mx
from mxnet import gluon, autograd, init, nd
from mxnet.gluon import nn
from gluonnlp.model.utils import _get_rnn_cell
from elmo_char_encoder import ElmoCharacterEncoder

elmo_options = {
  "lstm": {
    "use_skip_connections": True,
    "projection_dim": 512,
    "cell_clip": 3,
    "proj_clip": 3,
    "dim": 4096,
    "n_layers": 2
  },
  "char_cnn": {
    "activation": "relu",
    "filters": [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]],
    "n_highway": 2,
    "embedding": {
      "dim": 16
    },
    "n_characters": 262,
    "max_characters_per_token": 50
  }
}


class ElmoLSTM(gluon.Block):
    def __init__(self, mode, num_layers, input_size, hidden_size, dropout, weight_dropout, char_embedding, bidirectional=True):
        super(ElmoLSTM, self).__init__()

        self.num_layers = num_layers
        self.char_embedding = char_embedding

        lstm_input_size = input_size

        with self.name_scope():
            for layer_index in range(num_layers):
                forward_layer = _get_rnn_cell(mode, 1, lstm_input_size, hidden_size, dropout, weight_dropout, 0, 0, 0)
                backward_layer = _get_rnn_cell(mode, 1, lstm_input_size, hidden_size, dropout, weight_dropout, 0, 0, 0)

                setattr(self, 'forward_layer_{}'.format(layer_index), forward_layer)
                setattr(self, 'backward_layer_{}'.format(layer_index), backward_layer)

                lstm_input_size = hidden_size

    def begin_state(self, *args, **kwargs):
        return [getattr(self, 'forward_layer_{}'.format(layer_index)).begin_state(*args, **kwargs) for layer_index in range(self.num_layers)],\
               [getattr(self, 'backward_layer_{}'.format(layer_index)).begin_state(*args, **kwargs) for layer_index in range(self.num_layers)]

    def forward(self, inputs, states_forward=None, states_backward=None):
        seq_len = inputs.shape[0] if self.char_embedding else inputs[0].shape[0]

        if not (states_forward and states_backward):
            states_forward, states_backward = self.begin_state(batch_size=inputs.shape[1])

        outputs_forward = []
        out_states_forward = []
        outputs_backward = []
        out_states_backward = []

        for j in range(self.num_layers):
            outputs_forward.append([])
            for i in range(seq_len):
                if j == 0:
                    output, states_forward[j] = getattr(self, 'forward_layer_{}'.format(j))(inputs[i] if self.char_embedding else inputs[0][i], states_forward[j])
                else:
                    output, states_forward[j] = getattr(self, 'forward_layer_{}'.format(j))(outputs_forward[j-1][i], states_forward[j])
                outputs_forward[j].append(output)
            out_states_forward.append(states_forward[j])

            outputs_backward.append([None] * seq_len)
            for i in reversed(range(seq_len)):
                if j == 0:
                    output, states_backward[j] = getattr(self, 'backward_layer_{}'.format(j))(inputs[i] if self.char_embedding else inputs[1][i], states_backward[j])
                else:
                    output, states_backward[j] = getattr(self, 'backward_layer_{}'.format(j))(outputs_backward[j-1][i], states_backward[j])
                outputs_backward[j][i] = output
            out_states_backward.append(states_backward[j])

        for i in range(self.num_layers):
            outputs_forward[i] = mx.nd.stack(*outputs_forward[i])
            outputs_backward[i] = mx.nd.stack(*outputs_backward[i])

        return outputs_forward, out_states_forward, outputs_backward, out_states_backward

class ElmoBiLM(gluon.Block):
    """Standard RNN language model.

    Parameters
    ----------
    mode : str
        The type of RNN to use. Options are 'lstm', 'gru', 'rnn_tanh', 'rnn_relu'.
    vocab_size : int
        Size of the input vocabulary.
    embed_size : int
        Dimension of embedding vectors.
    hidden_size : int
        Number of hidden units for RNN.
    num_layers : int
        Number of RNN layers.
    dropout : float
        Dropout rate to use for encoder output.
    tie_weights : bool, default False
        Whether to tie the weight matrices of output dense layer and input embedding layer.
    """
    def __init__(self, mode, vocab_size, embed_size, hidden_size,
                 num_layers, tie_weights=False, dropout=0.5, char_embedding=False, options=elmo_options, **kwargs):
        if tie_weights:
            assert embed_size == hidden_size, "Embedding dimension must be equal to " \
                                              "hidden dimension in order to tie weights. " \
                                              "Got: emb: {}, hid: {}.".format(embed_size,
                                                                              hidden_size)
        super(ElmoBiLM, self).__init__(**kwargs)
        self._mode = mode
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._tie_weights = tie_weights
        self._vocab_size = vocab_size
        self.char_embedding = char_embedding
        self.options = options

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        if self.char_embedding:
            return ElmoCharacterEncoder(self.options)
        else:
            embedding = nn.HybridSequential()
            with embedding.name_scope():
                embedding.add(nn.Embedding(self._vocab_size, self._embed_size,
                                           weight_initializer=init.Uniform(0.1)))
                if self._dropout:
                    embedding.add(nn.Dropout(self._dropout))
            return embedding

    def _get_encoder(self):
        return ElmoLSTM(self._mode, self._num_layers, self._embed_size,
                              self._hidden_size // 2, self._dropout, 0, self.char_embedding)

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            output.add(nn.Dropout(self._dropout))
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False))
        return output

    def set_highway_bias(self):
        self.embedding.set_highway_bias()

    def begin_state(self, *args, **kwargs):
        return self.encoder.begin_state(*args, **kwargs)

    def forward(self, inputs, states_forward=None, states_backward=None): # pylint: disable=arguments-differ
        if self.char_embedding:
            embedded_inputs = self.embedding(inputs)
        else:
            embedded_inputs = self.embedding(inputs[0]), self.embedding(inputs[1])

        if not (states_forward and states_backward):
            states_forward, states_backward = self.begin_state(batch_size=inputs.shape[1])
        outputs_forward, out_states_forward, outputs_backward, out_states_backward = self.encoder(embedded_inputs, states_forward, states_backward)

        # out2 = mx.nd.empty((len(outputs_forward[-1]), outputs_forward[-1][0].shape[0], self._vocab_size))
        # out = []
        # for i in range(len(outputs_forward[-1])):
        #     # out[i, :, :] = self.decoder(mx.nd.concat(outputs_forward[-1][0], outputs_backward[-1][0], dim=1))
        #     out.append(self.decoder(mx.nd.concat(outputs_forward[-1][0], outputs_backward[-1][0], dim=1)))
        # out2 = mx.nd.stack(*out)
        forward_out = self.decoder(outputs_forward[-1])
        backward_out = self.decoder(outputs_backward[-1])
        return (forward_out, backward_out), (states_forward, states_backward)
