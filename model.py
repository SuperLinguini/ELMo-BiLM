import mxnet as mx
import h5py
import numpy as np
from mxnet import gluon, autograd, init, nd
from mxnet.gluon import nn, rnn
from gluonnlp.model.utils import _get_rnn_cell
from elmo_char_encoder import ElmoCharacterEncoder

from LSTMPCell import LSTMPCell

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

def _get_rnn_cell(mode, num_layers, input_size, hidden_size, dropout):
    """create rnn cell given specs"""
    rnn_cell = rnn.SequentialRNNCell()
    with rnn_cell.name_scope():
        for i in range(num_layers):
            if mode == 'rnn_relu':
                cell = rnn.RNNCell(hidden_size, 'relu', input_size=input_size)
            elif mode == 'rnn_tanh':
                cell = rnn.RNNCell(hidden_size, 'tanh', input_size=input_size)
            elif mode == 'lstm':
                cell = rnn.LSTMCell(hidden_size, input_size=input_size)
            elif mode == 'gru':
                cell = rnn.GRUCell(hidden_size, input_size=input_size)

            rnn_cell.add(cell)
            if dropout != 0:
                rnn_cell.add(rnn.DropoutCell(dropout))

    return rnn_cell

class ElmoLSTM(gluon.Block):
    def __init__(self, mode, num_layers, input_size, hidden_size, cell_size, dropout, skip_connection, char_embedding, weight_file=None, bidirectional=True):
        super(ElmoLSTM, self).__init__()

        self.num_layers = num_layers
        self.char_embedding = char_embedding
        self.weight_file = weight_file
        self.skip_connection = skip_connection

        lstm_input_size = input_size

        with self.name_scope():
            for layer_index in range(num_layers):
                # forward_layer = LSTMPCell(hidden_size, cell_size, input_size=lstm_input_size, memory_cell_clip_value=3, state_projection_clip_value=3)
                # backward_layer = LSTMPCell(hidden_size, cell_size, input_size=lstm_input_size, memory_cell_clip_value=3, state_projection_clip_value=3)

                forward_layer = _get_rnn_cell(mode, 1, lstm_input_size, hidden_size, dropout)#, cell_size=cell_size)
                backward_layer = _get_rnn_cell(mode, 1, lstm_input_size, hidden_size, dropout)#, cell_size=cell_size)

                setattr(self, 'forward_layer_{}'.format(layer_index), forward_layer)
                setattr(self, 'backward_layer_{}'.format(layer_index), backward_layer)

                lstm_input_size = hidden_size

    def begin_state(self, *args, **kwargs):
        return [getattr(self, 'forward_layer_{}'.format(layer_index)).begin_state(*args, **kwargs) for layer_index in range(self.num_layers)],\
               [getattr(self, 'backward_layer_{}'.format(layer_index)).begin_state(*args, **kwargs) for layer_index in range(self.num_layers)]

    def forward(self, inputs, states_forward=None, states_backward=None):
        seq_len = inputs.shape[0] if self.char_embedding else inputs[0].shape[0]

        if not (states_forward and states_backward):
            states_forward, states_backward = self.begin_state(batch_size=inputs.shape[1] if self.char_embedding else inputs[0].shape[1])

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
                    if self.skip_connection:
                        output = output + outputs_forward[j-1][i]
                outputs_forward[j].append(output)
            out_states_forward.append(states_forward[j])

            outputs_backward.append([None] * seq_len)
            for i in reversed(range(seq_len)):
                if j == 0:
                    output, states_backward[j] = getattr(self, 'backward_layer_{}'.format(j))(inputs[i] if self.char_embedding else inputs[1][i], states_backward[j])
                else:
                    output, states_backward[j] = getattr(self, 'backward_layer_{}'.format(j))(outputs_backward[j-1][i], states_backward[j])
                    if self.skip_connection:
                        output = output + outputs_backward[j-1][i]
                outputs_backward[j][i] = output
            out_states_backward.append(states_backward[j])

        for i in range(self.num_layers):
            outputs_forward[i] = mx.nd.stack(*outputs_forward[i])
            outputs_backward[i] = mx.nd.stack(*outputs_backward[i])

        return outputs_forward, out_states_forward, outputs_backward, out_states_backward

    def load_weights(self):
        """
        Load the pre-trained weights from the file.
        """
        # requires_grad = self.requires_grad

        with h5py.File(self.weight_file, 'r') as fin:
            for layer_index in range(self.num_layers):
                for i in range(2):
                    # lstm is an instance of LSTMPCell
                    lstm = getattr(self, 'forward_layer_{}'.format(layer_index) if i == 0 else 'backward_layer_{}'.format(layer_index))
                    cell_size = lstm._cell_size

                    dataset = fin['RNN_%s' % i]['RNN']['MultiRNNCell']['Cell%s' % layer_index]['LSTMCell']

                    # tensorflow packs together both W and U matrices into one matrix,
                    # but mxnet maintains individual matrices.  In addition, tensorflow
                    # packs the gates as input, memory, forget, output but mxnet
                    # uses input, forget, memory, output.  So we need to modify the weights.
                    tf_weights = np.transpose(dataset['W_0'][...])
                    torch_weights = tf_weights.copy()

                    # split the W from U matrices
                    input_size = lstm._input_size
                    input_weights = torch_weights[:, :input_size]
                    recurrent_weights = torch_weights[:, input_size:]
                    tf_input_weights = tf_weights[:, :input_size]
                    tf_recurrent_weights = tf_weights[:, input_size:]

                    # handle the different gate order convention
                    for torch_w, tf_w in [[input_weights, tf_input_weights],
                                          [recurrent_weights, tf_recurrent_weights]]:
                        torch_w[(1 * cell_size):(2 * cell_size), :] = tf_w[(2 * cell_size):(3 * cell_size), :]
                        torch_w[(2 * cell_size):(3 * cell_size), :] = tf_w[(1 * cell_size):(2 * cell_size), :]

                    lstm.i2h_weight.set_data(input_weights)
                    lstm.h2h_weight.set_data(recurrent_weights)
                    # lstm.input_linearity.weight.requires_grad = requires_grad
                    # lstm.state_linearity.weight.requires_grad = requires_grad

                    # the bias weights
                    tf_bias = dataset['B'][...]
                    # tensorflow adds 1.0 to forget gate bias instead of modifying the
                    # parameters...
                    tf_bias[(2 * cell_size):(3 * cell_size)] += 1
                    torch_bias = tf_bias.copy()
                    torch_bias[(1 * cell_size):(2 * cell_size)
                              ] = tf_bias[(2 * cell_size):(3 * cell_size)]
                    torch_bias[(2 * cell_size):(3 * cell_size)
                              ] = tf_bias[(1 * cell_size):(2 * cell_size)]
                    lstm.h2h_bias.set_data(torch_bias)
                    # lstm.state_linearity.bias.requires_grad = requires_grad

                    # the projection weights
                    proj_weights = np.transpose(dataset['W_P_0'][...])
                    lstm.h2proj_weight.set_data(proj_weights)
                    # lstm.state_projection.weight.requires_grad = requires_grad

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
    def __init__(self, mode, vocab_size, embed_size, hidden_size, cell_size,
                 num_layers, tie_weights=False, dropout=0.5, skip_connection=True, char_embedding=False, options=elmo_options, weight_file=None, **kwargs):
        if tie_weights:
            assert embed_size == hidden_size, "Embedding dimension must be equal to " \
                                              "hidden dimension in order to tie weights. " \
                                              "Got: emb: {}, hid: {}.".format(embed_size,
                                                                              hidden_size)
        super(ElmoBiLM, self).__init__(**kwargs)
        self._mode = mode
        self._embed_size = options['lstm']['projection_dim'] if char_embedding else embed_size
        self._hidden_size = hidden_size
        self._cell_size = cell_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._skip_connection = skip_connection
        self._tie_weights = tie_weights
        self._vocab_size = vocab_size
        self.char_embedding = char_embedding
        self.weight_file = weight_file
        self.options = options

        with self.name_scope():
            self.embedding = self._get_embedding()
            self.encoder = self._get_encoder()
            self.decoder = self._get_decoder()

    def _get_embedding(self):
        if self.char_embedding:
            return ElmoCharacterEncoder(self.options, self.weight_file)
        else:
            embedding = nn.HybridSequential()
            with embedding.name_scope():
                embedding.add(nn.Embedding(self._vocab_size, self._embed_size,
                                           weight_initializer=init.Uniform(0.1)))
                if self._dropout:
                    embedding.add(nn.Dropout(self._dropout))
            return embedding

    def _get_encoder(self):
        return ElmoLSTM(mode=self._mode, num_layers=self._num_layers, input_size=self._embed_size,
                              hidden_size=self._hidden_size, cell_size=self._cell_size, dropout=self._dropout,
                              skip_connection=self._skip_connection, char_embedding=self.char_embedding, weight_file=self.weight_file)

    def _get_decoder(self):
        output = nn.HybridSequential()
        with output.name_scope():
            if self._tie_weights:
                output.add(nn.Dense(self._vocab_size, flatten=False,
                                    params=self.embedding[0].params))
            else:
                output.add(nn.Dense(self._vocab_size, flatten=False, in_units=self._hidden_size))
        return output

    def set_highway_bias(self):
        self.embedding.set_highway_bias()

    def load_char_embedding_weights(self):
        self.embedding.load_weights()

    def load_word_embedding_weights(self):
        with h5py.File(self.weight_file, 'r') as fin:
            embedding_weights = fin['embedding'][...]
            self.embedding._children['0'].weight.set_data(nd.array(embedding_weights))

    def load_decoder(self):
        with h5py.File(self.weight_file, 'r') as fin:
            self.decoder._children['0'].weight.set_data(fin['softmax']['W'][...])
            self.decoder._children['0'].bias.set_data(fin['softmax']['b'][...])

    def load_lstm_weights(self):
        self.encoder.load_weights()

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


        if self._dropout:
            encoded_forward = nd.Dropout(outputs_forward[-1], p=self._dropout)
            encoded_backward = nd.Dropout(outputs_backward[-1], p=self._dropout)
        else:
            encoded_forward = outputs_forward[-1]
            encoded_backward = outputs_backward[-1]

        forward_out = self.decoder(encoded_forward)
        backward_out = self.decoder(encoded_backward)
        return (forward_out, backward_out), (states_forward, states_backward)
