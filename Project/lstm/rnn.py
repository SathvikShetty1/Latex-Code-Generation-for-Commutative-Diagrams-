import numpy as np
import torch
import torch.nn as nn


class LSTMImageToSeq(nn.Module):
    """
    A CaptioningRNN produces captions from image features using a recurrent
    neural network.

    The RNN receives input vectors of size D, has a vocab size of V, works on
    sequences of length T, has an RNN hidden dimension of H, uses word vectors
    of dimension W, and operates on minibatches of size N.

    Note that we don't use any regularization for the CaptioningRNN.
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128
    ):
        """
        Construct a new CaptioningRNN instance.

        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries,
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension D of input image feature vectors.
        - wordvec_dim: Dimension W of word vectors.
        - hidden_dim: Dimension H for the hidden state of the RNN.
        """

        super().__init__()

        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)
        self.hidden_dim = hidden_dim

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # Initialize word vectors
        # self.params["W_embed"] = np.random.randn(vocab_size, wordvec_dim)
        # self.params["W_embed"] /= 100

        self.visual_projection = nn.Linear(input_dim, hidden_dim)
        self.embedding = nn.Embedding(
            vocab_size, wordvec_dim, padding_idx=self._null)

        # Initialize CNN -> hidden state projection parameters
        # self.params["W_proj"] = np.random.randn(input_dim, hidden_dim)
        # self.params["W_proj"] /= np.sqrt(input_dim)
        # self.params["b_proj"] = np.zeros(hidden_dim)

        # Initialize parameters for the RNN
        # dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        # self.params["Wx"] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        # self.params["Wx"] /= np.sqrt(wordvec_dim)
        # self.params["Wh"] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        # self.params["Wh"] /= np.sqrt(hidden_dim)
        # self.params["b"] = np.zeros(dim_mul * hidden_dim)

        self.lstm = nn.LSTM(wordvec_dim, hidden_dim, batch_first=True)

        # Initialize output to vocab weights
        # self.params["W_vocab"] = np.random.randn(hidden_dim, vocab_size)
        # self.params["W_vocab"] /= np.sqrt(hidden_dim)
        # self.params["b_vocab"] = np.zeros(vocab_size)

        self.output = nn.Linear(hidden_dim, vocab_size)

        # Cast parameters to correct dtype
        # for k, v in self.params.items():
        #     self.params[k] = v.astype(self.dtype)

    def forward(self, features, captions):
        """
        Compute training-time loss for the RNN. We input image features and
        ground-truth captions for those images, and use an LSTM to compute
        loss and gradients on all parameters.

        Inputs:
        - features: Input image features, of shape (N, D)
        - captions: Input captions; an integer array of shape (N, T) where
          each element is in the range 0 <= y[i, t] < V

        Returns the score of shape (N, T, V)
        """

        N, _ = features.shape

        # h_init, cache_affine = affine_forward(features, W_proj, b_proj)
        h_init = self.visual_projection(features).unsqueeze(0)
        c_init = torch.zeros((1, N, self.hidden_dim)).cuda()
        # print("h device is ", h_init.device)
        # print("c device is ", c_init.device)

        # embed, cache_embed = word_embedding_forward(captions_in, W_embed)
        embed = self.embedding(captions)
        # print(embed.device)

        h, (h_n, c_n) = self.lstm(embed, (h_init, c_init))
        ah = self.output(h)

        # Not computing loss here
        # loss = temporal_softmax_loss(ah, captions_out, mask)

        return ah

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.

        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length

        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            # features = torch.Tensor(features)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * torch.ones((N, max_length)).int().cuda()

            # Create a partial caption, with only the start token.
            partial_caption = self._start * torch.ones(N).int().cuda()
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions
