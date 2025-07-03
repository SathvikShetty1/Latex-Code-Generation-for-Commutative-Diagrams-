import util as util
import torch
import numpy as np
import nltk
import sys
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

class Solver(object):
    """
    Takes in model and data. Model should be a nn.Module object.
      data has the components:
      - features: Array giving a minibatch of features for images, of shape (N, D)
      - captions: Array of captions for those images, of shape (N, T) where
        each element is in the range (0, V].
      - idx_to_word dictionary
      - word_to_idx dictionary
    """

    def __init__(self, model, data, idx_to_word, **kwargs):
        """
        Construct a new CaptioningSolver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - data: A dictionary of training and validation data

        Optional arguments:
        - batch_size: Size of minibatches used to compute loss and gradient during
          training.
        - num_epochs: The number of epochs to run for during training.
        - eval_steps: an evaluation on dev set will be carried out after eval_steps
          iterations.
        """
        self.model = model
        self.data = data

        # Unpack keyword arguments
        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)

        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)
        self.device = kwargs.pop("device", 'cuda')
        print(self.device)
        self.gpu_ids = kwargs.pop("gpu_ids", [])
        self.save_dir = kwargs.pop("save_dir", "./save/")
        self.eval_steps = kwargs.pop("eval_steps", 500)
        self.optim = torch.optim.Adam(
            self.model.parameters(), self.learning_rate)

        # Throw an error if there are extra keyword arguments
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self.optim = torch.optim.Adam(
            self.model.parameters(), self.learning_rate)

        self._reset()

        self.idx_to_word = idx_to_word
        self.model = torch.nn.DataParallel(model, self.gpu_ids)
        self.model = self.model.to(self.device)

    def _reset(self):
        """
        Set up some book-keeping variables for optimization.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.loss_history = []


    def _step(self, minibatch):
        """
        This is called by train().
        """
        captions, features = minibatch
        captions = captions.to(self.device)
        features = features.to(self.device)
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = (captions_out != self.model.module._null).long()

        logits = self.model(features, captions_in)

        loss = self.temporal_softmax_loss(
            logits, captions_out, mask)
        self.loss_history.append(loss.detach())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        """
        Run optimization to train the model.
        """
        tbx = SummaryWriter(self.save_dir)
        steps_till_eval = self.eval_steps
        num_train = self.data["train_captions"].shape[0]
    
        step = 0
        epoch = 1
        batches_strings = torch.split(
            self.data["train_captions"], self.batch_size)
        batches_images = torch.split(
            self.data["train_features"], self.batch_size)
        batches = list(zip(batches_strings, batches_images))

        while epoch <= self.num_epochs:
            with torch.enable_grad(), \
                    tqdm(total=num_train) as progress_bar:
                for batch in batches:
                    self._step(batch)
                    step += self.batch_size
                    
                    progress_bar.update(self.batch_size)
                    progress_bar.set_postfix(epoch=epoch,
                                             CELoss=self.loss_history[-1])
                    tbx.add_scalar('train/CELoss', self.loss_history[-1], step)
                    tbx.add_scalar('train/LR',
                               self.optim.param_groups[0]['lr'],
                               step)
                    
                    steps_till_eval -= self.batch_size
                    if steps_till_eval <= 0:
                        steps_till_eval = self.eval_steps

                        # Evaluate
                        tqdm.write(f'Evaluating at step {step}...')
                        dev_bleu = self.evaluate()
                        tbx.add_scalar('dev/bleu', dev_bleu, step)
            epoch += 1
    
    def evaluate(self):
        
        sample_codes = self.model.module.sample(self.data['dev_features'][:10].to(self.device), max_length=250)
        # Using only 10 examples of dev set to speed up evaluation
        # Delete [:10] to get true performance
        sample_codes = util.decode_codes(sample_codes, self.data['idx_to_word'])
        average_bleu = 0
        for i in range(len(sample_codes)):
            sample_caption = sample_codes[i]
            print(sample_caption)
            print('--------------')
            true_code = self.data['dev_codes'][i]
            print(true_code)
            print('--------------')
            average_bleu += nltk.translate.bleu_score.sentence_bleu([list(true_code)], list(sample_caption))
        average_bleu /= len(sample_codes)

        return average_bleu

    def temporal_softmax_loss(self, x, y, mask):
        """
        Inputs:
        - x: Input scores, of shape (N, T, V)
        - y: Ground-truth indices, of shape (N, T) where each element is in the range
             0 <= y[i, t] < V
        - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
          the scores at x[i, t] should contribute to the loss.

        Returns loss
        """

        N, T, V = x.shape

        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(
            x_flat,  y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)

        return loss

