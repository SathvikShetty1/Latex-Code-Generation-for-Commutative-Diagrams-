# === Modified solver_transformer.py ===

import util as util
import torch
import numpy as np
import nltk
import sys
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

class SolverTransformer(object):
    def __init__(self, model, data, idx_to_word, **kwargs):
        self.model = model
        self.data = data

        self.learning_rate = kwargs.pop("learning_rate", 0.001)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.print_every = kwargs.pop("print_every", 10)
        self.device = kwargs.pop("device", 'cuda')
        print(self.device)
        self.gpu_ids = kwargs.pop("gpu_ids", [])  # Not used anymore
        self.save_dir = kwargs.pop("save_dir", "./save/")
        self.eval_steps = kwargs.pop("eval_steps", 500)

        self.optim = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=3e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=80, gamma=0.5)

        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        self._reset()
        self.idx_to_word = idx_to_word
        self.model = self.model.to(self.device)
        self.scaler = GradScaler()  # For mixed-precision

    def _reset(self):
        self.epoch = 0
        self.loss_history = []

    def _step(self, minibatch):
        captions, features = minibatch
        captions = captions.to(self.device)
        features = features.to(self.device)
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        mask = (captions_out != self.model._null).long()

        with autocast():
            logits = self.model(features, captions_in)
            loss = self.transformer_temporal_softmax_loss(logits, captions_out, mask)

        self.loss_history.append(loss.detach())
        self.optim.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optim)
        self.scaler.update()

    def train(self):
        tbx = SummaryWriter(self.save_dir)
        steps_till_eval = self.eval_steps
        num_train = self.data["train_captions"].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch
        step = 0
        epoch = 1

        batches_strings = torch.split(self.data["train_captions"], self.batch_size)
        batches_images = torch.split(self.data["train_features"], self.batch_size)
        batches = list(zip(batches_strings, batches_images))

        while epoch <= self.num_epochs:
            with torch.enable_grad(), tqdm(total=num_train) as progress_bar:
                for batch in batches:
                    self.model.train()
                    self._step(batch)
                    step += self.batch_size
                    progress_bar.update(self.batch_size)
                    progress_bar.set_postfix(epoch=epoch, CELoss=self.loss_history[-1])
                    tbx.add_scalar('train/CELoss', self.loss_history[-1], step)
                    tbx.add_scalar('train/LR', self.optim.param_groups[0]['lr'], step)

                    steps_till_eval -= self.batch_size
                    if steps_till_eval <= 0:
                        steps_till_eval = self.eval_steps
                        tqdm.write(f'Evaluating at step {step}...')
                        dev_bleu, train_bleu = self.evaluate()
                        tbx.add_scalar('dev/bleu', dev_bleu, step)
                        tbx.add_scalar('train/bleu', train_bleu, step)
            epoch += 1
            self.scheduler.step()

    def evaluate(self):
        self.model.eval()
        dev_features = self.data['dev_features'][50:].to(self.device)
        sample_codes = self.model.sample(dev_features, max_length=300)
        sample_codes = util.decode_codes(sample_codes, self.data['idx_to_word'])
        average_bleu = 0
        for i in range(len(sample_codes)):
            sample_caption = sample_codes[i]
            true_code = self.data['dev_codes'][50+i]
            average_bleu += nltk.translate.bleu_score.sentence_bleu([list(true_code)], list(sample_caption))
        average_bleu /= len(sample_codes)

        train_codes = self.model.sample(self.data['train_features'][200:230].to(self.device), max_length=300)
        train_codes = util.decode_codes(train_codes, self.data['idx_to_word'])
        average_train_bleu = 0
        for i in range(len(train_codes)):
            train_caption = train_codes[i]
            true_code = self.data['train_codes'][200+i]
            average_train_bleu += nltk.translate.bleu_score.sentence_bleu([list(true_code)], list(train_caption))
        average_train_bleu /= len(train_codes)
        return average_bleu, average_train_bleu

    def transformer_temporal_softmax_loss(self, x, y, mask):
        N, T, V = x.shape
        x_flat = x.reshape(N * T, V)
        y_flat = y.reshape(N * T)
        mask_flat = mask.reshape(N * T)

        loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
        loss = torch.mul(loss, mask_flat)
        loss = torch.mean(loss)
        return loss
