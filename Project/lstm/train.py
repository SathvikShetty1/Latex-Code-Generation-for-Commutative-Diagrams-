import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import dataprocessing as dataprocessing
from solver import *
from rnn import *
import util as util
import matplotlib.pyplot as plt

train_image_path = "../data/train_images/*.png"
arr_images, size = dataprocessing.get_pad_images(train_image_path)
train_images = dataprocessing.get_images_tensor(arr_images, size)

train_string_path = "../data/train_strings/*.tex"
textstrings = dataprocessing.get_pad_strings(train_string_path)

max_len = max([len(text) for text in textstrings])
print(max_len)
train_strings = dataprocessing.get_text_array(textstrings)
for i in range(len(textstrings)):
    textstrings[i] = ''.join([char for char in textstrings[i] if char != '<NULL>'])

dev_image_path = "../data/dev_images/*.png"
dev_arr_images, dev_size = dataprocessing.get_pad_images(train_image_path)
dev_images = dataprocessing.get_images_tensor(dev_arr_images, dev_size)

dev_string_path = "../data/dev_strings/*.tex"
devcodes = dataprocessing.get_pad_strings(dev_string_path)
dev_strings = dataprocessing.get_text_array(devcodes)
for i in range(len(devcodes)):
    devcodes[i] = ''.join([char for char in devcodes[i] if char != '<NULL>'])

char_to_idx = dataprocessing.create_char_to_idx()
idx_to_char = dataprocessing.create_idx_to_char()
data = util.create_data_dict(train_images, train_strings, textstrings, char_to_idx, idx_to_char, dev_images, dev_strings, devcodes)
device, gpu_ids = util.get_available_devices()
print("Got device ", device)

if device == "cuda":
    torch.set_default_tensor_type('torch.cuda.LongTensor')

save_dir = "./save/"

imageToSeq = LSTMImageToSeq(
          word_to_idx=data['word_to_idx'],
          input_dim=data['train_features'].shape[1],
          wordvec_dim=32,
        )

solver = Solver(imageToSeq, data, idx_to_word=data['idx_to_word'],
           num_epochs=100,
           batch_size=32,
           learning_rate=0.001,
           print_every=10,
           device=device,
           gpu_ids=gpu_ids,
           save_dir=save_dir,
           eval_steps=1000
         )

solver.train()

print("Dev set BLEU score is ", solver.evaluate())

