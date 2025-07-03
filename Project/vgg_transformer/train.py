import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import dataprocessing as dataprocessing
from solver_transformer import *
from transformer import *
import util as util
import matplotlib.pyplot as plt
import cv2

train_image_path = "../data/train_images/*.png"
arr_images, size = dataprocessing.get_pad_images(train_image_path, True)
train_images = dataprocessing.get_images_tensor(arr_images, size, flatten=False)
#print(train_images.shape)
train_string_path = "../data/train_strings/*.tex"
textstrings = dataprocessing.get_pad_strings(train_string_path)
#print(max_len)
max_len = max([len(text) for text in textstrings])
print(max_len)
train_strings = dataprocessing.get_text_array(textstrings)
for i in range(len(textstrings)):
    textstrings[i] = ''.join([char for char in textstrings[i] if char != '<NULL>'])

dev_image_path = "../data/dev_images/*.png"
dev_arr_images, dev_size = dataprocessing.get_pad_images(dev_image_path, True)
dev_images = dataprocessing.get_images_tensor(dev_arr_images, dev_size, flatten=False)

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

# input_dim = data['train_features'].shape[1] * data['train_features'].shape[2]
input_dim = 1000
transformer = ImagetoSeqTransformer(
          word_to_idx=data['word_to_idx'],
          input_dim=input_dim,
          wordvec_dim=52,
          num_heads=4,
          num_layers=2,
          max_length=max_len,
        )

transformer_solver = SolverTransformer(transformer, data, idx_to_word=data['idx_to_word'],
           num_epochs=400,
           batch_size=64,
           learning_rate=0.001,
           print_every=10,
           device=device,
           gpu_ids=gpu_ids,
           save_dir=save_dir,
           eval_steps=2000
         )

transformer_solver.train()

print("Dev set BLEU score is ", transformer_solver.evaluate())


def show_samples():
    sample_captions = transformer.sample(train_images[:3], max_length=420)
    sample_captions = util.decode_codes(sample_captions, data['idx_to_word'])
    i = 5
    for sample_caption in sample_captions:
        
        img = cv2.imread('../data/smalltrain/diagram%d.png' % i, 0)
        i += 1
        
        plt.imshow(img)            
  
        plt.axis('off')
        plt.show()
        print(sample_caption)

