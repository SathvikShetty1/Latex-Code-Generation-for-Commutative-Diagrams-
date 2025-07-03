# util functions from https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
import torch
import string
import glob
import os
import numpy as np
import cv2
import math

import mytokenize

all_letters = " " + string.ascii_letters + "1234567890.,;:~'\"!@#$%^&*()[]{}\_-+=<>?/|`\n"
n_letters = len(all_letters)

def letterToIndex(letter):
    if letter in mytokenize.all_tokens:
        return mytokenize.all_tokens.index(letter)
    else:
        return -1

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

def create_char_to_idx():
    char_to_idx = {
        "<NULL>" : 0,
        "<START>" : 1,
        "<END>" : 2
    }
    for chr in mytokenize.all_tokens:
        char_to_idx[chr] = letterToIndex(chr) + 3
    return char_to_idx

def create_idx_to_char():
    return ["<NULL>", "<START>", "<END>"] + mytokenize.all_tokens

def get_pad_images(path, vgg=False):
  '''
  Read all images in the directory specified by path, and pad
  them to have the same shape. Return a numpy array of 
  all padded image.
  '''
  filelist = sorted(glob.glob(path))
  if vgg:
      imagelist = [np.array(cv2.imread(fname)) for fname in filelist]
  else:
      imagelist = [np.array(cv2.imread(fname, 0)) for fname in filelist]
  # print(np.amax(imagelist[0]))
  xdims, ydims = np.zeros(len(imagelist)), np.zeros(len(imagelist))

  for i in range(len(imagelist)):
    xdims[i], ydims[i] = imagelist[i].shape[0], imagelist[i].shape[1]
  x_max = int(np.max(xdims))
  y_max = int(np.max(ydims))

  for i in range(len(imagelist)):
    A = imagelist[i]
    x_len, y_len = A.shape[0], A.shape[1]
    pad_x = (x_max - x_len) // 2
    pad_y = (y_max - y_len) // 2
    if vgg:
        imagelist[i] = np.pad(A, ((pad_x, x_max - x_len - pad_x), (pad_y, y_max - y_len - pad_y), (0, 0)), 'constant', constant_values=255)
    else:
        imagelist[i] = np.pad(A, ((pad_x, x_max - x_len - pad_x), (pad_y, y_max - y_len - pad_y)), 'constant', constant_values=255)
  arr_images = np.array(imagelist)
  sample_size = len(imagelist)
  return arr_images, sample_size

def get_images_tensor(arr_images, size, flatten=True):
    if flatten:
        tensor_images = arr_images.reshape(size, -1)
    else:
        tensor_images = arr_images
    tensor_images = torch.from_numpy(tensor_images)
    tensor_images = tensor_images.float()
    tensor_images = 1 - tensor_images / 255
    return tensor_images

def get_pad_strings(stringpath, tokenized=False):
    stringlist = sorted(glob.glob(stringpath))
    textstrings = []

    for textfile in stringlist:
        with open(textfile, 'r') as text:
            tex = text.read().replace('\t', ' ')
            if tokenized:
                textstrings.append(["<START>"]+ mytokenize.get_string_tokens(tex)+["<END>"])
            else:
                textstrings.append(["<START>"]+list(tex)+["<END>"])

    lengths = np.zeros(len(textstrings))
    for i in range(len(textstrings)):
  
        lengths[i] = len(textstrings[i])
    max_len = int(np.max(lengths))
    for i in range(len(textstrings)):
        pad = max_len - len(textstrings[i])
        textstrings[i] = textstrings[i] + pad * ["<NULL>"]
    return textstrings

def get_text_array(textstrings):
    train_codes = torch.zeros((len(textstrings), len(textstrings[0])))
    for i in range(train_codes.shape[0]):
        for j in range(train_codes.shape[1]):
            if textstrings[i][j] == "<NULL>":
                idx = 0
            elif textstrings[i][j] == "<START>":
                idx = 1
            elif textstrings[i][j] == "<END>":
                idx = 2
            else:
                idx = letterToIndex(textstrings[i][j]) + 3
                if idx == 2:
                    print(''.join(textstrings[i]))
            train_codes[i][j] = idx
    train_codes = train_codes.long()
    return train_codes
