import numpy as np
import matplotlib.pyplot as plt
import torch

def create_minibatch(data, batch_size, split="train"):
    split_size = data["%s_captions" % split].shape[0]
    mask = np.random.choice(split_size, batch_size)
    captions = data["%s_captions" % split][mask]
    # image_idxs = data["%s_image_idxs" % split][mask]
    image_features = data["%s_features" % split][mask]
    
    return captions, image_features

def create_data_dict(train_features, train_strings, train_codes, char_to_idx, idx_to_char, dev_features, dev_strings, dev_codes):
    data = {
    'train_features' : train_features,
    'word_to_idx' : char_to_idx,
    'idx_to_word' : idx_to_char,
    'train_captions' : train_strings,
    'train_codes' : train_codes,
    'dev_features' : dev_features,
    'dev_captions' : dev_strings,
    'dev_codes' : dev_codes
    }
    return data

def decode_codes(codes, idx_to_char):
    singleton = False
    if codes.ndim == 1:
        singleton = True
        codes = codes[None]
    decoded = []
    N, T = codes.shape
    for i in range(N):
        chars = []
        for t in range(T):
            char = idx_to_char[codes[i, t]]
            
            chars.append(char)
            if char == "<END>":
                break
        decoded.append("".join(chars))
    if singleton:
        decoded = decoded[0]
    return decoded

def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids
