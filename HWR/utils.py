import numpy as np
import cv2
import HWR.loadData2_vgg as loadData
from pathlib import Path

HEIGHT = loadData.IMG_HEIGHT
WIDTH = loadData.IMG_WIDTH
output_max_len = loadData.OUTPUT_MAX_LEN
tokens = loadData.tokens
num_tokens = loadData.num_tokens
vocab_size = loadData.num_classes + num_tokens
index2letter = loadData.index2letter
FLIP = loadData.FLIP


def writePredict(result_file, input_images, predictions): # [batch_size, vocab_size] * max_output_len

    result_file.parent.mkdir(parents=True, exist_ok=True)
    predictions = predictions.data
    top_predictions = predictions.topk(1)[1].squeeze(2) # (15, 32)
    top_predictions = top_predictions.transpose(0, 1) # (32, 15)
    top_predictions = top_predictions.cpu().numpy()

    batch_count_n = []
    with open(result_file, 'a') as f:
        for n, seq in zip(input_images, top_predictions):
            f.write(n+' ')
            count_n = 0
            for i in seq:
                if i ==tokens['END_TOKEN']:
                    #f.write('<END>')
                    break
                else:
                    if i ==tokens['GO_TOKEN']:
                        f.write('<GO>')
                    elif i ==tokens['PAD_TOKEN']:
                        f.write('<PAD>')
                    else:
                        f.write(index2letter[i-num_tokens])
                    count_n += 1
            batch_count_n.append(count_n)
            f.write('\n')
    return batch_count_n
