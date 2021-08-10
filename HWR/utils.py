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
WORD_LEVEL = loadData.WORD_LEVEL

load_data_func = loadData.loadData

def visualizeAttn(img, first_img_real_len, attn, epoch, count_n, name):

    folder_name = 'imgs'
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    img = img[:, :first_img_real_len]
    img = img.cpu().numpy()
    img -= img.min()
    img *= 255./img.max()
    img = img.astype(np.uint8)
    weights = [img] # (80, 460)
    #for m in attn[:count_n+1]: # also show the last <EOS>
    for m in attn[:count_n]:
        mask_img = np.vstack([m]*10) # (10, 55)
        mask_img *= 255./mask_img.max()
        mask_img = mask_img.astype(np.uint8)
        mask_img = cv2.resize(mask_img, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        weights.append(mask_img)
    output = np.vstack(weights)
    if loadData.FLIP:
        output = np.flip(output, 1)
    cv2.imwrite(folder_name+'/'+name+'_'+str(epoch)+'.jpg', output)


def writePredict(result_folder, result_file, input_images, predictions): # [batch_size, vocab_size] * max_output_len

    Path(result_folder).mkdir(parents=True, exist_ok=True)
    predictions = predictions.data
    top_predictions = predictions.topk(1)[1].squeeze(2) # (15, 32)
    top_predictions = top_predictions.transpose(0, 1) # (32, 15)
    top_predictions = top_predictions.cpu().numpy()

    batch_count_n = []
    with open(f'{result_folder}/{result_file}.log', 'a') as f:
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


def writeLoss(loss_value, flag):
    folder_name = 'pred_logs'
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    if flag == 'train':
        file_name = folder_name + '/loss_train.log'
    elif flag == 'valid':
        file_name = folder_name + '/loss_valid.log'
    elif flag == 'test':
        file_name = folder_name + '/loss_test.log'
    with open(file_name, 'a') as f:
        f.write(str(loss_value))
        f.write(' ')
