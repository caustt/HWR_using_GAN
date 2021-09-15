import torch.utils.data as D
import cv2
import yaml
import numpy as np
from pathlib import Path

from HWR import marcalAugmentor

with open('config.yaml') as f:
    config = yaml.safe_load(f)

RM_BACKGROUND = False
FLIP = False # flip the image
#BATCH_SIZE = 64

OUTPUT_MAX_LEN = 23 # max-word length is 21  This value should be larger than 21+2 (<GO>+groundtruth+<END>)
IMG_WIDTH = 1011 # m01-084-07-00 max_length
IMG_HEIGHT = 64
#IMG_WIDTH = 256 # img_width < 256: padding   img_width > 256: resize to 256

def labelDictionary():
    labels = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())

class IAM_words(D.Dataset):
    def __init__(self, file_label, image_dir=Path(config['iam_words']), augmentation=True):
        self.file_label = file_label
        self.output_max_len = OUTPUT_MAX_LEN
        self.augmentation = augmentation
        self.image_dir = image_dir

        self.transformer = marcalAugmentor.augmentor

    def __getitem__(self, index):
        word = self.file_label[index]
        img, img_width = self.readImage_keepRatio(word[0], flip=FLIP)
        label, label_mask = self.label_padding(' '.join(word[1:]), num_tokens)
        return word[0], img, img_width, label
        #return {'index_sa': file_name, 'input_sa': in_data, 'output_sa': out_data, 'in_len_sa': in_len, 'out_len_sa': out_data_mask}

    def __len__(self):
        return len(self.file_label)

    def readImage_keepRatio(self, file_name, flip):
        url = self.image_dir / file_name
        img = cv2.imread(str(url), 0)
        if img is None:
            print('###!Cannot find image: ' + str(url))

        # Use adaptive thresholding in both models
        thresh, _ = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        img[img > thresh] = 255

        rate = float(IMG_HEIGHT) / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
        # c04-066-01-08.png 4*3, for too small images do not augment
        if self.augmentation: # augmentation for training data
            img_new = self.transformer(img)
            if img_new.shape[0] != 0 and img_new.shape[1] != 0:
                rate = float(IMG_HEIGHT) / img_new.shape[0]
                img = cv2.resize(img_new, (int(img_new.shape[1]*rate)+1, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC) # INTER_AREA con error
            else:
                img = 255 - img
        else:
            img = 255 - img

        img_width = img.shape[-1]

        if img_width > IMG_WIDTH:
            outImg = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            #outImg = img[:, :IMG_WIDTH]
            img_width = IMG_WIDTH
        else:
            outImg = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='uint8')
            outImg[:, :img_width] = img
        outImg = outImg/255. #float64
        outImg = outImg.astype('float32')
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        outImgFinal = np.zeros([3, *outImg.shape])
        for i in range(3):
            outImgFinal[i] = (outImg - mean[i]) / std[i]
        return outImgFinal, img_width


    def label_padding(self, labels, num_tokens):
        new_label_len = []
        ll = [letter2index[i] for i in labels]
        num = self.output_max_len - len(ll) - 2
        new_label_len.append(len(ll)+2)
        ll = np.array(ll) + num_tokens
        ll = list(ll)
        ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
        if not num == 0:
            ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN

        def make_weights(seq_lens, output_max_len):
            new_out = []
            for i in seq_lens:
                ele = [1]*i + [0]*(output_max_len -i)
                new_out.append(ele)
            return new_out
        return ll, make_weights(new_label_len, self.output_max_len)

