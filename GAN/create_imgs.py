import cv2
import Levenshtein as Lev
import random
import numpy as np
import torch
from .network_tro import ConTranModel
from .load_data import IMG_HEIGHT, IMG_WIDTH, NUM_WRITERS, letter2index,\
    tokens, num_tokens, OUTPUT_MAX_LEN, index2letter
from .modules_tro import normalize
import os
from pathlib import Path
import yaml


with open('config.yaml') as f:
    config = yaml.safe_load(f)


def read_image(img_path):

    img = cv2.imread(img_path, 0)

    rate = float(IMG_HEIGHT) / img.shape[0]
    img = cv2.resize(img, (int(img.shape[1] * rate) + 1, IMG_HEIGHT),
                     interpolation=cv2.INTER_CUBIC)  # INTER_AREA con error
    img = img / 255.  # 0-255 -> 0-1

    reversed_img = 1. - img
    img_width = img.shape[-1]

    if img_width > IMG_WIDTH:
        out_img = reversed_img[:, :IMG_WIDTH]
    else:
        out_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
        out_img[:, :img_width] = reversed_img
    out_img = out_img.astype('float32')

    mean = 0.5
    std = 0.5
    out_img_final = (out_img - mean) / std
    return out_img_final


def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll)+2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens['PAD_TOKEN']] * num) # replace PAD_TOKEN
    return ll


def create_images(img_names, model_file, n_words, img_folder, result_folder):
    gpu = torch.device('cuda')

    '''data preparation'''
    imgs = [read_image(img_folder + img_name) for img_name in img_names]
    random.shuffle(imgs)
    final_imgs = imgs[:50]
    if len(final_imgs) < 50:
        while len(final_imgs) < 50:
            num_cp = 50 - len(final_imgs)
            final_imgs = final_imgs + imgs[:num_cp]

    imgs = torch.from_numpy(np.array(final_imgs)).unsqueeze(0).to(gpu) # 1,50,64,216

    with open(config['text_corpus'], 'r') as _f:
        texts = _f.read().split()
        texts = texts[0:n_words]
    labels = torch.from_numpy(np.array([np.array(label_padding(label, num_tokens)) for label in texts])).to(gpu)

    '''model loading'''
    model = ConTranModel(NUM_WRITERS, 0, True).to(gpu)
    print('Loading ' + model_file)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    num = 0
    with torch.no_grad():
        f_xs = model.gen.enc_image(imgs)
        for label in labels:
            label = label.unsqueeze(0)
            f_xt, f_embed = model.gen.enc_text(label, f_xs.shape)
            f_mix = model.gen.mix(f_xs, f_embed)
            xg = model.gen.decode(f_mix, f_xt)
            pred = model.rec(xg, label, img_width=torch.from_numpy(np.array([IMG_WIDTH])))

            label = label.squeeze().cpu().numpy().tolist()
            pred = torch.topk(pred, 1, dim=-1)[1].squeeze()
            pred = pred.cpu().numpy().tolist()
            for j in range(num_tokens):
                label = list(filter(lambda x: x!=j, label))
                pred = list(filter(lambda x: x!=j, pred))
            label = ''.join([index2letter[c-num_tokens] for c in label])
            pred = ''.join([index2letter[c-num_tokens] for c in pred])
            ed_value = Lev.distance(pred, label)
            if ed_value <= 100:
                num += 1
                xg = xg.cpu().numpy().squeeze()
                xg = normalize(xg)
                xg = 255 - xg
                path = f'{result_folder}/{num}.{label}-{pred}.png'
                ret = cv2.imwrite(path, xg)
                if not ret:
                    import pdb; pdb.set_trace()
                    xg


def create_images_from_source(writer_id, n_words, img_base, model, dataset_name='runs'):
    result_folder = Path(config['result_paths']['synthesized_images'], dataset_name, str(writer_id))

    if dataset_name == 'iam':
        target_file = Path(config['result_paths']['labels_path'], 'iam', f'style_of_{writer_id}')
        with open(target_file, 'r') as f:
            data = f.readlines()
            data = [line.split(' ')[0] for line in data]
        img_names = [line.split(',')[1] + '.png' for line in data]
    else:
        img_names = os.listdir(img_base)

    result_folder.mkdir(parents=True, exist_ok=True)
    create_images(img_names, model, n_words, img_base, result_folder)

