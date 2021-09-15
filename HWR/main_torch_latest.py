import torch
from torch.autograd import Variable
import numpy as np
from .utils import vocab_size, tokens

LABEL_SMOOTH = True

Bi_GRU = True
VISUALIZE_TRAIN = True

BATCH_SIZE = 32
learning_rate = 2 * 1e-4
lr_milestone = [20, 40, 60, 80, 100]

lr_gamma = 0.5

START_TEST = 1e4 # 1e4: never run test 0: run test from beginning
FREEZE = False
freeze_milestone = [65, 90]
EARLY_STOP_EPOCH = 20 # None: no early stopping
HIDDEN_SIZE_ENC = 512
HIDDEN_SIZE_DEC = 512 # model/encoder.py SUM_UP=False: enc:dec = 1:2  SUM_UP=True: enc:dec = 1:1
CON_STEP = None # CON_STEP = 4 # encoder output squeeze step
EMBEDDING_SIZE = 60 # IAM
TRADEOFF_CONTEXT_EMBED = None # = 5 tradeoff between embedding:context vector = 1:5
TEACHER_FORCING = False
MODEL_SAVE_EPOCH = 1

class LabelSmoothing(torch.nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

log_softmax = torch.nn.LogSoftmax(dim=-1)
crit = LabelSmoothing(vocab_size, tokens['PAD_TOKEN'], 0.4)
# predict and gt follow the same shape of cross_entropy
# predict: 704, 83   gt: 704
def loss_label_smoothing(predict, gt):
    def smoothlabel_torch(x, amount=0.25, variance=5):
        mu = amount/x.shape[0]
        sigma = mu/variance
        noise = np.random.normal(mu, sigma, x.shape).astype('float32')
        smoothed = x*torch.from_numpy(1-noise.sum(1)).view(-1, 1).cuda() + torch.from_numpy(noise).cuda()
        return smoothed

    def one_hot(src): # src: torch.cuda.LongTensor
        ones = torch.eye(vocab_size).cuda()
        return ones.index_select(0, src)

    gt_local = one_hot(gt.data)
    gt_local = smoothlabel_torch(gt_local)
    loss_f = torch.nn.BCEWithLogitsLoss()
    gt_local = Variable(gt_local)
    res_loss = loss_f(predict, gt_local)
    return res_loss

def teacher_force_func(epoch):
    if epoch < 50:
        teacher_rate = 0.5
    elif epoch < 150:
        teacher_rate = (50 - (epoch-50)//2) / 100.
    else:
        teacher_rate = 0.
    return teacher_rate

def teacher_force_func_2(epoch):
    if epoch < 200:
        teacher_rate = (100 - epoch//2) / 100.
    else:
        teacher_rate = 0.
    return teacher_rate


def sort_batch(batch):
    n_batch = len(batch)
    train_index = []
    train_in = []
    train_in_len = []
    train_out = []
    for i in range(n_batch):
        idx, img, img_width, label = batch[i]
        train_index.append(idx)
        train_in.append(img)
        train_in_len.append(img_width)
        train_out.append(label)

    train_index = np.array(train_index)
    train_in = np.array(train_in, dtype='float32')
    train_out = np.array(train_out, dtype='int64')
    train_in_len = np.array(train_in_len, dtype='int64')

    train_in = torch.from_numpy(train_in)
    train_out = torch.from_numpy(train_out)
    train_in_len = torch.from_numpy(train_in_len)

    train_in_len, idx = train_in_len.sort(0, descending=True)
    train_in = train_in[idx]
    train_out = train_out[idx]
    train_index = train_index[idx]
    return train_index, train_in, train_in_len, train_out