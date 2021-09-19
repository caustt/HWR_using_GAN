import string

CREATE_PAIRS = False

IMG_HEIGHT = 64
IMG_WIDTH = 216
MAX_CHARS = 10
# NUM_CHANNEL = 15
NUM_CHANNEL = 50
EXTRA_CHANNEL = NUM_CHANNEL + 1
NUM_WRITERS = 500  # iam
NORMAL = True
OUTPUT_MAX_LEN = MAX_CHARS + 2  # <GO>+groundtruth+<END>


def labelDictionary():
    labels = list(string.ascii_lowercase + string.ascii_uppercase)
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter


num_classes, letter2index, index2letter = labelDictionary()
tokens = {'GO_TOKEN': 0, 'END_TOKEN': 1, 'PAD_TOKEN': 2}
num_tokens = len(tokens.keys())
vocab_size = num_classes + num_tokens
