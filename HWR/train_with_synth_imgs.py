from .main_torch_latest import BATCH_SIZE, sort_batch, HIDDEN_SIZE_DEC, HIDDEN_SIZE_ENC, EMBEDDING_SIZE, \
    TRADEOFF_CONTEXT_EMBED, Bi_GRU, CON_STEP, crit, learning_rate, lr_milestone, lr_gamma
from .loadData2_vgg import IAM_words
from .utils import writePredict, HEIGHT, WIDTH, output_max_len, vocab_size, FLIP
import torch
from torch import optim
from .models.encoder_vgg import Encoder
from .models.decoder import Decoder
from .models.attention import locationAttention as Attention
from .models.seq2seq import Seq2Seq
from pathlib import Path
import yaml
import subprocess as sub

with open('config.yaml') as f:
    config = yaml.safe_load(f)

EPOCH = int(config['number_training_epochs'])


def init_s2s_model(model_file=config['hwr_default_model']):
    encoder = Encoder(HIDDEN_SIZE_ENC, HEIGHT, WIDTH, Bi_GRU, CON_STEP, FLIP).cuda()
    decoder = Decoder(HIDDEN_SIZE_DEC, EMBEDDING_SIZE, vocab_size, Attention, TRADEOFF_CONTEXT_EMBED).cuda()
    seq2seq = Seq2Seq(encoder, decoder, output_max_len, vocab_size).cuda()
    print(f'Loading {model_file}')
    seq2seq.load_state_dict(torch.load(model_file))  # load
    return seq2seq


def calc_loss(output, labels):
    test_label = labels.permute(1, 0)[1:].contiguous().view(-1)  # remove<GO>
    output_label = output.view(-1, vocab_size)  # remove last <EOS>
    log_softmax = torch.nn.LogSoftmax(dim=-1)
    loss = crit(log_softmax(output_label), test_label)
    return loss


def train(train_loader, seq2seq, opt, prediction_log_file=None):
    seq2seq.train()
    total_loss = 0
    for num, (train_index, train_in, train_in_len, train_out) in enumerate(train_loader):
        train_in, train_out = train_in.cuda(), train_out.cuda()
        output, attention_weights = seq2seq(train_in, train_out, train_in_len,
                                            teacher_rate=0, train=True)  # (100-1, 32, 62+1)

        if prediction_log_file:
            writePredict(prediction_log_file, train_index, output)

        loss = calc_loss(output, train_out)
        total_loss += loss.data.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    total_loss /= (num + 1)
    return total_loss


def test(test_loader, seq2seq, prediction_log_file):
    seq2seq.eval()
    total_loss = 0
    for num, (test_index, test_in, test_in_len, test_out) in enumerate(test_loader):
        with torch.no_grad():
            test_in, test_out = test_in.cuda(), test_out.cuda()
            output, attention_weights = seq2seq(test_in, test_out, test_in_len, teacher_rate=False, train=False)

        writePredict(prediction_log_file, test_index, output)
        loss = calc_loss(output, test_out)
        total_loss += loss.data.item()

    total_loss /= (num + 1)
    return total_loss


def train_with_synth_imgs(run_or_writer_id, labels_file, image_folder, iam=False):
    train_loader = get_data_loader(labels_file, image_folder)
    seq2seq = init_s2s_model()
    opt = optim.Adam(seq2seq.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(opt, milestones=lr_milestone, gamma=lr_gamma)

    for epoch in range(EPOCH):
        if iam:
            train(train_loader, seq2seq, opt, Path(config['result_paths']['evaluations'],
                                                   f'writer_{run_or_writer_id}_train_predict_seq.{epoch + 1}.log'))
        else:
            train(train_loader, seq2seq, opt, Path(config['result_paths']['evaluations'],
                                                   f'run_{run_or_writer_id}_train_predict_seq.{epoch + 1}.log'))
        scheduler.step()

    result_folder = 'iam' if iam else 'runs'
    model_path = Path(config['result_paths']['model_weights'], result_folder, f'{run_or_writer_id}.model')
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(seq2seq.state_dict(), model_path)
    print(f'Trained model saved as {model_path}')
    return model_path


def test_images(labels_file, image_folder, model_path, prediction_file_prefix):
    test_loader = get_data_loader(labels_file, image_folder)
    seq2seq = init_s2s_model(model_path)
    prediction_log_file = Path(config['result_paths']['evaluations'], f'{prediction_file_prefix}test_predict_seq.log')
    test(test_loader, seq2seq, prediction_log_file)
    cer = calc_cer(labels_file, prediction_log_file)
    print(f'CER of {model_path}: {cer}')


def calc_cer(labels_file, predictions_file):
    cer = sub.Popen(['HWR/tasas_cer.sh', labels_file, predictions_file], stdout=sub.PIPE)
    cer = float(cer.stdout.read().decode('utf8')) / 100
    return cer


def get_data_loader(labels_file, image_folder=None):
    with open(labels_file, 'r') as f_tr:
        data = f_tr.readlines()
        file_labels = [i[:-1].split(' ') for i in data]

    if image_folder:
        data_set = IAM_words(file_labels, image_folder, augmentation=False)
    else:
        data_set = IAM_words(file_labels, augmentation=False)
    data_loader = torch.utils.data.DataLoader(data_set, collate_fn=sort_batch, batch_size=BATCH_SIZE, shuffle=False,
                                              num_workers=0, pin_memory=True)
    return data_loader
