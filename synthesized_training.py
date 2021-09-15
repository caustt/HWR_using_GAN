import argparse
import yaml
import shutil
from pathlib import Path
from GAN.create_imgs import create_images_from_source
from HWR.train_with_synth_imgs import train_with_synth_imgs, test_images
from utils import create_writer_id, parse_data, create_train_partition

parser = argparse.ArgumentParser(description='Create synthesized images from input images and train HWR with them.')
parser.add_argument('--n_generated_images', default=150, type=int,
                    help='The number of images that the GAN will produce and the HWR train on.')
parser.add_argument('--input_folder', default='input_images/', help='Folder that contains the input images.')
parser.add_argument('--test_folder', default='test_images/', help='Folder that contains the images to test the trained model on.')
parser.add_argument('-t', action='store_true',
                    help='If specified, the model will be tested on images in the test_folder and compared to the default model.')
parser.add_argument('--iam', type=str, default=None, help='If you want to train on a specified IAM writer.')

args = parser.parse_args()

with open('config.yaml') as f:
    config = yaml.safe_load(f)

# Prepare writer input
if args.iam:
    dataset = 'iam'
    run_id = args.iam
    synthesized_img_folder = Path(config['result_paths']['synthesized_images'], dataset, run_id)
    input_folder = config['iam_words']

    # create index file with words of this writer in train_images_names for creating imgages with GAN
    # create index file with words for HWR testing in HWR_Groundtruth
    _, groundtruth_labels = parse_data(run_id)

    # delete old images
    if synthesized_img_folder.exists():
        shutil.rmtree(synthesized_img_folder)
else:
    dataset = 'runs'
    synthesized_img_folder = Path(config['result_paths']['synthesized_images'], dataset)
    run_id = str(create_writer_id(synthesized_img_folder))
    synthesized_img_folder = Path(synthesized_img_folder, run_id)
    input_folder = args.input_folder

create_images_from_source(
    run_id,
    args.n_generated_images,
    input_folder,
    config['gan_default_model'],
    dataset)
hwr_training_labels_file = create_train_partition(
    run_id,
    synthesized_img_folder,
    config['result_paths']['labels_path'],
    dataset)
trained_model_path = train_with_synth_imgs(
    run_id,
    hwr_training_labels_file,
    synthesized_img_folder,
    dataset == 'iam')

if args.t:
    if args.iam:
        test_images(groundtruth_labels, Path(config['iam_words']),
                    trained_model_path, f'writer_{run_id}_')
        test_images(groundtruth_labels, Path(config['iam_words']), config['hwr_default_model'],
                    f'writer_{run_id}_original_')
    else:
        test_folder = Path(args.test_folder)
        labels_file = Path(test_folder, 'labels.txt')
        test_images(labels_file, test_folder,
                    trained_model_path, f'run_{run_id}_')
        test_images(labels_file, test_folder, config['hwr_default_model'],
                    f'run_{run_id}_original_')

