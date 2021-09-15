import yaml
from pathlib import Path
from os import listdir

with open('config.yaml') as f:
    config = yaml.safe_load(f)


def parse_data(writer):
    image_files, labels = [], []
    with open(config['iam_writer_mappings'], 'r') as writer_mapping:
        for line in writer_mapping.readlines():
            if line.split(',')[0] == writer:
                labels.append(" ".join(line.split(' ')[1:]))
                image_files.append(line.split(',')[1].split(' ')[0])

    example_labels = Path(config['result_paths']['labels_path'], 'iam', f'style_of_{writer}')
    example_labels.parent.mkdir(parents=True, exist_ok=True)
    with open(example_labels, 'w+') as f:
        for index, (image_file, label) in enumerate(zip(image_files, labels)):
            line = f'{writer},{image_file} {label}'
            f.write(line)
            if index >= 14:
                break

    groundtruth_labels = Path(config['result_paths']['labels_path'], 'iam', f'gt_{writer}')
    with open(groundtruth_labels, 'w+') as f:
        for image_file, label in zip(image_files, labels):
            line = f'{image_file}.png {label}'
            f.write(line)
    return example_labels, groundtruth_labels


def create_train_partition(writer, img_path, labels_path, dataset='iam'):
    label_file_path = Path(labels_path, dataset, f'train_partition_for_{writer}')
    label_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_file_path, 'w+') as label_file:
        for img in Path(img_path).iterdir():
            filename = img.name
            label = filename.split('.')[1].split('-')[0]
            line = f'{filename} {label}\n'
            label_file.write(line)
    return label_file_path


def create_writer_id(synthesized_img_folder):
    synthesized_img_folder.mkdir(parents=True, exist_ok=True)
    file_names = [int(folder) for folder in listdir(synthesized_img_folder)]
    if not len(file_names) == 0:
        run_id = max(file_names) + 1
    else:
        run_id = 1
    return run_id
