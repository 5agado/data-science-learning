import argparse
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime

from dcgan import DCGan

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from tmp_load_data import load_celeba_tfdataset


def main(_=None):
    parser = argparse.ArgumentParser(description='Train GAN')
    parser.add_argument('--config', required=True, help="config path")
    parser.add_argument('--name', required=True, help="model name")
    parser.add_argument('--model-dir', required=True, help="model directory")
    parser.add_argument('--data-dir', required=True, help="data directory")
    parser.add_argument('--epochs', default=1000, help="number of training epochs")

    args = parser.parse_args()

    CONFIG_PATH = args.config
    MODEL_NAME = args.name
    MODEL_DIR = Path(args.model_dir)
    DATA_DIR = Path(args.data_dir)
    NB_EPOCHS = args.epochs

    # load model config
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.load(f)
    IMG_SHAPE = config['data']['input_shape']

    # load data
    train_ds = load_celeba_tfdataset(DATA_DIR, config, zipped=False)
    test_ds = load_celeba_tfdataset(DATA_DIR, config, zipped=False)

    # instantiate GAN
    gan = DCGan(IMG_SHAPE, config)

    # setup model directory for checkpoint and tensorboard logs
    model_dir = MODEL_DIR / MODEL_NAME
    model_dir.mkdir(exist_ok=True, parents=True)
    log_dir = model_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")

    # run train
    gan._train(train_ds=gan.setup_dataset(train_ds),
               validation_ds=gan.setup_dataset(test_ds),
               nb_epochs=NB_EPOCHS,
               log_dir=log_dir,
               checkpoint_dir=None,
               is_tfdataset=True)


if __name__ == "__main__":
    main(sys.argv[1:])
