import pandas as pd
import numpy as np
import torch
import random
import os
from SbertTrainer import SbertTrainer
from datetime import datetime
import argparse
from torch.backends import cudnn


PROJECT_PATH = './'
# used to rescale from gold labels [1,4] where 4 represents completely different articles.
SCORE_MIN = -0.1
SCORE_MAX = 1


# Set all seeds to make results reproducible (deterministic mode)
def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def mapOnDifferentRange(value, input_start, input_end, output_start, output_end):
    return output_start + ((output_end - output_start) / (input_end - input_start)) * (value - input_start)


def main(training_fields, file_name, lang="en", bs=8, epochs=3, max_len=256, warmup_steps=0.10):
    SEED = 1024
    set_seed(SEED)

    file_name = file_name.split(".")[0]

    if lang == "en":
        model_name = "sentence-transformers/all-mpnet-base-v2"
    else:
        model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

    cache_folder = PROJECT_PATH + 'cache'
    output_path = PROJECT_PATH + 'output/' + file_name + '/'
    output_path = output_path + model_name + '_' + datetime.now().strftime("%d-%m-%Y__%H:%M:%S")

    trainer = SbertTrainer(SEED)
    trainer.set_model_parameters(model_name=model_name, model_dimension=768, output_path=output_path,
                                 cache_folder=cache_folder)
    trainer.set_train_parameters(bs, epochs, max_len, warmup_steps)

    directory = PROJECT_PATH + 'datasets/train/'
    X_train = pd.read_csv(directory + file_name + "_X_train.csv", sep=",")
    X_val = pd.read_csv(directory + file_name + "_X_val.csv", sep=",")
    y_train = pd.read_csv(directory + file_name + "_y_train.csv", sep=",")
    y_val = pd.read_csv(directory + file_name + "_y_val.csv", sep=",")

    # rescale gold labels on [-0.1,1] range
    y_train['Overall'] = y_train['Overall'].apply(lambda x: mapOnDifferentRange(x, input_start=4, input_end=1, output_start=SCORE_MIN, output_end=SCORE_MAX))
    y_val['Overall'] = y_val['Overall'].apply(lambda x: mapOnDifferentRange(x, input_start=4, input_end=1, output_start=SCORE_MIN, output_end=SCORE_MAX))

    trained_on = "_".join(training_fields)
    output_path = output_path + '_' + trained_on
    trainer.model_output_path = output_path

    # train Multi-Fields LM
    if len(training_fields) > 1:
        trainer.train_concatenated(training_fields=training_fields, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
    else:  # train Single-Field LM
        trainer.train(field_name=training_fields, X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to run Single-Field and Multilingual-Field training')

    parser.add_argument('-file_name', type=str, help='File to use for training. Related splitted files will be used')

    parser.add_argument('-l', type=str, help='Language of input data ["en" for english, "multi" for multilingual]')
    parser.add_argument('-bs', type=int, help='Batch size to use. Default is 8')
    parser.add_argument('-epochs', type=int, help='Training epochs. Default is 3')
    parser.add_argument('-max_len', type=int, help='Max length used for tokenization. Default is 256')
    parser.add_argument('-ws', type=float, help='Warmup steps expressed in percentage of train iterations. Default is 10% (0.10)')
    parser.add_argument('-tf1', type=str, help='First training field')
    parser.add_argument('-tf2', type=str, help='Second training field, optional')
    parser.add_argument('-tf3', type=str, help='Third training field, optional')
    args = parser.parse_args()

    considered_fields = [args.tf1, args.tf2, args.tf3]
    considered_fields = [elem for elem in considered_fields if elem is not None]

    if args.tf1 is None:
        parser.error("Specify at least one training field!!!")
    if args.l is None:
        parser.error("Specify language of input file")
    if args.file_name is None:
        parser.error("-file_name parameter required. Specify the file to use for training. File must be located at /datasets/train/")

    opt_bs = args.bs if args.bs is not None else 8
    opt_epochs = args.epochs if args.epochs is not None else 3
    opt_max_len = args.max_len if args.max_len is not None else 256
    opt_warmup_steps = args.ws if args.ws is not None else 0.1
    language = args.l if args.l is not None else "en"
    file = args.file_name

    main(considered_fields, file_name=file, lang=language, bs=opt_bs, epochs=opt_epochs, max_len=opt_max_len,
         warmup_steps=opt_warmup_steps)

