import logging
import math
import os
import random
import numpy as np
import pandas as pd
import torch.backends.cudnn
import torch.cuda
import torch.random
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses
from sentence_transformers import LoggingHandler
from sentence_transformers import SentenceTransformer, models
from MultipleFieldsExtendedEmbeddingSimilarityEvaluator import MultipleFieldsExtendedEmbeddingSimilarityEvaluator
from ExtendedEmbeddingSimilarityEvaluator import ExtendedEmbeddingSimilarityEvaluator
from MultipleFieldsCosineSimilarityLoss import MultipleFieldsCosineSimilarityLoss
from MultipleFieldsNet import MultipleFieldsNet

logger = logging.getLogger(__name__)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


#### /print debug information to stdout

def weights_init_xavier_uniform(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
        m.bias.data.fill_(0)


def warmup_steps_percentage(percent, dataloader_length, num_epochs):
    warmup_steps = math.ceil(dataloader_length * num_epochs * percent)
    logging.info("Warmup-steps: {}".format(warmup_steps))
    return warmup_steps


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def create_input_examples(field_name, inputs, target):  # field_name = title, description, text
    col1 = field_name + '1'
    col2 = field_name + '2'
    headers = [col1, col2, 'Overall']

    data = [inputs[col1], inputs[col2], target]
    input_df = pd.concat(data, axis=1, keys=headers)
    input_values = input_df.values.tolist()

    samples = [InputExample(texts=[elem[0], elem[1]], label=float(elem[2])) for elem in input_values]
    return samples


def create_input_examples_concatenation(training_fields, inputs, targets):
    field1_news1 = training_fields[0] + '1'
    field1_news2 = training_fields[0] + '2'
    field2_news1 = training_fields[1] + '1'
    field2_news2 = training_fields[1] + '2'
    if len(training_fields) == 2:
        headers = [field1_news1, field1_news2, field2_news1, field2_news2,  'Overall']

        data = [inputs[field1_news1], inputs[field1_news2], inputs[field2_news1], inputs[field2_news2], targets]
        input_df = pd.concat(data, axis=1, keys=headers)
        input_values = input_df.values.tolist()

        samples = [InputExample(texts=[elem[0], elem[1], elem[2], elem[3]], label=float(elem[4])) for elem in
                   input_values]
    elif len(training_fields) == 3:
        field3_news1 = training_fields[2] + '1'
        field3_news2 = training_fields[2] + '2'
        headers = [field1_news1, field1_news2, field2_news1, field2_news2, field3_news1, field3_news2, 'Overall']

        data = [inputs[field1_news1], inputs[field1_news2], inputs[field2_news1], inputs[field2_news2],
                inputs[field3_news1], inputs[field3_news2], targets]
        input_df = pd.concat(data, axis=1, keys=headers)
        input_values = input_df.values.tolist()

        samples = [InputExample(texts=[elem[0], elem[1], elem[2], elem[3], elem[4], elem[5]], label=float(elem[6])) for
                   elem in input_values]

    return samples


class SbertTrainer:
    def __init__(self, seed):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None
        self.model_name = ''
        self.model_dimension = 768
        self.seed = seed
        self.cache_folder = ''
        self.model_output_path = ''
        self.batch_size = 8
        self.num_epochs = 3
        self.max_seq_length = 256
        self.warmup_steps = 100

    def set_model_parameters(self, model_name, model_dimension, output_path, cache_folder='./cache'):
        self.model_name = model_name
        self.model_dimension = model_dimension
        self.cache_folder = cache_folder
        self.model_output_path = output_path

    def set_train_parameters(self, batch_size=8, epochs=3, max_len=256, warmup_steps=0.1):
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.max_seq_length = max_len
        self.warmup_steps = warmup_steps

    def train(self, field_name, X_train, X_val, y_train, y_val):
        set_seed(self.seed)

        training_field = field_name[0]

        logger.info(f"Create new model based on {self.model_name}")
        word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], cache_folder=self.cache_folder)

        train_samples = create_input_examples(field_name=training_field, inputs=X_train, target=y_train)
        val_samples = create_input_examples(field_name=training_field, inputs=X_val, target=y_val)

        evaluator = ExtendedEmbeddingSimilarityEvaluator.from_input_examples(train_examples=train_samples,
                                                                             validation_examples=val_samples)
        evaluator.training_fields = training_field

        # Define train dataset, dataloader and the train loss
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=self.batch_size)
        train_loss = losses.CosineSimilarityLoss(self.model)

        logger.info(f"Training on field {training_field}")

        if self.warmup_steps < 1:  # is expressed as percentage of total number of iteration
            self.warmup_steps = warmup_steps_percentage(self.warmup_steps, len(train_dataloader), self.num_epochs)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=evaluator,
                       epochs=self.num_epochs,
                       warmup_steps=self.warmup_steps,
                       save_best_model=True,
                       output_path=self.model_output_path)


    def train_concatenated(self, training_fields, X_train, X_val, y_train, y_val):
        set_seed(self.seed)

        logger.info(f"Create new model based on {self.model_name}")

        word_embedding_model = models.Transformer(self.model_name, max_seq_length=self.max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model], cache_folder=self.cache_folder)

        train_samples = create_input_examples_concatenation(training_fields=training_fields, inputs=X_train, targets=y_train)
        val_samples = create_input_examples_concatenation(training_fields=training_fields, inputs=X_val, targets=y_val)

        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=self.batch_size)

        model_fully_conn = MultipleFieldsNet(in_features=len(training_fields) * self.model_dimension,
                                             out_features=self.model_dimension).to(device=self.device)
        model_fully_conn.apply(weights_init_xavier_uniform)
        model_fully_conn.train()

        train_loss = MultipleFieldsCosineSimilarityLoss(model_bert=self.model, model_fully_conn=model_fully_conn)
        evaluator = MultipleFieldsExtendedEmbeddingSimilarityEvaluator.from_input_examples(
            training_fields=training_fields,
            train_examples=train_samples, validation_examples=val_samples)

        evaluator.fully_connected_model = model_fully_conn

        logger.info(f"Training on fields {','.join(training_fields)}")

        if self.warmup_steps < 1:  # is expressed as percentage of total number of iteration
            self.warmup_steps = warmup_steps_percentage(self.warmup_steps, len(train_dataloader), self.num_epochs)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=evaluator,
                       epochs=self.num_epochs,
                       warmup_steps=self.warmup_steps,
                       save_best_model=True,
                       output_path=self.model_output_path)
