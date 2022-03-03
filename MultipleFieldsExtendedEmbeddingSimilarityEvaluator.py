import csv
import logging
import os
import random
from typing import List
import numpy as np
import torch
import torch.backends.cudnn
from scipy.stats import pearsonr
from sentence_transformers import util
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from sentence_transformers.readers import InputExample
from torch import nn

logger = logging.getLogger(__name__)


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


class MultipleFieldsExtendedEmbeddingSimilarityEvaluator(SentenceEvaluator):

    def __init__(self, train_sentences1_field1: List[str], train_sentences2_field1: List[str],
                 train_sentences1_field2: List[str], train_sentences2_field2: List[str],
                 train_sentences1_field3: List[str], train_sentences2_field3: List[str],
                 train_scores: List[float],
                 validation_sentences1_field1: List[str], validation_sentences2_field1: List[str],
                 validation_sentences1_field2: List[str], validation_sentences2_field2: List[str],
                 validation_sentences1_field3: List[str], validation_sentences2_field3: List[str],
                 validation_scores: List[float],
                 training_fields: List[str],
                 batch_size: int = 8,
                 main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False,
                 write_csv: bool = True):

        self.num_fields = len(training_fields)
        self.training_fields = training_fields

        self.train_sentences1_field1 = train_sentences1_field1
        self.train_sentences2_field1 = train_sentences2_field1
        self.train_sentences1_field2 = train_sentences1_field2
        self.train_sentences2_field2 = train_sentences2_field2

        # if training_fields == 2 these two will be empty lists
        self.train_sentences1_field3 = train_sentences1_field3
        self.train_sentences2_field3 = train_sentences2_field3

        self.train_scores = train_scores

        self.validation_sentences1_field1 = validation_sentences1_field1
        self.validation_sentences2_field1 = validation_sentences2_field1
        self.validation_sentences1_field2 = validation_sentences1_field2
        self.validation_sentences2_field2 = validation_sentences2_field2

        # if training_fields == 2 these two will be empty lists
        self.validation_sentences1_field3 = validation_sentences1_field3
        self.validation_sentences2_field3 = validation_sentences2_field3

        self.validation_scores = validation_scores

        self.loss_function = nn.MSELoss()
        self.fully_connected_model = None

        assert len(self.train_sentences1_field1) == len(self.train_sentences2_field1)
        assert len(self.train_sentences1_field1) == len(self.train_scores)
        assert len(self.validation_sentences1_field1) == len(self.validation_sentences2_field1)
        assert len(self.validation_sentences1_field1) == len(self.validation_scores)

        assert len(self.train_sentences1_field2) == len(self.train_sentences2_field2)
        assert len(self.train_sentences1_field2) == len(self.train_scores)
        assert len(self.validation_sentences1_field2) == len(self.validation_sentences2_field2)
        assert len(self.validation_sentences1_field2) == len(self.validation_scores)

        if self.num_fields == 3:
            assert len(self.train_sentences1_field3) == len(self.train_sentences2_field3)
            assert len(self.train_sentences1_field3) == len(self.train_scores)
            assert len(self.validation_sentences1_field3) == len(self.validation_sentences2_field3)
            assert len(self.validation_sentences1_field3) == len(self.validation_scores)

        self.write_csv = write_csv
        self.main_similarity = SimilarityFunction.COSINE
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar


        self.csv_file = "similarity_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["training_field", "epoch", "steps", "train_cosine_pearson", "validation_cosine_pearson",
                            "train_mse_loss", "validation_mse_loss"]

    @classmethod
    def from_input_examples(cls, training_fields: int, train_examples: List[InputExample],
                            validation_examples: List[InputExample],
                            **kwargs):
        num_fields = len(training_fields)
        train_sentences1_field1 = []
        train_sentences2_field1 = []
        train_sentences1_field2 = []
        train_sentences2_field2 = []

        # used only if training_fields == 3
        train_sentences1_field3 = []
        train_sentences2_field3 = []

        train_scores = []

        for example in train_examples:
            train_sentences1_field1.append(example.texts[0])
            train_sentences2_field1.append(example.texts[1])
            train_sentences1_field2.append(example.texts[2])
            train_sentences2_field2.append(example.texts[3])
            train_scores.append(example.label)
            if num_fields == 3:
                train_sentences1_field3.append(example.texts[4])
                train_sentences2_field3.append(example.texts[5])

        validation_sentences1_field1 = []
        validation_sentences2_field1 = []
        validation_sentences1_field2 = []
        validation_sentences2_field2 = []

        # used only if training_fields == 3
        validation_sentences1_field3 = []
        validation_sentences2_field3 = []

        validation_scores = []

        for example in validation_examples:
            validation_sentences1_field1.append(example.texts[0])
            validation_sentences2_field1.append(example.texts[1])
            validation_sentences1_field2.append(example.texts[2])
            validation_sentences2_field2.append(example.texts[3])
            if num_fields == 3:
                validation_sentences1_field3.append(example.texts[4])
                validation_sentences2_field3.append(example.texts[5])

            validation_scores.append(example.label)

        return cls(train_sentences1_field1, train_sentences2_field1,
                   train_sentences1_field2, train_sentences2_field2,
                   train_sentences1_field3, train_sentences2_field3,
                   train_scores,
                   validation_sentences1_field1, validation_sentences2_field1,
                   validation_sentences1_field2, validation_sentences2_field2,
                   validation_sentences1_field3, validation_sentences2_field3,
                   validation_scores, training_fields, **kwargs)

    def getEmbeddings(self, model, sentences1_field1, sentences2_field1, sentences1_field2, sentences2_field2,
                      sentences1_field3, sentences2_field3):
        embeddings1_field1 = model.encode(sentences1_field1, batch_size=self.batch_size,
                                          show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2_field1 = model.encode(sentences2_field1, batch_size=self.batch_size,
                                          show_progress_bar=self.show_progress_bar, convert_to_numpy=True)

        embeddings1_field2 = model.encode(sentences1_field2, batch_size=self.batch_size,
                                          show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2_field2 = model.encode(sentences2_field2, batch_size=self.batch_size,
                                          show_progress_bar=self.show_progress_bar, convert_to_numpy=True)

        embeddings1_field1 = torch.from_numpy(embeddings1_field1).to("cuda")
        embeddings2_field1 = torch.from_numpy(embeddings2_field1).to("cuda")
        embeddings1_field2 = torch.from_numpy(embeddings1_field2).to("cuda")
        embeddings2_field2 = torch.from_numpy(embeddings2_field2).to("cuda")

        if self.num_fields == 3:
            embeddings1_field3 = model.encode(sentences1_field3, batch_size=self.batch_size,
                                              show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
            embeddings2_field3 = model.encode(sentences2_field3, batch_size=self.batch_size,
                                              show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
            embeddings1_field3 = torch.from_numpy(embeddings1_field3).to("cuda")
            embeddings2_field3 = torch.from_numpy(embeddings2_field3).to("cuda")
            return embeddings1_field1, embeddings2_field1, embeddings1_field2, embeddings2_field2, embeddings1_field3, embeddings2_field3

        return embeddings1_field1, embeddings2_field1, embeddings1_field2, embeddings2_field2, None, None

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        validation_pearson_cosine = -999999999

        if epoch != -1:
            if steps == -1:

                train_embeddings1_field1, train_embeddings2_field1, train_embeddings1_field2, train_embeddings2_field2, \
                train_embeddings1_field3, train_embeddings2_field3 = self.getEmbeddings(
                    model=model,
                    sentences1_field1=self.train_sentences1_field1,
                    sentences2_field1=self.train_sentences2_field1,
                    sentences1_field2=self.train_sentences1_field2,
                    sentences2_field2=self.train_sentences2_field2,
                    sentences1_field3=self.train_sentences1_field3,
                    sentences2_field3=self.train_sentences2_field3)
                train_labels = self.train_scores

                if self.num_fields == 3:
                    embedding1_train = torch.cat([train_embeddings1_field1, train_embeddings1_field2, train_embeddings1_field3], dim=1)
                    embedding2_train = torch.cat([train_embeddings2_field1, train_embeddings2_field2, train_embeddings2_field3], dim=1)
                else:
                    embedding1_train = torch.cat([train_embeddings1_field1, train_embeddings1_field2], dim=1)
                    embedding2_train = torch.cat([train_embeddings2_field1, train_embeddings2_field2], dim=1)

                self.fully_connected_model.eval()

                x1_t = self.fully_connected_model(embedding1_train).to("cuda")
                x2_t = self.fully_connected_model(embedding2_train).to("cuda")

                cosine_train = util.pytorch_cos_sim(x1_t, x2_t)
                simil_train = np.array([cosine_train[i][i].item() for i in range(len(train_embeddings1_field1))]).tolist()
                train_pearson_cosine, _ = pearsonr(train_labels, simil_train)
                logger.info("Cosine-Similarity on Train set:\tPearson: {:.4f}".format(train_pearson_cosine))

                validation_embeddings1_field1, validation_embeddings2_field1, validation_embeddings1_field2, validation_embeddings2_field2, \
                validation_embeddings1_field3, validation_embeddings2_field3 = self.getEmbeddings(
                    model=model,
                    sentences1_field1=self.validation_sentences1_field1,
                    sentences2_field1=self.validation_sentences2_field1,
                    sentences1_field2=self.validation_sentences1_field2,
                    sentences2_field2=self.validation_sentences2_field2,
                    sentences1_field3=self.validation_sentences1_field3,
                    sentences2_field3=self.validation_sentences2_field3)
                validation_labels = self.validation_scores

                if self.num_fields == 3:
                    embedding1_validation = torch.cat([validation_embeddings1_field1, validation_embeddings1_field2, validation_embeddings1_field3], dim=1)
                    embedding2_validation = torch.cat([validation_embeddings2_field1, validation_embeddings2_field2, validation_embeddings2_field3],dim=1)
                else:
                    embedding1_validation = torch.cat([validation_embeddings1_field1, validation_embeddings1_field2], dim=1)
                    embedding2_validation = torch.cat([validation_embeddings2_field1, validation_embeddings2_field2], dim=1)

                x1_v = self.fully_connected_model(embedding1_validation).to("cuda")
                x2_v = self.fully_connected_model(embedding2_validation).to("cuda")

                cosine_validation = util.pytorch_cos_sim(x1_v, x2_v)
                simil_validation = np.array([cosine_validation[i][i].item() for i in range(len(validation_embeddings1_field2))]).tolist()
                validation_pearson_cosine, _ = pearsonr(validation_labels, simil_validation)
                logger.info("Cosine-Similarity on Validation set :\tPearson: {:.4f}".format(validation_pearson_cosine))

                # Compute Loss with MSE Loss on Train and Validation batches
                train_input = torch.tensor(simil_train, requires_grad=False)
                train_target = torch.tensor(train_labels)
                train_mse_loss = self.loss_function(train_input, train_target)
                logger.info(f"MSE Loss on Train set: {train_mse_loss}")

                validation_input = torch.tensor(simil_validation, requires_grad=False)
                validation_target = torch.tensor(validation_labels)
                validation_mse_loss = self.loss_function(validation_input, validation_target)
                logger.info(f"MSE Loss on Validation set: {validation_mse_loss}")

                if output_path is not None and self.write_csv:
                    csv_path = os.path.join(output_path, self.csv_file)
                    output_file_exists = os.path.isfile(csv_path)
                    with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not output_file_exists:
                            writer.writerow(self.csv_headers)

                        writer.writerow(
                            [','.join(self.training_fields), epoch, steps, train_pearson_cosine, validation_pearson_cosine,
                             train_mse_loss.item(), validation_mse_loss.item()])

                torch.save(self.fully_connected_model.state_dict(), output_path + "state_fully_conn_epoch" + str(epoch))

                self.fully_connected_model.train()
            else:
                validation_pearson_cosine = -999999999
                self.fully_connected_model.train()
        else:
            pass

        return validation_pearson_cosine
