import csv
import logging
import os
from typing import List
import torch
from scipy.stats import pearsonr
import torch.backends.cudnn
from sentence_transformers.evaluation import SentenceEvaluator, SimilarityFunction
from sentence_transformers.readers import InputExample
from sklearn.metrics.pairwise import paired_cosine_distances
from torch import nn
import numpy as np
import random


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

class ExtendedEmbeddingSimilarityEvaluator(SentenceEvaluator):

    def __init__(self, train_sentences1: List[str], train_sentences2: List[str], train_scores: List[float],
                 validation_sentences1: List[str], validation_sentences2: List[str], validation_scores: List[float],
                 batch_size: int = 8,
                 main_similarity: SimilarityFunction = None, name: str = '', show_progress_bar: bool = False,
                 write_csv: bool = True):

        self.train_sentences1 = train_sentences1
        self.train_sentences2 = train_sentences2
        self.train_scores = train_scores

        self.validation_sentences1 = validation_sentences1
        self.validation_sentences2 = validation_sentences2
        self.validation_scores = validation_scores

        self.train_loss = nn.MSELoss()
        self.validation_loss = nn.MSELoss()

        assert len(self.train_sentences1) == len(self.train_sentences2)
        assert len(self.train_sentences1) == len(self.train_scores)
        assert len(self.validation_sentences1) == len(self.validation_sentences2)
        assert len(self.validation_sentences1) == len(self.validation_scores)

        self.write_csv = write_csv
        self.main_similarity = SimilarityFunction.COSINE
        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.train_field = ''
        self.csv_file = "similarity_evaluation" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["training_field", "epoch", "steps", "train_cosine_pearson", "validation_cosine_pearson",
                            "train_mse_loss", "validation_mse_loss"]

    @classmethod
    def from_input_examples(cls, train_examples: List[InputExample], validation_examples: List[InputExample],
                            **kwargs):
        train_sentences1 = []
        train_sentences2 = []
        train_scores = []

        for example in train_examples:
            train_sentences1.append(example.texts[0])
            train_sentences2.append(example.texts[1])
            train_scores.append(example.label)

        validation_sentences1 = []
        validation_sentences2 = []
        validation_scores = []

        for example in validation_examples:
            validation_sentences1.append(example.texts[0])
            validation_sentences2.append(example.texts[1])
            validation_scores.append(example.label)

        return cls(train_sentences1, train_sentences2, train_scores, validation_sentences1,
                   validation_sentences2, validation_scores, **kwargs)

    def getEmbeddings(self, model, sentences1, sentences2):
        embeddings1 = model.encode(sentences1, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        embeddings2 = model.encode(sentences2, batch_size=self.batch_size,
                                   show_progress_bar=self.show_progress_bar, convert_to_numpy=True)
        return embeddings1, embeddings2


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        validation_pearson_cosine = -999999999

        if epoch != -1:
            if steps == -1:  # is the end of an entire epoch

                out_txt = " after epoch {}:".format(epoch)
                logger.info("Evaluating " + out_txt)
                train_embeddings1, train_embeddings2 = self.getEmbeddings(model=model,
                                                                          sentences1=self.train_sentences1,
                                                                          sentences2=self.train_sentences2)
                train_labels = self.train_scores

                train_cosine_scores = 1 - (paired_cosine_distances(train_embeddings1, train_embeddings2))
                train_pearson_cosine, _ = pearsonr(train_labels, train_cosine_scores)
                logger.info("Cosine-Similarity on Train set:\tPearson: {:.4f}".format(train_pearson_cosine))

                validation_embeddings1, validation_embeddings2 = self.getEmbeddings(model=model,
                                                                                    sentences1=self.validation_sentences1,
                                                                                    sentences2=self.validation_sentences2)
                validation_labels = self.validation_scores

                validation_cosine_scores = 1 - (paired_cosine_distances(validation_embeddings1, validation_embeddings2))
                validation_pearson_cosine, _ = pearsonr(validation_labels, validation_cosine_scores)
                logger.info("Cosine-Similarity on Validation set :\tPearson: {:.4f}".format(validation_pearson_cosine))

                # Compute Loss with MSE Loss on Train and Validation batches
                train_input = torch.tensor(train_cosine_scores, requires_grad=False)
                train_target = torch.tensor(train_labels)
                train_mse_loss = self.train_loss(train_input, train_target)
                logger.info(f"MSE Loss on Train set: {train_mse_loss}")

                validation_input = torch.tensor(validation_cosine_scores, requires_grad=False)
                validation_target = torch.tensor(validation_labels)
                validation_mse_loss = self.validation_loss(validation_input, validation_target)
                logger.info(f"MSE Loss on Validation set: {validation_mse_loss}")

                if output_path is not None and self.write_csv:
                    csv_path = os.path.join(output_path, self.csv_file)
                    output_file_exists = os.path.isfile(csv_path)
                    with open(csv_path, newline='', mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not output_file_exists:
                            writer.writerow(self.csv_headers)

                        writer.writerow([self.train_field, epoch, steps, train_pearson_cosine, validation_pearson_cosine,
                             train_mse_loss.item(), validation_mse_loss.item()])

            else:
                validation_pearson_cosine = -999999999
        else:
            pass

        return validation_pearson_cosine


