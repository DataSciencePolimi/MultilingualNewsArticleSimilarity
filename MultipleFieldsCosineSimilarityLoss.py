import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from MultipleFieldsNet import MultipleFieldsNet

class MultipleFieldsCosineSimilarityLoss(nn.Module):

    def __init__(self, model_bert: SentenceTransformer, model_fully_conn: MultipleFieldsNet, loss_fct=nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(MultipleFieldsCosineSimilarityLoss, self).__init__()
        self.model_bert = model_bert
        self.model_fully_conn = model_fully_conn
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]],  labels: Tensor):
        labels = labels.float()
        embeddings = [self.model_bert(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        '''Even Positions  are the fields of the first article, and they need to be concatenated together
        The odd positions are the fields of the second article, and they need to be be concatenated together 
        '''

        fields_first_article = [embeddings[i] for i in range(len(embeddings)) if i % 2 == 0]
        fields_second_article = [embeddings[i] for i in range(len(embeddings)) if i % 2 != 0]

        embedding1 = torch.cat(fields_first_article, dim=1)
        embedding2 = torch.cat(fields_second_article, dim=1)

        x1 = self.model_fully_conn(embedding1)
        x2 = self.model_fully_conn(embedding2)

        output = self.cos_score_transformation(torch.cosine_similarity(x1, x2))
        return self.loss_fct(output, labels.view(-1))

