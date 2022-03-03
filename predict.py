import numpy as np
import pandas as pd
import random
import os
from sentence_transformers import SentenceTransformer, util
import torch
from torch import nn
from torch.backends import cudnn
from MultipleFieldsNet import MultipleFieldsNet
import argparse

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
PROJECT_PATH = './'


def set_seed(seed):
    """Set all seeds to make results reproducible (deterministic mode).
       When seed is None, disables deterministic mode.
    :param seed: an integer to your choosing
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def computeCosineSimilarity(emb1, emb2, length):
    cosine = util.pytorch_cos_sim(emb1, emb2)
    similarity = np.array([cosine[i][i].item() for i in range(length)]).tolist()
    return similarity


def weights_init_xavier_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data, gain=1.0)
        m.bias.data.fill_(0)

def create_submission_file(df, model, model_dimension, training_fields, id_column_name, score_column_name, model_path,
                           best_epoch):
    length = -1

    field1_news1 = training_fields[0] + '1'  # example "title1"
    field1_news2 = training_fields[0] + '2'

    field1_news1_list = df[field1_news1].tolist()
    field1_news2_list = df[field1_news2].tolist()
    length = len(field1_news1_list)
    field1_emb1 = model.encode(field1_news1_list, convert_to_tensor=True).to(DEVICE)
    field1_emb2 = model.encode(field1_news2_list, convert_to_tensor=True).to(DEVICE)

    # Multiple-field LMs
    if len(training_fields) >= 2:
        field2_news1 = training_fields[1] + '1'
        field2_news2 = training_fields[1] + '2'

        field2_news1_list = df[field2_news1].tolist()
        field2_news2_list = df[field2_news2].tolist()
        length = len(field2_news1_list)
        field2_emb1 = model.encode(field2_news1_list, convert_to_tensor=True).to(DEVICE)
        field2_emb2 = model.encode(field2_news2_list, convert_to_tensor=True).to(DEVICE)

        if len(training_fields) == 3:
            field3_news1 = training_fields[2] + '1'
            field3_news2 = training_fields[2] + '2'

            field3_news1_list = df[field3_news1].tolist()
            field3_news2_list = df[field3_news2].tolist()
            length = len(field3_news1_list)
            field3_emb1 = model.encode(field3_news1_list, convert_to_tensor=True).to(DEVICE)
            field3_emb2 = model.encode(field3_news2_list, convert_to_tensor=True).to(DEVICE)

            embedding1_train = torch.cat([field1_emb1, field2_emb1, field3_emb1], dim=1).to(DEVICE)
            embedding2_train = torch.cat([field1_emb2, field2_emb2, field3_emb2], dim=1).to(DEVICE)

        else:
            embedding1_train = torch.cat([field1_emb1, field2_emb1], dim=1).to(DEVICE)
            embedding2_train = torch.cat([field1_emb2, field2_emb2], dim=1).to(DEVICE)

        similarity = -999

        model_fully_conn = MultipleFieldsNet(in_features=len(training_fields) * model_dimension, out_features=model_dimension).to(
            DEVICE)
        model_fully_conn.apply(weights_init_xavier_uniform)

        path_saved_weights = model_path + "/evalstate_fully_conn_epoch" + str(best_epoch)
        model_fully_conn.load_state_dict(torch.load(path_saved_weights))
        model_fully_conn.eval()

        x1 = model_fully_conn(embedding1_train).to(DEVICE)
        x2 = model_fully_conn(embedding2_train).to(DEVICE)

        similarity = computeCosineSimilarity(emb1=x1, emb2=x2, length=length)

    # Single-field LMs
    else:
        similarity = computeCosineSimilarity(emb1=field1_emb1, emb2=field1_emb2, length=length)

    df_out = pd.DataFrame({})
    pair_id = df['pair_id'].tolist()
    df_out.insert(0, id_column_name, pair_id)
    df_out.insert(1, score_column_name, similarity)

    return df_out, similarity


def main(training_fields, file_path, path_to_model):

    df = pd.read_csv(PROJECT_PATH + file_path, sep=',')
    name_and_extension = file_path.split("/")[-1]
    file_name = name_and_extension.split(".")[0]

    # for Multiple-Fields LM we save the weights of the Dense layer at each epoch. We have to use the one related to the best model.
    evaluation_file = path_to_model + '/eval/similarity_evaluation_results.csv'  #csv file containing performances during training
    evaluation_df = pd.read_csv(evaluation_file, sep=",")
    idx = evaluation_df[['validation_cosine_pearson']].idxmax()
    best_epoch = evaluation_df.loc[idx, 'epoch'].item()


    model_name = SentenceTransformer(PROJECT_PATH + path_to_model)
    model_dimension = 768
    id_column_name = "pair_id"
    score_column_name = "Overall"
    df_out, _ = create_submission_file(df, model_name, model_dimension, training_fields, id_column_name, score_column_name,
                                       PROJECT_PATH + path_to_model, best_epoch=best_epoch)

    trained = "_".join(training_fields)

    df_out.to_csv(PROJECT_PATH + "predictions/" + file_name + "_" + "prediction_on_" + trained + ".csv", sep=",", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to make predictions using fine-tuned Single-Fields or Multiple-Fields LMs')

    parser.add_argument('-predict_on', type=str, help='Input file on which make predictions')
    parser.add_argument('-model_dir', type=str, help='absolute path to the model directory')
    args = parser.parse_args()

    if args.model_dir is None:
        parser.error("You must specify one model!!!")
    if args.predict_on is None:
        parser.error("-predict_on parameter required. Specify the file on which make predictions")

    model_dir = args.model_dir
    file = args.predict_on

    used_fields = []
    if "title" in model_dir:
        used_fields.append("title")
    if "description" in model_dir:
        used_fields.append("description")
    if "text" in model_dir:
        used_fields.append("text")

    main(training_fields=used_fields, file_path=file, path_to_model=model_dir)