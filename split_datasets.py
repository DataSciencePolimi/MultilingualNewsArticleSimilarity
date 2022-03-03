import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

PROJECT_PATH = './'


def train_val_split(df, test_size=0.2, stratification_field="lang_type_simil", seed=1024):
    X_train, X_val, y_train, y_val = train_test_split(df.drop(labels=['Overall'], axis=1),
                                                      df['Overall'],
                                                      random_state=seed,
                                                      test_size=test_size,
                                                      stratify=df[stratification_field])
    return X_train, X_val, y_train, y_val


def main(file, seed, test_size):
    directory = PROJECT_PATH + 'datasets/train/'
    df = pd.read_csv(directory + file, sep=",")

    # create new field used for stratified split
    df['lang_type_simil'] = df["url1_lang"] + df["url2_lang"] + df['Overall'].astype(int).astype(str)

    X_train, X_val, y_train, y_val = train_val_split(df, test_size=test_size, seed=seed)

    # Save files
    name = file.split(".")[0]
    X_train.to_csv(directory + name + "_X_train.csv", sep=",", index=False)
    X_val.to_csv(directory + name + "_X_val.csv", sep=",", index=False)
    y_train.to_csv(directory + name + "_y_train.csv", sep=",", index=False)
    y_val.to_csv(directory + name + "_y_val.csv", sep=",", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to split dataset in train/validation ')
    parser.add_argument('-file_name', type=str, help='Input file to split')
    parser.add_argument('-test_size', type=float, help='Test size. Default is 20% expressed as 0.2')
    parser.add_argument('-seed', type=int, help='Random seed to use. Default is 1024')

    args = parser.parse_args()
    file_name = args.file_name
    random_seed = args.seed
    ts = args.test_size

    if file_name is None:
        parser.error("-file_name parameter required. Specify file to split! File must be located at /datasets/train/")
    if ts >= 1:
        parser.error("Test size must be in (0,1) !")

    main(file_name, seed=random_seed, test_size=ts)
