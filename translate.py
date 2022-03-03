import argparse
import numpy as np
import pandas as pd
from easynmt import EasyNMT

PROJECT_PATH = './'


def translate_to_eng(model, row, col):
    content = row[col]
    if content is np.NaN:
        return np.NaN
    lang = ""
    if '1' in col:
        lang = row['url1_lang']
    else:
        lang = row['url2_lang']
    if lang != 'en':
        translations = model.translate(content, source_lang=lang, target_lang='en')
        return translations
    else:
        return str(content)


def main(file):
    df = pd.read_csv(PROJECT_PATH + file, sep=",")
    model = EasyNMT('opus-mt')
    columns = ['title1', 'title2', 'description1', 'description2', 'text1', 'text2']

    for field in columns:
        df[field] = df.apply(lambda row: translate_to_eng(model, row, col=field), axis=1)
    df.to_csv(file + '_translated.csv', sep=',', index=False)


# Passing the whole dataset to this script might be too heavy. Consider translating in batches and reassembling them
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to split dataset in train/validation ')
    parser.add_argument('-file_name', type=str, help='Input file to split')

    args = parser.parse_args()
    file_name = args.file_name

    if file_name is None:
        parser.error("-file_name parameter required. Specify file to translate")

    main(file=file_name)
