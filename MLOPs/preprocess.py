import os
import pickle
import click
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pd.read_csv(filename)
    return df


def embedding(tokens, word2vec_model):
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(300)


@click.command()
@click.option(
    "--data_path",
    help="Location of both the Quora train and test dataset CSV file"
)
@click.option(
    "--output_path",
    help="Location where the resulting files will be saved"
)

def run_embedding_generation(data_path: str, output_path: str):
     train_df = read_dataframe(
        os.path.join(data_path, "traindf.csv")
    )
     test_df = read_dataframe(
         os.path.join(data_path, "testdf.csv")
    )

    # Preprocessing
     train_df['clean_text1'] = train_df['clean_text1'].astype(str)
     train_df['clean_text2'] = train_df['clean_text2'].astype(str)
     test_df['clean_text1'] = test_df['clean_text1'].astype(str)
     test_df['clean_text2'] = test_df['clean_text2'].astype(str)
     train_df['question1_token'] = train_df.clean_text1.apply(word_tokenize)
     train_df['question2_token'] = train_df.clean_text2.apply(word_tokenize)
     test_df['question1_token'] = test_df.clean_text1.apply(word_tokenize)
     test_df['question2_token'] = test_df.clean_text2.apply(word_tokenize)

    # Word2Vec model training
     tokens = train_df['question1_token'].tolist() + train_df['question2_token'].tolist() + test_df['question1_token'].tolist() + test_df['question2_token'].tolist()
     word2vec_model = Word2Vec(tokens, window=5, vector_size=300, min_count=1, workers=4)

    # Generate embeddings
     train_df['embedding_question1'] = train_df['question1_token'].apply(lambda tokens: embedding(tokens, word2vec_model))
     train_df['embedding_question2'] = train_df['question2_token'].apply(lambda tokens: embedding(tokens, word2vec_model))


    # Calculate cosine similarity
     def cos_similarity(row):
        embedding1 = row['embedding_question1']
        embedding2 = row['embedding_question2']
        similarity_score = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity_score

     train_df['cos_similarity'] = train_df.apply(cos_similarity, axis=1)
     test_df['cos_similarity'] = test_df.apply(cos_similarity, axis=1)

     y_train = train_df['is_duplicate']
     y_test = test_df['is_duplicate']

    # Separate columns for feature scaling
     columns_to_exclude = ['embedding_question1', 'embedding_question2']
     X_train = train_df.drop(['is_duplicate','question1_token','question2_token', 'clean_text1', 'clean_text2'], axis=1)
     X_test = test_df.drop(['is_duplicate','question1_token','question2_token', 'clean_text1', 'clean_text2'],axis=1)

    # Scale features
     columns_to_scale = [col for col in X_train.columns if col not in columns_to_exclude]

     scaler = MinMaxScaler()


     train_quest1vec = pd.DataFrame(X_train['embedding_question1'].tolist(), columns=[f'q1_{i}' for i in range(300)])
     train_quest2vec = pd.DataFrame(X_train['embedding_question2'].tolist(), columns=[f'q2_{i}' for i in range(300)])

     test_quest1vec = pd.DataFrame(X_test['embedding_question1'].tolist(), columns=[f'q1_{i}' for i in range(300)])
     test_quest2vec = pd.DataFrame(X_test['embedding_question2'].tolist(), columns=[f'q2_{i}' for i in range(300)])

     train_scaled_columns = scaler.fit_transform(X_train[columns_to_scale])
     test_scaled_columns = scaler.transform(X_test[columns_to_scale])
     train_scaled_data = pd.DataFrame(train_scaled_columns, columns=[f'scaled_{i}' for i in range(len(columns_to_scale))])
     test_scaled_data = pd.DataFrame(test_scaled_columns, columns=[f'scaled_{i}' for i in range(len(columns_to_scale))])
    

    # Concatenate embeddings and scaled features
     X_train = pd.concat([train_quest1vec, train_quest2vec, train_scaled_data], axis=1)
     X_test = pd.concat([test_quest1vec, test_quest2vec, test_scaled_data], axis=1)

    # Create output directory unless it already exists
     os.makedirs(output_path, exist_ok=True)

    # Save embeddings and processed data
     dump_pickle(scaler, os.path.join(output_path, "scaler.pkl"))
     dump_pickle(word2vec_model, os.path.join(output_path, "word2vec_model.pkl"))
     dump_pickle((X_train, y_train), os.path.join(output_path, "train.pkl"))
     dump_pickle((X_test, y_test), os.path.join(output_path, "test.pkl"))


if __name__ == '__main__':
    run_embedding_generation()


