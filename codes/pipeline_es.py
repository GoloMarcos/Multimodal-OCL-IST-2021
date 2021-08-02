from gc import collect
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import sys
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, concatenate, multiply, average, subtract, add
from tensorflow.keras.models import Model, save_model
from scikit_learn.sklearn_svdd.svm import SVDD


def foldValidation(folds):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    return kf


def interest_outlier(df, is_Bug=False, class_interest='', class_outlier=''):
    if is_Bug:

        df_interest = df[df.category == class_interest]

        df_outlier = df[(df.category != class_interest) & (df.is_test == 1)]
    else:

        df_interest = df[df.category != class_outlier]

        df_outlier = df[(df.category == class_outlier) & (df.is_test == 1)]

    return df_interest, df_outlier


def train_test_split_one_class(kf, df_int, column, percent):
    train_test = []

    for train_index, test_index in kf.split(df_int):

        df_train = np.array(df_int[column].to_list())[train_index]
        df_test = np.array(df_int[column].to_list())[test_index]

        if percent != 1:
            df_train, _ = train_test_split(df_train, train_size=percent, random_state=42)

        train_test.append((df_train, df_test))

        del df_train
        del df_test
        collect()

    return train_test


def make_representation(train_test, df_outlier, representation_type, vectorizer='', cluster_list='',
                        parameter_list_AE='', parameter_list_VAE='', parameter_list_MAE='', parameter_list_MVAE=''):
    representations = list()
    fold = 1

    for df_train, df_test in train_test:

        if representation_type == 'Maalej' or representation_type == 'DBERTML':

            representations.append((np.array(df_train), np.array(df_test), np.array(df_outlier)))

        elif representation_type == 'FastText':
            length = len(df_train[0]) - 300
            X_train = np.array(df_train[:, length:])
            X_test = np.array(df_test[:, length:])
            X_outlier = np.array(df_outlier[:, length:])

            representations.append((X_train, X_test, X_outlier))

        elif representation_type == 'BoW':

            vectorizer.fit(df_train)
            X_train = vectorizer.transform(df_train)
            X_test = vectorizer.transform(df_test)
            X_outlier = vectorizer.transform(df_outlier)

            representations.append((X_train.toarray(), X_test.toarray(), X_outlier.toarray()))

        elif representation_type == 'Density':

            X_train, X_test, X_outlier = make_density_information(cluster_list, df_train, df_test, np.array(df_outlier))

            representations.append((X_train, X_test, X_outlier))

        elif representation_type == 'AE':

            epoch = parameter_list_AE[0]
            arq = parameter_list_AE[1]
            model_file_name = parameter_list_AE[2] + '_Fold' + str(fold)
            path_model = parameter_list_AE[3]

            ae, encoder = autoencoder(arq, len(df_train[0]))

            ae.fit(df_train, df_train, epochs=epoch, batch_size=32, verbose=0)

            save_model(ae, path_model + model_file_name + '_autoencoder')
            save_model(encoder, path_model + model_file_name + '_encoder_AE')

            X_train = encoder.predict(np.array(df_train))
            X_test = encoder.predict(np.array(df_test))
            X_outlier = encoder.predict(np.array(df_outlier))

            representations.append((X_train, X_test, X_outlier))

        elif representation_type == 'VAE':

            epoch = parameter_list_VAE[0]
            arq = parameter_list_VAE[1]
            model_file_name = parameter_list_VAE[2] + '_Fold' + str(fold)
            path_model = parameter_list_VAE[3]

            vae, encoder, decoder = VariationalAutoencoder(arq, len(df_train[0]))

            vae.fit(df_train, df_train, epochs=epoch, batch_size=32, verbose=0)

            save_model(encoder, path_model + model_file_name + '_encoder_VAE')
            save_model(decoder, path_model + model_file_name + '_decoder_VAE')

            X_train, _, _ = encoder.predict(np.array(df_train))
            X_test, _, _ = encoder.predict(np.array(df_test))
            X_outlier, _, _ = encoder.predict(np.array(df_outlier))

            representations.append((X_train, X_test, X_outlier))

        elif representation_type == 'MAE':

            epoch = parameter_list_MAE[0]
            arq = parameter_list_MAE[1]
            model_file_name = parameter_list_MAE[2] + '_Fold' + str(fold)
            path_model = parameter_list_MAE[3]
            operator = parameter_list_MAE[4]

            denisty_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                    np.array(df_outlier))

            aem, encoder = multimodal_autoencoder(arq, len(df_train[0]), len(cluster_list), operator)

            aem.fit([df_train, denisty_train], [df_train, denisty_train], epochs=epoch, batch_size=32, verbose=0)

            save_model(aem, path_model + model_file_name + '_mae')
            save_model(encoder, path_model + model_file_name + '_encoder_MAE')

            X_train = encoder.predict([np.array(df_train), denisty_train])
            X_test = encoder.predict([np.array(df_test), density_test])
            X_outlier = encoder.predict([np.array(df_outlier), density_outlier])

            representations.append((X_train, X_test, X_outlier))

        elif representation_type == 'MVAE':

            epoch = parameter_list_MVAE[0]
            arq = parameter_list_MVAE[1]
            model_file_name = parameter_list_MVAE[2] + '_Fold' + str(fold)
            path_model = parameter_list_MVAE[3]
            operator = parameter_list_MVAE[4]

            denisty_train, density_test, density_outlier = make_density_information(cluster_list, df_train, df_test,
                                                                                    np.array(df_outlier))

            mvae, encoder, decoder = MultimodalVAE(arq, len(df_train[0]), len(cluster_list), operator)

            mvae.fit([df_train, denisty_train], [df_train, denisty_train], epochs=epoch, batch_size=32, verbose=0)

            save_model(encoder, path_model + model_file_name + '_encoder_MVAE')
            save_model(decoder, path_model + model_file_name + '_decoder_MVAE')

            X_train, _, _ = encoder.predict([np.array(df_train), denisty_train])
            X_test, _, _ = encoder.predict([np.array(df_test), density_test])
            X_outlier, _, _ = encoder.predict([np.array(df_outlier), density_outlier])

            representations.append((X_train, X_test, X_outlier))

        fold += 1

    return representations


def init_metrics():
    metrics = {
        'precision': [],
        'recall': [],
        'f1-score': [],
        'auc_roc': [],
        'accuracy': [],
        'time': []
    }
    return metrics


def save_values(metricas, values):
    for key in metricas.keys():
        metricas[key].append(values[key])


def evaluation_one_class(preds_interest, preds_outliers):
    y_true = [1] * len(preds_interest) + [-1] * len(preds_outliers)
    y_pred = list(preds_interest) + list(preds_outliers)
    return classification_report(y_true, y_pred, output_dict=True)

def evaluate_model(X_train, X_test, X_outlier, model):

    one_class_classifier = model.fit(X_train)

    Y_pred_interest = one_class_classifier.predict(X_test)

    Y_pred_ruido = one_class_classifier.predict(X_outlier)

    score_interest = one_class_classifier.decision_function(X_test)

    score_outlier = one_class_classifier.decision_function(X_outlier)

    y_true = np.array([1] * len(X_test) + [-1] * len(X_outlier))

    fpr, tpr, treshold = roc_curve(y_true, np.concatenate([score_interest, score_outlier]))

    dic = evaluation_one_class(Y_pred_interest, Y_pred_ruido)

    metrics = {}
    metrics['precision'] = dic['1']['precision']
    metrics['recall'] = dic['1']['recall']
    metrics['f1-score'] = dic['1']['f1-score']
    metrics['auc_roc'] = roc_auc_score(y_true, np.concatenate([score_interest, score_outlier]))
    metrics['accuracy'] = dic['accuracy']

    return metrics, fpr, tpr


def evaluate_models(models, representations, file_name, line_parameters, path):

    for model in tqdm(models):

        lp = model + '_' + line_parameters
        fn = file_name + '_' + model.split('_')[0] + '.csv'
        metrics = init_metrics()

        for reps in representations:
            start = time.time()
            values, fpr, tpr = evaluate_model(reps[0], reps[1], reps[2], models[model])
            end = time.time()
            time_ = end - start
            values['time'] = time_

            save_values(metrics, values)

        write_results(metrics, fn, lp, path)


def write_results(metrics, file_name, line_parameters, path):
    if not Path(path + file_name).is_file():
        file_ = open(path + file_name, 'w')
        string = 'Parameters'

        for metric in metrics.keys():
            string += ';' + metric + '-mean;' + metric + '-std'
        string += '\n'

        file_.write(string)
        file_.close()

    file_ = open(path + file_name, 'a')
    string = line_parameters

    for metric in metrics.keys():
        string += ';' + str(np.mean(metrics[metric])) + ';' + str(np.std(metrics[metric]))

    string += '\n'
    file_.write(string)
    file_.close()


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


class MyTokenizer:
    def __init__(self, language):
        self.wnl = WordNetLemmatizer()
        if language == 'english':
            self.STOPWORDS = nltk.corpus.stopwords.words('english')
            self.stemmer = nltk.stem.SnowballStemmer('english')

        if language == 'italian':
            self.STOPWORDS = nltk.corpus.stopwords.words('italian')
            self.stemmer = nltk.stem.SnowballStemmer('italian')

        if language == 'multilingual':
            self.STOPWORDS = set(nltk.corpus.stopwords.words('italian')).union(
                set(nltk.corpus.stopwords.words('english')))
            self.stemmer = nltk.stem.SnowballStemmer('english')

    def __call__(self, doc):
        L1 = [t for t in word_tokenize(doc)]
        L2 = []
        for token in L1:
            if token not in self.STOPWORDS and token.isnumeric() is False and len(token) > 2 and hasNumbers(
                    token) is False:
                L2.append(token)
        L3 = [self.stemmer.stem(self.wnl.lemmatize(t)) for t in L2]
        return L3


def term_weight_type(bow_type, language_tokenizer='multilingual'):

    if bow_type == 'term-frequency-IDF':
        vectorizer = TfidfVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language_tokenizer))
    elif bow_type == 'binary':
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language_tokenizer), binary=True)
    else:
        vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(1, 1), min_df=1,
                                     tokenizer=MyTokenizer(language_tokenizer))

    return vectorizer


def make_density_information(cluster_list, df_train, df_test, df_outlier):
    L_X_train = []
    L_X_test = []
    L_X_outlier = []

    for cluster in cluster_list:
        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(df_train)

        X_train_temp = silhouette_samples(df_train, kmeans.labels_).reshape(len(df_train), 1)
        L_X_train.append(X_train_temp)

        X_test_temp = silhouette_samples(df_test, kmeans.predict(df_test)).reshape(len(df_test), 1)
        L_X_test.append(X_test_temp)

        X_outlier_temp = silhouette_samples(df_outlier, kmeans.predict(df_outlier)).reshape(len(df_outlier), 1)
        L_X_outlier.append(X_outlier_temp)

    return np.concatenate(L_X_train, axis=1), np.concatenate(L_X_test, axis=1), np.concatenate(L_X_outlier, axis=1)


def autoencoder(arq, input_length):
    encoder_inputs = Input(shape=(input_length,), name='encoder_input')

    if len(arq) == 3:
        first_dense_encoder = Dense(arq[0], activation="linear")(encoder_inputs)

        second_dense_encoder = Dense(arq[1], activation="linear")(first_dense_encoder)

        encoded = Dense(arq[2], activation="linear")(second_dense_encoder)

        first_dense_decoder = Dense(arq[1], activation="linear")(encoded)

        second_dense_decoder = Dense(arq[0], activation="linear")(first_dense_decoder)

        decoder_output = Dense(input_length, activation="linear")(second_dense_decoder)

    elif len(arq) == 2:
        first_dense_encoder = Dense(arq[0], activation="linear")(encoder_inputs)

        encoded = Dense(arq[1], activation="linear")(first_dense_encoder)

        first_dense_decoder = Dense(arq[0], activation="linear")(encoded)

        decoder_output = Dense(input_length, activation="linear")(first_dense_decoder)

    elif len(arq) == 1:
        encoded = Dense(arq[0], activation="linear")(encoder_inputs)

        decoder_output = Dense(input_length, activation="linear")(encoded)

    encoder = Model(encoder_inputs, encoded)

    autoencoder = Model(encoder_inputs, decoder_output)

    autoencoder.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='mse')

    return autoencoder, encoder


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, factor_multiply, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.factor_multiply = factor_multiply

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data, reconstruction)
            )
            reconstruction_loss *= self.factor_multiply
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }


def Encoder_VAE(arq, input_dim):
    encoder_inputs = keras.Input(shape=(input_dim,), name='encoder_input')

    if len(arq) == 3:
        first_dense = Dense(arq[0], activation="linear")(encoder_inputs)

        second_dense = Dense(arq[1], activation="linear")(first_dense)

        z_mean = layers.Dense(arq[2], name="Z_mean")(second_dense)
        z_log_var = layers.Dense(arq[2], name="Z_log_var")(second_dense)
        z = Sampling()([z_mean, z_log_var])

    if len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(encoder_inputs)

        z_mean = layers.Dense(arq[1], name="Z_mean")(first_dense)
        z_log_var = layers.Dense(arq[1], name="Z_log_var")(first_dense)
        z = Sampling()([z_mean, z_log_var])

    if len(arq) == 1:
        z_mean = layers.Dense(arq[0], name="Z_mean")(encoder_inputs)
        z_log_var = layers.Dense(arq[0], name="Z_log_var")(encoder_inputs)
        z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model([encoder_inputs], [z_mean, z_log_var, z], name="Encoder")

    return encoder


def Decoder_VAE(arq, output_dim):
    latent_inputs = keras.Input(shape=(arq[(len(arq) - 1)],), name='decoder_input')

    if len(arq) == 3:
        first_dense = Dense(arq[1], activation="linear")(latent_inputs)

        second_dense = Dense(arq[0], activation="linear")(first_dense)

        decoder_outputs = Dense(output_dim, activation="linear")(second_dense)

    if len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(latent_inputs)

        decoder_outputs = Dense(output_dim, activation="linear")(first_dense)

    if len(arq) == 1:
        decoder_outputs = Dense(output_dim, activation="linear")(latent_inputs)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder


def VariationalAutoencoder(arq, input_dim):
    encoder = Encoder_VAE(arq, input_dim)

    decoder = Decoder_VAE(arq, input_dim)

    vae = VAE(encoder, decoder, input_dim)

    vae.compile(optimizer=keras.optimizers.Adam())

    return vae, encoder, decoder


def multimodal_autoencoder(arq, first_input_len, second_input_len, operator):
    first_input = Input(shape=(first_input_len,), name='first_input_encoder')

    second_input = Input(shape=(second_input_len,), name='second_input_encoder')

    l1 = Dense(np.max([first_input_len, second_input_len]), activation='linear')(first_input)
    l2 = Dense(np.max([first_input_len, second_input_len]), activation='linear')(second_input)

    fusion = None
    if operator == 'concatenate':
        fusion = concatenate([l1, l2])
    if operator == 'multiply':
        fusion = multiply([l1, l2])
    if operator == 'average':
        fusion = average([l1, l2])
    if operator == 'subtract':
        fusion = subtract([l1, l2])
    if operator == 'add':
        fusion = add([l1, l2])

    if len(arq) == 3:
        first_dense_encoder = Dense(arq[0], activation="linear")(fusion)

        second_dense_encoder = Dense(arq[1], activation="linear")(first_dense_encoder)

        encoded = Dense(arq[2], activation="linear")(second_dense_encoder)

        first_dense_decoder = Dense(arq[1], activation="linear")(encoded)

        second_dense_decoder = Dense(arq[0], activation="linear")(first_dense_decoder)

        first_decoder_output = Dense(first_input_len, activation="linear")(second_dense_decoder)

        second_decoder_output = Dense(second_input_len, activation="linear")(second_dense_decoder)

    if len(arq) == 2:
        first_dense_encoder = Dense(arq[0], activation="linear")(fusion)

        encoded = Dense(arq[1], activation="linear")(first_dense_encoder)

        first_dense_decoder = Dense(arq[0], activation="linear")(encoded)

        first_decoder_output = Dense(first_input_len, activation="linear")(first_dense_decoder)

        second_decoder_output = Dense(second_input_len, activation="linear")(first_dense_decoder)

    if len(arq) == 1:
        encoded = Dense(arq[0], activation="linear")(fusion)

        first_decoder_output = Dense(first_input_len, activation="linear")(encoded)

        second_decoder_output = Dense(second_input_len, activation="linear")(encoded)

    encoder = Model([first_input, second_input], encoded)

    autoencoder = Model([first_input, second_input], [first_decoder_output, second_decoder_output])

    autoencoder.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='mse')

    return autoencoder, encoder


class MVAE(keras.Model):
    def __init__(self, encoder, decoder, factor_multiply_embedding, factor_multiply_density, **kwargs):
        super(MVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.factor_multiply_embedding = factor_multiply_embedding
        self.factor_multiply_density = factor_multiply_density

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder((data[0], data[1]))

            reconstruction = self.decoder(z)

            embedding_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[0], reconstruction[0])
            )

            embedding_loss *= self.factor_multiply_embedding

            density_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[1], reconstruction[1])
            )

            density_loss *= self.factor_multiply_density

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = embedding_loss + density_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "tota loss": total_loss,
            "embedding loss": embedding_loss,
            "denisty loss": density_loss,
            "kl loss": kl_loss,
        }


def Encoder_MVAE(arq, embedding_dim, density_dim, operator):
    embedding_inputs = keras.Input(shape=(embedding_dim,), name='first_input_encoder')
    density_inputs = keras.Input(shape=(density_dim,), name='second_input_encoder')

    l1 = Dense(np.max([embedding_dim, density_dim]), activation='linear')(embedding_inputs)
    l2 = Dense(np.max([embedding_dim, density_dim]), activation='linear')(density_inputs)

    fusion = None
    if operator == 'concatenate':
        fusion = concatenate([l1, l2])
    if operator == 'multiply':
        fusion = multiply([l1, l2])
    if operator == 'average':
        fusion = average([l1, l2])
    if operator == 'subtract':
        fusion = subtract([l1, l2])
    if operator == 'add':
        fusion = add([l1, l2])

    if len(arq) == 3:
        first_dense = Dense(arq[0], activation="linear")(fusion)

        second_dense = Dense(arq[1], activation="linear")(first_dense)

        z_mean = layers.Dense(arq[2], name="Z_mean")(second_dense)
        z_log_var = layers.Dense(arq[2], name="Z_log_var")(second_dense)
        z = Sampling()([z_mean, z_log_var])

    if len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(fusion)

        z_mean = layers.Dense(arq[1], name="Z_mean")(first_dense)
        z_log_var = layers.Dense(arq[1], name="Z_log_var")(first_dense)
        z = Sampling()([z_mean, z_log_var])

    if len(arq) == 1:
        z_mean = layers.Dense(arq[0], name="Z_mean")(fusion)
        z_log_var = layers.Dense(arq[0], name="Z_log_var")(fusion)
        z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model([embedding_inputs, density_inputs], [z_mean, z_log_var, z], name="encoder")

    return encoder


def Decoder_MVAE(arq, embedding_dim, density_dim):
    latent_inputs = keras.Input(shape=(arq[(len(arq) - 1)],), name='input_decoder')

    if len(arq) == 3:
        first_dense = Dense(arq[1], activation="linear")(latent_inputs)

        second_dense = Dense(arq[0], activation="linear")(first_dense)

        embedding_outputs = Dense(embedding_dim, activation="linear")(second_dense)

        density_outputs = Dense(density_dim, activation="linear")(second_dense)

    if len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(latent_inputs)

        embedding_outputs = Dense(embedding_dim, activation="linear")(first_dense)

        density_outputs = Dense(density_dim, activation="linear")(first_dense)

    if len(arq) == 1:
        embedding_outputs = Dense(embedding_dim, activation="linear")(latent_inputs)

        density_outputs = Dense(density_dim, activation="linear")(latent_inputs)

    decoder = keras.Model(latent_inputs, [embedding_outputs, density_outputs], name="decoder")

    return decoder


def MultimodalVAE(arq, embedding_dim, density_dim, operator):
    encoder = Encoder_MVAE(arq, embedding_dim, density_dim, operator)

    decoder = Decoder_MVAE(arq, embedding_dim, density_dim)

    mvae = MVAE(encoder, decoder, embedding_dim, density_dim)

    mvae.compile(optimizer=keras.optimizers.Adam())

    return mvae, encoder, decoder


def run(datasets, models, preprocessing):

    # Parameters
    path_results = './results/'
    percents = [0.25, 0.50, 0.75, 1]
    folds = 10
    kf = foldValidation(folds)
    line_parameters = ''
    term_weight_list = ['term-frequency-IDF', 'term-frequency', 'binary']
    cluster_matrix = [[3, 6, 7, 8], [2, 3, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9, 10]]
    epochs = [5, 10, 50]
    arqs = [[384, 128], [256, 128], [256]]
    path_ae_models = './Autoencoder/'
    path_vae_models = './VAE/'
    path_mae_models = './MAE/'
    path_mvae_models = './MVAE/'
    operators = ['concatenate', 'multiply', 'average', 'subtract', 'add']

    for dataset in tqdm(datasets.keys()):

        df = datasets[dataset]

        df_int, df_outiler = interest_outlier(df, class_outlier='irr')

        for percent in tqdm(percents):

            if preprocessing == 'Maalej':

                train_test_interest = train_test_split_one_class(kf, df_int, 'Maalej Features', percent)

                representations = make_representation(train_test_interest,
                                                      np.array(df_outiler['Maalej Features'].to_list()), preprocessing)

                file_name = dataset + '_' + preprocessing + '_' + str(percent)

                evaluate_models(models, representations, file_name, line_parameters, path_results)

                del representations
                del train_test_interest
                collect()

            elif preprocessing == 'FastText':

                train_test_interest = train_test_split_one_class(kf, df_int, 'Maalej Features', percent)

                representations = make_representation(train_test_interest,
                                                      np.array(df_outiler['Maalej Features'].to_list()), preprocessing)

                file_name = dataset + '_' + preprocessing + '_' + str(percent)

                evaluate_models(models, representations, file_name, line_parameters, path_results)

                del representations
                del train_test_interest
                collect()

            elif preprocessing == 'BoW':

                for term_weight in term_weight_list:
                    print('Term Wheight: ' + term_weight)
                    vectorizer = term_weight_type(term_weight)

                    train_test_interest = train_test_split_one_class(kf, df_int, 'text', percent)

                    representations = make_representation(train_test_interest, np.array(df_outiler['text'].to_list()),
                                                          preprocessing, vectorizer=vectorizer)

                    file_name = dataset + '_' + preprocessing + '_' + term_weight + '_' + str(percent)

                    evaluate_models(models, representations, file_name, line_parameters, path_results)

                    del representations
                    del train_test_interest
                    collect()

            elif preprocessing == 'DBERTML':

                train_test_interest = train_test_split_one_class(kf, df_int, 'DistilBERT Multilingual', percent)

                representations = make_representation(train_test_interest,
                                                      np.array(df_outiler['DistilBERT Multilingual'].to_list()),
                                                      preprocessing)

                file_name = dataset + '_' + preprocessing + '_' + str(percent)

                evaluate_models(models, representations, file_name, line_parameters, path_results)

                del representations
                del train_test_interest
                collect()

            elif preprocessing == 'Density':

                for cluster_list in cluster_matrix:
                    print('Cluster List: ' + str(cluster_list))
                    train_test_interest = train_test_split_one_class(kf, df_int, 'DistilBERT Multilingual', percent)

                    representations = make_representation(train_test_interest,
                                                          np.array(df_outiler['DistilBERT Multilingual'].to_list()),
                                                          preprocessing, cluster_list=cluster_list)

                    file_name = dataset + '_' + preprocessing + '_' + str(percent)

                    line_parameters = str(cluster_list)

                    evaluate_models(models, representations, file_name, line_parameters, path_results)

                    del representations
                    del train_test_interest
                    collect()

            elif preprocessing == 'AE':

                for epoch in epochs:
                    print('Epoch: ' + str(epoch))
                    for arq in arqs:

                        arq_name = str(arq).replace('[', '(')
                        arq_name = arq_name.replace(']', ')')
                        print('Arq: ' + arq_name)

                        model_file_name = dataset + '_' + str(epoch) + '_' + arq_name + '_' + str(percent)

                        parameter_list_AE = (epoch, arq, model_file_name, path_ae_models)

                        train_test_interest = train_test_split_one_class(kf, df_int, 'DistilBERT Multilingual', percent)

                        representations = make_representation(train_test_interest,
                                                              np.array(df_outiler['DistilBERT Multilingual'].to_list()),
                                                              preprocessing, parameter_list_AE=parameter_list_AE)

                        result_file_name = dataset + '_' + preprocessing + '_' + str(percent)

                        line_parameters = str(epoch) + '_' + str(arq)

                        evaluate_models(models, representations, result_file_name, line_parameters, path_results)

                        del representations
                        del train_test_interest
                        collect()

            elif preprocessing == 'VAE':

                for epoch in epochs:
                    print('Epoch: ' + str(epoch))
                    for arq in arqs:
                        arq_name = str(arq).replace('[', '(')
                        arq_name = arq_name.replace(']', ')')
                        print('Arq: ' + arq_name)

                        model_file_name = dataset + '_' + str(epoch) + '_' + arq_name + '_' + str(percent)

                        parameter_list_VAE = (epoch, arq, model_file_name, path_vae_models)

                        train_test_interest = train_test_split_one_class(kf, df_int, 'DistilBERT Multilingual', percent)

                        representations = make_representation(train_test_interest,
                                                              np.array(df_outiler['DistilBERT Multilingual'].to_list()),
                                                              preprocessing, parameter_list_VAE=parameter_list_VAE)

                        result_file_name = dataset + '_' + preprocessing + '_' + str(percent)

                        line_parameters = str(epoch) + '_' + str(arq)

                        evaluate_models(models, representations, result_file_name, line_parameters, path_results)

                        del representations
                        del train_test_interest
                        collect()

            elif preprocessing == 'MAE':

                for epoch in epochs:
                    print('Epoch: ' + str(epoch))
                    for arq in arqs:

                        arq_name = str(arq).replace('[', '(')
                        arq_name = arq_name.replace(']', ')')
                        print('Arq: ' + arq_name)

                        for operator in operators:
                            print('Operator: ' + operator)
                            for cluster_list in cluster_matrix:
                                cluster_name = str(cluster_list).replace('[', '(')
                                cluster_name = cluster_name.replace(']', ')')

                                print('Cluster List: ' + cluster_name)

                                model_file_name = dataset + '_' + str(
                                    epoch) + '_' + arq_name + '_' + operator + '_' + cluster_name + '_' + str(percent)

                                parameter_list_MAE = (epoch, arq, model_file_name, path_mae_models, operator)

                                train_test_interest = train_test_split_one_class(kf, df_int, 'DistilBERT Multilingual',
                                                                                 percent)

                                representations = make_representation(train_test_interest, np.array(
                                    df_outiler['DistilBERT Multilingual'].to_list()), preprocessing,
                                                                      cluster_list=cluster_list,
                                                                      parameter_list_MAE=parameter_list_MAE)

                                result_file_name = dataset + '_' + preprocessing + '_' + str(percent)

                                line_parameters = str(epoch) + '_' + str(arq) + '_' + str(cluster_list) + '_' + str(
                                    operator)

                                evaluate_models(models, representations, result_file_name, line_parameters,
                                                path_results)

                                del representations
                                del train_test_interest
                                collect()

            elif preprocessing == 'MVAE':

                for epoch in epochs:
                    print('Epoch: ' + str(epoch))
                    for arq in arqs:

                        arq_name = str(arq).replace('[', '(')
                        arq_name = arq_name.replace(']', ')')
                        print('Arq: ' + arq_name)

                        for operator in operators:
                            print('Operator: ' + operator)
                            for cluster_list in cluster_matrix:
                                cluster_name = str(cluster_list).replace('[', '(')
                                cluster_name = cluster_name.replace(']', ')')

                                print('Cluster List: ' + cluster_name)

                                model_file_name = dataset + '_' + str(
                                    epoch) + '_' + arq_name + '_' + operator + '_' + cluster_name + '_' + str(percent)

                                parameter_list_MVAE = (epoch, arq, model_file_name, path_mvae_models, operator)

                                train_test_interest = train_test_split_one_class(kf, df_int, 'DistilBERT Multilingual',
                                                                                 percent)

                                representations = make_representation(train_test_interest, np.array(
                                    df_outiler['DistilBERT Multilingual'].to_list()), preprocessing,
                                                                      cluster_list=cluster_list,
                                                                      parameter_list_MVAE=parameter_list_MVAE)

                                result_file_name = dataset + '_' + preprocessing + '_' + str(percent)

                                line_parameters = str(epoch) + '_' + str(arq) + '_' + str(cluster_list) + '_' + str(
                                    operator)

                                evaluate_models(models, representations, result_file_name, line_parameters,
                                                path_results)

                                del representations
                                del train_test_interest
                                collect()

        del df
        del df_int
        del df_outiler
        collect()


if __name__ == '__main__':

    os.system("./download.sh")

    datasets = {
        'ARE': pd.read_pickle('datasets/ARE.plk'),
        'TEN': pd.read_pickle('datasets/TEN.plk'),
        'TIT': pd.read_pickle('datasets/TIN.plk')
    }

    models = {
        'SVDD_Linear_scale': SVDD(kernel='linear', gamma='scale'),
        'SVDD_Linear_auto': SVDD(kernel='linear', gamma='auto'),
        'SVDD_Sigmoid_scale': SVDD(kernel='sigmoid', gamma='scale'),
        'SVDD_Sigmoid_auto': SVDD(kernel='sigmoid', gamma='auto'),
        'SVDD_Poly_2_scale': SVDD(kernel='poly', degree=2, gamma='scale'),
        'SVDD_Poly_2_auto': SVDD(kernel='poly', degree=2, gamma='auto'),
        'SVDD_Poly_3_scale': SVDD(kernel='poly', degree=3, gamma='scale'),
        'SVDD_Poly_3_auto': SVDD(kernel='poly', degree=3, gamma='auto'),
        'SVDD_Poly_4_scale': SVDD(kernel='poly', degree=4, gamma='scale'),
        'SVDD_Poly_4_auto': SVDD(kernel='poly', degree=4, gamma='auto'),
        'SVDD_RBF_0.001_scale': SVDD(kernel='rbf', nu=0.001, gamma='scale'),
        'SVDD_RBF_0.01_scale': SVDD(kernel='rbf', nu=0.01, gamma='scale'),
        'SVDD_RBF_0.05_scale': SVDD(kernel='rbf', nu=0.05, gamma='scale'),
        'SVDD_RBF_0.1_scale': SVDD(kernel='rbf', nu=0.1, gamma='scale'),
        'SVDD_RBF_0.2_scale': SVDD(kernel='rbf', nu=0.2, gamma='scale'),
        'SVDD_RBF_0.3_scale': SVDD(kernel='rbf', nu=0.3, gamma='scale'),
        'SVDD_RBF_0.4_scale': SVDD(kernel='rbf', nu=0.4, gamma='scale'),
        'SVDD_RBF_0.5_scale': SVDD(kernel='rbf', nu=0.5, gamma='scale'),
        'SVDD_RBF_0.6_scale': SVDD(kernel='rbf', nu=0.6, gamma='scale'),
        'SVDD_RBF_0.7_scale': SVDD(kernel='rbf', nu=0.7, gamma='scale'),
        'SVDD_RBF_0.8_scale': SVDD(kernel='rbf', nu=0.8, gamma='scale'),
        'SVDD_RBF_0.9_scale': SVDD(kernel='rbf', nu=0.9, gamma='scale'),
        'SVDD_RBF_0.001_auto': SVDD(kernel='rbf', nu=0.001, gamma='auto'),
        'SVDD_RBF_0.01_auto': SVDD(kernel='rbf', nu=0.01, gamma='auto'),
        'SVDD_RBF_0.05_auto': SVDD(kernel='rbf', nu=0.05, gamma='auto'),
        'SVDD_RBF_0.1_auto': SVDD(kernel='rbf', nu=0.1, gamma='auto'),
        'SVDD_RBF_0.2_auto': SVDD(kernel='rbf', nu=0.2, gamma='auto'),
        'SVDD_RBF_0.3_auto': SVDD(kernel='rbf', nu=0.3, gamma='auto'),
        'SVDD_RBF_0.4_auto': SVDD(kernel='rbf', nu=0.4, gamma='auto'),
        'SVDD_RBF_0.5_auto': SVDD(kernel='rbf', nu=0.5, gamma='auto'),
        'SVDD_RBF_0.6_auto': SVDD(kernel='rbf', nu=0.6, gamma='auto'),
        'SVDD_RBF_0.7_auto': SVDD(kernel='rbf', nu=0.7, gamma='auto'),
        'SVDD_RBF_0.8_auto': SVDD(kernel='rbf', nu=0.8, gamma='auto'),
        'SVDD_RBF_0.9_auto': SVDD(kernel='rbf', nu=0.9, gamma='auto')
    }

    preprocessing = sys.argv[1]

    run(datasets, models, preprocessing)
