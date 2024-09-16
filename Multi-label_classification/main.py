from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import *
import random
seed =0
random.seed(seed)
np.random.seed(seed)


def load_data():
    # load the input data
    df = get_input_data()
    return  df

def preprocess_data(df):
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)
    # translate data to english
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame, test_size):
    return Data(X, df, test_size)

def perform_modelling(data):
    model, predictions = model_predict(data)
    model_evaluate(model, data)
    return predictions


if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    X, df = get_embeddings(df)
    data = get_data_object(X, df, test_size=Config.TEST_SIZE)
    binary_predictions = perform_modelling(data)

    # print(binary_predictions)
    # print(data.classes)
    # print(data.count_classes)

