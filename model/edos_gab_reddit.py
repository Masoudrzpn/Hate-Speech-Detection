# Importing libraries
import pandas as pd
import random as python_random
import numpy as np
import emoji
import argparse
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,  classification_report
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import plot_confusion_matrix
import tensorflow
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")
# Make reproducible as much as possible
np.random.seed(1234)
tensorflow.random.set_seed(1234)
python_random.seed(1234)


def create_arg_parser():
    """This function creates all command arguments, for data input,
    model selection, and custom parameters, please see the help sections
    for a detailed description."""
    parser = argparse.ArgumentParser()
    # Data input arguments
    parser.add_argument("-d1", "--data_file1", default="EDOS_Gab_Reddit_train.csv",
                        type=str, help="Our train set")
    parser.add_argument("-d2", "--data_file2", default="EDOS_Gab_Reddit_dev.csv",
                        type=str, help="Our dev set")
    parser.add_argument("-d3_1", "--data_file3_1", default="Qian_Gab_test.csv",
                        type=str, help="Our test set")
    parser.add_argument("-d3_2", "--data_file3_2", default="Qian_Reddit_test.csv",
                        type=str, help="Our test set")
    parser.add_argument("-d3_3", "--data_file3_3", default="Gao_Fox_test.csv",
                        type=str, help="Our test set")
    parser.add_argument("-d3_4", "--data_file3_4", default="GHC_Gab_test.csv",
                        type=str, help="Our test set")
    parser.add_argument("-d3_5", "--data_file3_5", default="EDOS_Gab_Reddit_test.csv",
                        type=str, help="Our test set")
    parser.add_argument("-d3_6", "--data_file3_6", default="HateCheck.csv",
                        type=str, help="Our test set")
    parser.add_argument("-d3_7", "--data_file3_7", default="HateCheck_0.csv",
                        type=str, help="Our test set")
    parser.add_argument("-d3_8", "--data_file3_8", default="HateCheck_1.csv",
                        type=str, help="Our test set")
    parser.add_argument("-d3_9", "--data_file3_9", default="HateCheck_women.csv",
                        type=str, help="Our test set")


    # Model arguments
    parser.add_argument("-tf", "--transformer", default="GroNLP/hateBERT",
                        type=str,
                        help="this argument takes the pretrained language "
                             "model URL from HuggingFace default is HateBERT, "
                             "please visit HuggingFace for full URL")
    # Parameter arguments
    parser.add_argument("-lr", "--learn_rate", default=1e-5, type=float,
                        help="Set a custom learn rate for "
                             "the pretrained language model, default is 1e-5")
    parser.add_argument("-bs", "--batch_size", default=8, type=int,
                        help="Set a custom batch size for "
                             "the pretrained language model, default is 8")
    parser.add_argument("-sl", "--sequence_length", default=58, type=int,
                        help="Set a custom maximum sequence length for "
                             "the pretrained language model, default is 58")
    parser.add_argument("-epoch", "--epochs", default=8, type=int,
                        help="This argument selects the amount of epochs "
                             "to run the model with, default is 8 epoch")

    args = parser.parse_args()
    return args


def read_data(d1, d2, d3_1, d3_2, d3_3, d3_4, d3_5, d3_6, d3_7, d3_8, d3_9):
    """Reading in both datasets and returning them as pandas dataframes
    with only the text and labels."""
    # read in data to pandas
    df1 = pd.read_csv(d1)
    df2 = pd.read_csv(d2)
    df3_1 = pd.read_csv(d3_1)
    df3_2 = pd.read_csv(d3_2)
    df3_3 = pd.read_csv(d3_3)
    df3_4 = pd.read_csv(d3_4)
    df3_5 = pd.read_csv(d3_5)
    df3_6 = pd.read_csv(d3_6)
    df3_7 = pd.read_csv(d3_7)
    df3_8 = pd.read_csv(d3_8)
    df3_9 = pd.read_csv(d3_9)

    # select the columns in our datasets
    df1.columns = ['label', 'text']
    df2.columns = ['label', 'text']
    df3_1.columns = ['label', 'text']
    df3_2.columns = ['label', 'text']
    df3_3.columns = ['label', 'text']
    df3_4.columns = ['label', 'text']
    df3_5.columns = ['label', 'text']
    df3_6.columns = ['label', 'text']
    df3_7.columns = ['label', 'text']
    df3_8.columns = ['label', 'text']
    df3_9.columns = ['label', 'text']


    return df1, df2, df3_1, df3_2, df3_3, df3_4, df3_5, df3_6, df3_7, df3_8, df3_9


def preprocess(text):
    """Removes hashtags and converts links to [URL] and usernames starting
    with @ to [USER], it also converts emojis to their textual form."""
    documents = []
    for instance in text:
        instance = re.sub(r'@([^ ]*)', '[USER]', instance)
        instance = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.'
                          r'([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
                          '[URL]', instance)
        instance = emoji.demojize(instance)
        instance = instance.replace('#', '')
        documents.append(instance)
    return documents


def train_transformer(lm, epoch, bs, lr, sl, X_train, Y_train, X_dev, Y_dev):
    """This function takes as input the train file, dev file, model name,
    and parameters. It trains the model with the specified parameters and
    returns the trained model."""
    print("Training model: {}\nWith parameters:\nLearn rate: {}, "
          "Batch size: {}\nEpochs: {}, Sequence length: {}"
          .format(lm, lr, bs, epoch, sl))

    # Selecting the correct tokenizer for the model, and selecting the model
    tokenizer = AutoTokenizer.from_pretrained(lm)
    model = TFAutoModelForSequenceClassification.from_pretrained(lm,
                                                                 num_labels=2,
                                                                 from_pt=True)

    # Tokenzing the train and dev texts
    tokens_train = tokenizer(X_train, padding=True, max_length=sl,
                             truncation=True, return_tensors="np").data
    tokens_dev = tokenizer(X_dev, padding=True, max_length=sl,
                           truncation=True, return_tensors="np").data

    # Setting the loss function for binary task and optimization function
    loss_function = BinaryCrossentropy(from_logits=True)
    optim = Adam(learning_rate=lr)

    # Early stopping
    early_stopper = tensorflow.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True,
        mode="auto")
    # Encoding the labels with sklearns LabelBinazrizer
    encoder = LabelBinarizer()
    Y_train = encoder.fit_transform(Y_train)
    Y_dev = encoder.fit_transform(Y_dev)
    Y_train_bin = np.hstack((1 - Y_train, Y_train))
    Y_dev_bin = np.hstack((1 - Y_dev, Y_dev))

    # Compiling the model and training it with the given parameter settings
    model.compile(loss=loss_function, optimizer=optim, metrics=["accuracy"])
    model.fit(tokens_train, Y_train_bin, verbose=1, epochs=epoch,
              batch_size=bs, validation_data=(tokens_dev, Y_dev_bin),
              callbacks=[early_stopper])
    return model


def test_transformer(lm, epoch, bs, lr, sl, model, X_test, Y_test, ident):
    """This function takes as input the trained transformer model, name of
    the model, parameters, and the test files, and predicts the labels
    for the test set and returns the accuracy score with a summarization
    of the model."""
    print(
        "Testing model: {} on {} set\nWith parameters:\nLearn rate: {}, "
        "Batch size: {}\nEpochs: {}, Sequence length: {}"
        .format(lm, ident, lr, bs, epoch, sl))

    # Selecting the correct tokenizer for the model,
    # and applying it to the test set
    tokenizer = AutoTokenizer.from_pretrained(lm)
    tokens_test = tokenizer(X_test, padding=True, max_length=sl,
                            truncation=True, return_tensors="np").data

    # Getting predicitions on the test set and converting
    # the logits to sigmoid probabilities (binary)
    Y_pred = model.predict(tokens_test)["logits"]
    prob = tensorflow.nn.softmax(Y_pred)

    # converting gold labels with LabelBinarizer
    encoder = LabelBinarizer()
    Y_test = encoder.fit_transform(Y_test)
    Y_test_bin = np.hstack((1 - Y_test, Y_test))

    # Converting the predicitions and gold set
    # to their original numerical label value
    pred = np.argmax(prob, axis=1)
    gold = np.argmax(Y_test_bin, axis=1)

    # Printing classification report (rounding on 3 decimals)
    print("Classification Report on {} set:\n{}"
          .format(ident, classification_report(gold, pred, digits=3)))
    return gold, pred


def main():
    """Main function to train and finetune pretrained language models"""
    # Create the command arguments for the script
    args = create_arg_parser()

    # Creating parameter variables
    lr = args.learn_rate
    bs = args.batch_size
    sl = args.sequence_length
    ep = args.epochs

    # Reading data
    ori_train, ori_dev, ori_test_3_1, ori_test_3_2, ori_test_3_3, ori_test_3_4, ori_test_3_5, ori_test_3_6, ori_test_3_7, ori_test_3_8, ori_test_3_9 = read_data(
                                                               d1=args.data_file1, 
                                                               d2=args.data_file2, 
                                                               d3_1=args.data_file3_1, 
                                                               d3_2=args.data_file3_2, 
                                                               d3_3=args.data_file3_3, 
                                                               d3_4=args.data_file3_4, 
                                                               d3_5=args.data_file3_5, 
                                                               d3_6=args.data_file3_6,
                                                               d3_7=args.data_file3_7,
                                                               d3_8=args.data_file3_8,
                                                               d3_9=args.data_file3_9
                                                               )

    X_train = preprocess(ori_train['text'].tolist())
    Y_train = ori_train['label'].tolist()

    X_dev = preprocess(ori_dev['text'].tolist())
    Y_dev = ori_dev['label'].tolist()

    X_test_3_1 = preprocess(ori_test_3_1['text'].tolist())
    Y_test_3_1 = ori_test_3_1['label'].tolist()

    X_test_3_2 = preprocess(ori_test_3_2['text'].tolist())
    Y_test_3_2 = ori_test_3_2['label'].tolist()

    X_test_3_3 = preprocess(ori_test_3_3['text'].tolist())
    Y_test_3_3 = ori_test_3_3['label'].tolist()

    X_test_3_4 = preprocess(ori_test_3_4['text'].tolist())
    Y_test_3_4 = ori_test_3_4['label'].tolist()

    X_test_3_5 = preprocess(ori_test_3_5['text'].tolist())
    Y_test_3_5 = ori_test_3_5['label'].tolist()

    X_test_3_6 = preprocess(ori_test_3_6['text'].tolist())
    Y_test_3_6 = ori_test_3_6['label'].tolist()

    X_test_3_7 = preprocess(ori_test_3_7['text'].tolist())
    Y_test_3_7 = ori_test_3_7['label'].tolist()

    X_test_3_8 = preprocess(ori_test_3_8['text'].tolist())
    Y_test_3_8 = ori_test_3_8['label'].tolist()

    X_test_3_9 = preprocess(ori_test_3_9['text'].tolist())
    Y_test_3_9 = ori_test_3_9['label'].tolist()


    # Running model
    print('Running HateBERT:')

    HateBERT = train_transformer(args.transformer, ep, bs, lr, sl,
                                 X_train, Y_train, X_dev, Y_dev)
    
    # HateBert.save_weights()

    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_test_3_1, Y_test_3_1, "test")
    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_test_3_2, Y_test_3_2, "test")
    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_test_3_3, Y_test_3_3, "test")
    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_test_3_4, Y_test_3_4, "test")
    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_test_3_5, Y_test_3_5, "test")
    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_test_3_6, Y_test_3_6, "test")
    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_test_3_7, Y_test_3_7, "test")
    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_test_3_8, Y_test_3_8, "test")
    test_transformer(args.transformer, ep, bs, lr, sl, HateBERT,
                     X_test_3_9, Y_test_3_9, "test")
    
if __name__ == '__main__':
    main()
