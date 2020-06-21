import pandas as pd
from sklearn.model_selection import train_test_split

def split_human_data(args):
    ### Splitting into train and test text files for human faces

    x = pd.read_csv("/content/drive/My Drive/CSC420Project/human.txt", sep=" ", header=None)[0].values.tolist()
    x = x[:32467] #shuffle data if you want randomeness 
    X_train, X_test= train_test_split(x, test_size=0.3, random_state=420)

    with open(args.human_train_path, "w") as f:
        for item in X_train:
            f.write(item + "\n")

    with open(args.human_test_path, "w") as f:
        for item in X_test:
            f.write(item + "\n")

def split_cartoon_data(args):
    ### Splitting into train and test text files for cartoon faces

    x = pd.read_csv("/content/drive/My Drive/CSC420Project/cartoon.txt", sep=" ", header=None)[0].values.tolist()
    x = x[:100000] #shuffle data if you want randomeness 
    X_train, X_test= train_test_split(x, test_size=0.3, random_state=420)

    with open(args.cartoon_train_path, "w") as f:
        for item in X_train:
            f.write(item + "\n")

    with open(args.cartoon_test_path, "w") as f:
        for item in X_test:
            f.write(item + "\n")