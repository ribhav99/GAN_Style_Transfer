import pandas as pd
from sklearn.model_selection import train_test_split

x = pd.read_csv("/Users/gerald/Desktop/GAN_Style_Transfer/data/cartoon.txt", sep=" ", header=None)[0].values.tolist()
X_train, X_test= train_test_split(x, test_size=0.3, random_state=420)

print(len(X_train))
with open("/Users/gerald/Desktop/GAN_Style_Transfer/data/cartoon_train.txt", "w") as f:
    for item in X_train:
        f.write(item + "\n")

with open("/Users/gerald/Desktop/GAN_Style_Transfer/data/cartoon_test.txt", "w") as f:
    for item in X_test:
        f.write(item + "\n")
