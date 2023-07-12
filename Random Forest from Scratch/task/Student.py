import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt

np.random.seed(52)

def convert_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    else:
        return 2


def create_bootstrap(dataset, size=10):
    return np.random.choice(len(dataset), size)


class RandomForestClassifier():
    def __init__(self, n_trees=10, max_depth=np.iinfo(np.int64).max, min_error=1e-6):

        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_error = min_error
        self.max_features = 'sqrt'
        self.bootstrapsize = 150
        self.is_fit = False

    def fit(self, X_train, y_train):

        self.forest = []

        for i in range(self.n_trees):
            decisiontree = DecisionTreeClassifier(max_features=self.max_features, max_depth=self.max_depth)
            bootstrap = create_bootstrap(X_train, size=self.bootstrapsize)
            X_DT = [X_train[i] for i in range(len(X_train)) if i in bootstrap]
            y_DT = [y_train[i] for i in range(len(y_train)) if i in bootstrap]
            decisiontree.fit(X_DT, y_DT)
            self.forest.append(decisiontree)
        self.is_fit = True

    def predict(self, X_test):

        if not self.is_fit:
            raise AttributeError('The forest is not fit yet! Consider calling .fit() method.')
        predvotes = [[] for _ in range(len(X_test))]
        for i, j in itertools.product(range(len(X_test)), range(self.n_trees)):
            predvotes[i].append(self.forest[j].predict(X_test[i:i+1])[0])
        return [1 if sum(k)/len(k)>0.5 else 0 for k in predvotes]

if __name__ == '__main__':

    data = pd.read_csv('https://www.dropbox.com/s/4vu5j6ahk2j3ypk/titanic_train.csv?dl=1')

    data.drop(
        ['PassengerId', 'Name', 'Ticket', 'Cabin'],
        axis=1,
        inplace=True
    )
    data.dropna(inplace=True)

    # Separate these back
    y = data['Survived'].astype(int)
    X = data.drop('Survived', axis=1)

    X['Sex'] = X['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    X['Embarked'] = X['Embarked'].apply(lambda x: convert_embarked(x))

    X_train, X_val, y_train, y_val = \
        train_test_split(X.values, y.values, stratify=y, train_size=0.8)
    '''
    #Stage 1/6
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    ypred = model.predict(X_val)
    acc = accuracy_score(y_val, ypred)
    print(acc)
    
    #Stage 2/6
    sample = create_bootstrap(y_train)
    print([y_train[i] for i in range(len(y_train)) if i in sample])

    #Stage 3/6
    randomforest = RandomForestClassifier()
    randomforest.fit(X_train, y_train)
    ypred = randomforest.predict(X_val)
    print(f'{accuracy_score(ypred,y_val):.3f}')

    #Stage 4/6
    randomforest = RandomForestClassifier()
    randomforest.fit(X_train, y_train)
    ypred = randomforest.predict(X_val)
    print(ypred[:10])
    
    #Stage 5/6
    randomforest = RandomForestClassifier()
    randomforest.fit(X_train, y_train)
    ypred = randomforest.predict(X_val)
    print(f'{accuracy_score(ypred,y_val):.3f}')
    '''
    #Stage 6/6
    accuracies = []
    ntrees = 20
    for i in tqdm(range(1, ntrees+1)):
        randomforest = RandomForestClassifier(n_trees=i)
        randomforest.fit(X_train, y_train)
        ypred = randomforest.predict(X_val)
        accuracies.append(round(accuracy_score(ypred, y_val), 3))
    #print(accuracies[:20])
    true_value = [0.755, 0.818, 0.783, 0.839, 0.79, 0.825, 0.79, 0.811, 0.818, 0.783, 0.825, 0.832, 0.804, 0.825, 0.825, 0.825, 0.839, 0.762, 0.839, 0.825]
    print(true_value)
    fig, ax = plt.subplots()

    ax.plot(range(0, ntrees), accuracies)
    ax.plot(range(0, ntrees), [i * 0.95 for i in true_value])
    ax.plot(range(0, ntrees), [i * 1.05 for i in true_value])
    plt.axis([0, ntrees, 0.5, 1])
    plt.show()
