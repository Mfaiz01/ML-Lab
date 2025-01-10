# ML-Lab


1. Implement and demonstrate the find-s algorithm for finding the most specific hypothesis based on a given set of training data samples. Read the training data from a.csv file.

S1:
import csv

h=['0'for i in range(6)]
with open("D:/traininddata.csv") as f:
    data=csv.reader(f)
    data=list(data)
    
    for i in data:
        if i[-1]=="Yes":
            for j in range(6):
                if h[j]=='0':
                    h[j]=i[j]
                elif h[j]!=i[j]:
                    h[j]='?'

    print(h)

s2:

import csv
hypo = ['%','%','%','%','%','%'];

with open('D:/traininddata.csv') as csv_file:
    readcsv = csv.reader(csv_file, delimiter=',')
    print(readcsv)

    data = []
    print("\nThe given training examples are:")
    for row in readcsv:
        print(row)
        if row[len(row)-1].upper() == "YES":
            data.append(row)

print("\nThe positive examples are:");
for x in data:
    print(x);
print("\n");

TotalExamples = len(data);
i=0;
j=0;
k=0;
print("The steps of the Find-s algorithm are :\n",hypo);
list = [];
p=0;
d=len(data[p])-1;
for j in range(d):
    list.append(data[i][j]);
hypo=list;
i=1;
for i in range(TotalExamples):
    for k in range(d):
        if hypo[k]!=data[i][k]:
            hypo[k]='?';
            k=k+1;   
        else:
            hypo[k];
    print(hypo);
i=i+1;

print("\nThe maximally specific Find-s hypothesis for the given training examples is :");
list=[];
for i in range(d):
    list.append(hypo[i]);
print(list);


2. For a given set of training data examples stored in a .csv file, implement and demonstrate the candidate-elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.

s1:

import csv

with open("D:/traininddata.csv") as f:
    csv_file = csv.reader(f)
    data = list(csv_file)  # Ensure 'list' is not shadowed

    s = data[1][:-1]
    g = [['?' for i in range(len(s))] for j in range(len(s))]

    for i in data:
        if i[-1] == "Yes":
            for j in range(len(s)):
                if i[j] != s[j]:
                    s[j] = '?'
                    g[j][j] = '?'

        elif i[-1] == "No":
            for j in range(len(s)):
                if i[j] != s[j]:
                    g[j][j] = s[j]
                else:
                    g[j][j] = "?"

        print("\nSteps of Candidate Elimination Algorithm", data.index(i) + 1)
        print(s)
        print(g)

    gh = []
    for i in g:
        for j in i:
            if j != '?':
                gh.append(i)
                break

    print("\nFinal specific hypothesis:\n", s)
    print("\nFinal general hypothesis:\n", gh)

s2:

import numpy as np
import pandas as pd

# Loading Data from a CSV File
data = pd.DataFrame(data=pd.read_csv('D:/traininddata.csv'))
print(data)

# Separating concept features from Target
concepts = np.array(data.iloc[:,0:-1])
print(concepts)

# Isolating target into a separate DataFrame
# copying last column to target array
target = np.array(data.iloc[:,-1])
print(target)

def learn(concepts, target):
 
    '''
    learn() function implements the learning method of the Candidate elimination algorithm.
    Arguments:
        concepts - a data frame with all the features
        target - a data frame with corresponding output values
    '''

    # Initialise S0 with the first instance from concepts
    # .copy() makes sure a new list is created instead of just pointing to the same memory location
    specific_h = concepts[0].copy()
    print("\nInitialization of specific_h and general_h")
    print(specific_h)
    #h=["#" for i in range(0,5)]
    #print(h)

    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]
    print(general_h)
    # The learning iterations
    for i, h in enumerate(concepts):

        # Checking if the hypothesis has a positive target
        if target[i] == "Yes":
            for x in range(len(specific_h)):

                # Change values in S & G only if values change
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'

        # Checking if the hypothesis has a positive target
        if target[i] == "No":
            for x in range(len(specific_h)):
                # For negative hyposthesis change values only  in G
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

        print("\nSteps of Candidate Elimination Algorithm",i+1)
        print(specific_h)
        print(general_h)
 
    # find indices where we have empty rows, meaning those that are unchanged
    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]
    for i in indices:
        # remove those rows from general_h
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    # Return final values
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("\nFinal Specific_h:", s_final, sep="\n")
print("\nFinal General_h:", g_final, sep="\n")


3. Write a program to demonstrate the working of the decision tree based id3 algorithm. Use an appropriate data set for building the decision tree and apply this knowledge to classify a new sample.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from io import StringIO

data = pd.read_csv('D:/tennisdata.csv')
print("The first 5 values of data is \n", data.head())

# Splitting the data into features and target
X = data.iloc[:, :-1]
print("\nThe first 5 values of Train data is \n", X.head())
y = data.iloc[:, -1]
print("\nThe first 5 values of Train output is \n", y.head())

# Encoding categorical features
le_outlook = LabelEncoder()
X['Outlook'] = le_outlook.fit_transform(X['Outlook'])

le_Temperature = LabelEncoder()
X['Temperature'] = le_Temperature.fit_transform(X['Temperature'])

le_Humidity = LabelEncoder()
X['Humidity'] = le_Humidity.fit_transform(X['Humidity'])

le_Windy = LabelEncoder()
X['Windy'] = le_Windy.fit_transform(X['Windy'])

print("\nNow the Train data is:\n", X.head())

# Encoding the target
le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print("\nNow the Train output is:\n", y)

# Training the Decision Tree Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X, y)

# Function to encode input data
def labelEncoderForInput(list1):
    list1[0] = le_outlook.transform([list1[0]])[0]
    list1[1] = le_Temperature.transform([list1[1]])[0]
    list1[2] = le_Humidity.transform([list1[2]])[0]
    list1[3] = le_Windy.transform([list1[3]])[0]
    return [list1]

# Predicting for a new input
inp1 = ["Rainy", "Cool", "High", "False"]
pred1 = labelEncoderForInput(inp1)

# Convert the input to a DataFrame with the same column names as training data
pred1_df = pd.DataFrame(pred1, columns=X.columns)

# Making the prediction
y_pred = classifier.predict(pred1_df)

# Wrapping y_pred[0] in a list for inverse_transform
print("\nFor input {0}, we obtain {1}".format(inp1, le_PlayTennis.inverse_transform([y_pred[0]])[0]))

S2:
import numpy as np
import csv

def read_data(filename):
    with open(filename, 'r') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        headers = next(datareader)
        metadata = headers
        traindata = [row for row in datareader]
    return metadata, traindata

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""

    def __str__(self):
        return self.attribute

def subtables(data, col, delete):
    table_dict = {}
    items = np.unique(data[:, col])
    count = np.zeros(items.shape[0], dtype=np.int32)

    for idx, item in enumerate(items):
        mask = (data[:, col] == item)
        subset = data[mask]
        count[idx] = subset.shape[0]

        if delete:
            subset = np.delete(subset, col, axis=1)
        
        table_dict[item] = subset

    return items, table_dict

def entropy(S):
    unique_classes, counts = np.unique(S, return_counts=True)
    probabilities = counts / len(S)
    return -np.sum(probabilities * np.log2(probabilities))

def gain_ratio(data, col):
    items, subsets = subtables(data, col, delete=False)
    total_entropy = entropy(data[:, -1])
    intrinsic_value = 0
    weighted_entropy = 0

    for item in items:
        subset = subsets[item]
        ratio = subset.shape[0] / data.shape[0]
        weighted_entropy += ratio * entropy(subset[:, -1])
        if ratio > 0:
            intrinsic_value += -ratio * np.log2(ratio)

    information_gain = total_entropy - weighted_entropy
    if intrinsic_value == 0:
        return 0  # Avoid division by zero
    return information_gain / intrinsic_value

def create_node(data, metadata):
    if np.unique(data[:, -1]).size == 1:
        leaf = Node("")
        leaf.answer = np.unique(data[:, -1])[0]
        return leaf

    gains = [gain_ratio(data, col) for col in range(data.shape[1] - 1)]
    split_attr = np.argmax(gains)
    node = Node(metadata[split_attr])

    metadata = np.delete(metadata, split_attr, 0)
    items, subsets = subtables(data, split_attr, delete=True)

    for item in items:
        child_node = create_node(subsets[item], metadata.copy())
        node.children.append((item, child_node))

    return node

def print_tree(node, level=0):
    indent = "   " * level
    if node.answer:
        print(f"{indent}Answer: {node.answer}")
    else:
        print(f"{indent}Attribute: {node.attribute}")
        for value, child in node.children:
            print(f"{indent}  Value: {value}")
            print_tree(child, level + 1)

# Example usage
metadata, traindata = read_data("D:/tennisdata.csv")
data = np.array(traindata, dtype=str)
root = create_node(data, np.array(metadata))
print_tree(root)


4. Build an artificial neural network by implementing the backpropagation algorithm and test the same using appropriate datasets.

import numpy as np
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = X/np.amax(X, axis=0)
y = y/100

class Neural_Network(object):
    def __init__(self):
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        o = self.sigmoid(self.z3) 
        return o 

    def sigmoid(self, s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        return s * (1 - s)
    
    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error*self.sigmoidPrime(o)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train (self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

NN = Neural_Network()
for i in range(1000):
       print ("\nInput: \n" + str(X))
       print ("\nActual Output: \n" + str(y))
       print ("\nPredicted Output: \n" + str(NN.forward(X)))
       print ("\nLoss: \n" + str(np.mean(np.square(y - NN.forward(X)))))
       NN.train(X, y)


5. Write a program to implement the naïve bayesian classifier for a sample training data set stored as a.csv file. Compute the accuracy of the classifier, considering few test datasets.

import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('D:/tennisdata.csv')
print("The first 5 values of data is :\n",data.head())

X = data.iloc[:,:-1]
print("\nThe First 5 values of train data is\n",X.head())
y = data.iloc[:,-1]
print("\nThe first 5 values of Train output is\n",y.head())

le_outlook = LabelEncoder()
X.Outlook = le_outlook.fit_transform(X.Outlook)
le_Temperature = LabelEncoder()
X.Temperature = le_Temperature.fit_transform(X.Temperature)
le_Humidity = LabelEncoder()
X.Humidity = le_Humidity.fit_transform(X.Humidity)
le_Windy = LabelEncoder()
X.Windy = le_Windy.fit_transform(X.Windy)

print("\nNow the Train data is :\n",X.head())
le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print("\nNow the Train output is\n",y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
classifier = GaussianNB()
classifier.fit(X_train,y_train)

from sklearn.metrics import accuracy_score
print("Accuracy is:",accuracy_score(classifier.predict(X_test),y_test))


6. Assuming a set of documents that need to be classified, use the naïve bayesian classifier model to perform this task. built-in java classes/api can be used to write the program. Calculate the accuracy, precision, and recall for your data set.

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train',categories=categories,shuffle=True)
twenty_test = fetch_20newsgroups(subset='test',categories=categories,shuffle=True)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_tf = count_vect.fit_transform(twenty_train.data)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_tf)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import metrics
mod = MultinomialNB()
mod.fit(X_train_tfidf, twenty_train.target)

X_test_tf = count_vect.transform(twenty_test.data)
X_test_tfidf = tfidf_transformer.transform(X_test_tf)
predicted = mod.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(twenty_test.target, predicted))
print(classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names))
print("confusion matrix is \n",metrics.confusion_matrix(twenty_test.target, predicted))

