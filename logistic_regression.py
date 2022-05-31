# -*- coding: utf-8 -*-
"""
# Logistic Regression with StandardScaler
"""

import warnings 
import numpy as np 
import pandas as pd 
import seaborn as se 
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,plot_confusion_matrix
warnings.filterwarnings('ignore')

"""### Importing dataset

Pandas is an open-source, BSD-licensed library providing high-performance, easy-to-use data manipulation and data analysis tools.

We will use panda's library to read the CSV file using its storage path.And we use the head function to display the initial row or entry.
"""

#df=pd.read_csv(file_path)
#df.head()
import pandas as pd
import io
from google.colab import files

uploaded = files.upload()
import io

df = pd.read_csv(io.BytesIO(uploaded['heart.csv']))

df.head()

"""### Feature Selections

It is the process of reducing the number of input variables when developing a predictive model. Used to reduce the number of input variables to both reduce the computational cost of modelling and, in some cases, to improve the performance of the model.

We will assign all the required input features to X and target/outcome to Y.
"""

X = df[['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']]
Y = df['HeartDisease']

"""### Data Preprocessing

Since the majority of the machine learning models in the Sklearn library doesn't handle string category data and Null value, we have to explicitly remove or replace null values. The below snippet have functions, which removes the null value if any exists. And convert the string classes data in the datasets by encoding them to integer classes.

"""

def NullClearner(df):
    if(isinstance(df, pd.Series) and (df.dtype in ["float64","int64"])):
        df.fillna(df.mean(),inplace=True)
        return df
    elif(isinstance(df, pd.Series)):
        df.fillna(df.mode()[0],inplace=True)
        return df
    else:return df
def EncodeX(df):
    return pd.get_dummies(df)
def EncodeY(df):
    if len(df.unique())<=2:
        return df
    else:
        un_EncodedT=np.sort(pd.unique(df), axis=-1, kind='mergesort')
        df=LabelEncoder().fit_transform(df)
        EncodedT=[xi for xi in range(len(un_EncodedT))]
        print("Encoded Target: {} to {}".format(un_EncodedT,EncodedT))
        return df

x=X.columns.to_list()
for i in x:
    X[i]=NullClearner(X[i])  
X=EncodeX(X)
Y=EncodeY(NullClearner(Y))
X.head()

"""#### Correlation Map

In order to check the correlation between the features, we will plot a correlation matrix. It is effective in summarizing a large amount of data where the goal is to see patterns.
"""

f,ax = plt.subplots(figsize=(18, 18))
matrix = np.triu(X.corr())
se.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, mask=matrix)
plt.show()

"""#### Distribution Of Target Variable"""

plt.figure(figsize = (10,6))
se.countplot(Y)

"""### Data Splitting

The train-test split is a procedure for evaluating the performance of an algorithm. The procedure involves taking a dataset and dividing it into two subsets. The first subset is utilized to fit/train the model. The second subset is used for prediction. The main motive is to estimate the performance of the model on new data.
"""

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=123)

"""#### Handling Target Imbalance

The challenge of working with imbalanced datasets is that most machine learning techniques will ignore, and in turn have poor performance on, the minority class, although typically it is performance on the minority class that is most important.

One approach to addressing imbalanced datasets is to oversample the minority class. The simplest approach involves duplicating examples in the minority class.We will perform overspampling using imblearn library. 
"""

x_train,y_train = RandomOverSampler(random_state=123).fit_resample(x_train, y_train)

"""### Model

Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model (a form of binary regression). This can be extended to model several classes of events.

#### Model Tuning Parameters

    1. penalty : {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}, default=’l2’
> Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties. ‘elasticnet’ is only supported by the ‘saga’ solver. If ‘none’ (not supported by the liblinear solver), no regularization is applied.

    2. C : float, default=1.0
> Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.

    3. tol : float, default=1e-4
> Tolerance for stopping criteria.

    4. solver : {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
> Algorithm to use in the optimization problem.  
For small datasets, ‘<code>liblinear</code>’ is a good choice, whereas ‘<code>sag</code>’ and ‘<code>saga</code>’ are faster for large ones.  
For multiclass problems, only ‘<code>newton-cg</code>’, ‘<code>sag</code>’, ‘<code>saga</code>’ and ‘<code>lbfgs</code>’ handle multinomial loss; ‘<code>liblinear</code>’ is limited to one-versus-rest schemes.
* ‘<code>newton-cg</code>’, ‘<code>lbfgs</code>’, ‘<code>sag</code>’ and ‘<code>saga</code>’ handle L2 or no penalty.
* ‘<code>liblinear</code>’ and ‘<code>saga</code>’ also handle L1 penalty.
* ‘<code>saga</code>’ also supports ‘<code>elasticnet</code>’ penalty.
* ‘<code>liblinear</code>’ does not support setting <code>penalty='none'</code>.

    5. random_state : int, RandomState instance, default=None
> Used when <code>solver</code> == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data.

    6. max_iter : int, default=100
> Maximum number of iterations taken for the solvers to converge.

    7. multi_class : {‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
> If the option chosen is ‘<code>ovr</code>’, then a binary problem is fit for each label. For ‘<code>multinomial</code>’ the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. ‘<code>multinomial</code>’ is unavailable when <code>solver</code>=’<code>liblinear</code>’. ‘auto’ selects ‘ovr’ if the data is binary, or if <code>solver</code>=’<code>liblinear</code>’, and otherwise selects ‘<code>multinomial</code>’.

    8. verbose : int, default=0
> For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.

    9. n_jobs : int, default=None
> Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”. This parameter is ignored when the <code>solver</code> is set to ‘liblinear’ regardless of whether ‘multi_class’ is specified or not. <code>None</code> means 1 unless in a joblib.parallel_backend context. <code>-1</code> means using all processors
"""

# Build Model here
model = make_pipeline(StandardScaler(),LogisticRegression(random_state = 123,n_jobs = -1))
model.fit(x_train, y_train)

"""#### Model Accuracy

score() method return the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy which is a harsh metric since you require for each sample that each label set be correctly predicted.
"""

print("Accuracy score {:.2f} %\n".format(model.score(x_test,y_test)*100))

"""#### Confusion Matrix

A confusion matrix is utilized to understand the performance of the classification model or algorithm in machine learning for a given test set where results are known.
"""

plot_confusion_matrix(model,x_test,y_test,cmap=plt.cm.Blues)

"""#### Classification Report
A Classification report is used to measure the quality of predictions from a classification algorithm. How many predictions are True, how many are False.

* **where**:
    - Precision:- Accuracy of positive predictions.
    - Recall:- Fraction of positives that were correctly identified.
    - f1-score:-  percent of positive predictions were correct
    - support:- Support is the number of actual occurrences of the class in the specified dataset.
"""

print(classification_report(y_test,model.predict(x_test)))
