from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import tree
import pickle
from sklearn.preprocessing import FunctionTransformer
Log10 = FunctionTransformer(np.log10, validate=True)
from joblib import dump, load


#### Modelo ######

column_transformer = ColumnTransformer(
    [
        ("total_Taxable_Amount", Log10, [0]),
        ("Region", OneHotEncoder(), [1]),
        ('binning', KBinsDiscretizer(n_bins=20, strategy='quantile', encode='ordinal'), [2, 3])
    ], remainder='passthrough'
)

pipeline = Pipeline([
    ('ct', column_transformer),
    ('rf_clf',
     RandomForestClassifier(random_state=1, n_estimators=100, max_depth=10, min_samples_split=5, max_features=10))
])


def train(df):

    X = df.drop(['Opportunity_ID', 'Target'], axis=1)
    y = df[['Target']].values.ravel()


    pipeline.fit(X, y)

    # guardamos el modelo de interes
    #dump(pipeline, '/home/dipa/proyectos/7506-Organizacion-de-Datos/tp_2/models/random_forest.joblib')
    dump(pipeline, '/home/dipa/proyectos/7506-Organizacion-de-Datos/tp_2/models/random_forest_product_family.joblib')

    return pipeline

def predict(df):


    X = df.drop('Opportunity_ID', axis=1)

    # cargamos el modelo previamente entrenado
    #pipeline = load('/home/dipa/proyectos/7506-Organizacion-de-Datos/tp_2/models/random_forest.joblib')
    pipeline = load('/home/dipa/proyectos/7506-Organizacion-de-Datos/tp_2/models/random_forest_product_family.joblib')

    # nos quedamos con la probabilidades de que cumpla con la condicion buscada
    df['Target'] = pd.Series(pipeline.predict_proba(X)[:, -1])
    results = df.loc[:, ['Opportunity_ID','Target']]

    # guardamos los resultados
    results.to_csv('/home/dipa/proyectos/7506-Organizacion-de-Datos/tp_2/results_2.csv', index=False)

    return results