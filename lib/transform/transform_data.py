import os
import numpy as np
from lib.prepare_data import load

### Se carga el file correspondiente
WORK_DIR = '/home/dipa/proyectos/7506-Organizacion-de-Datos/tp_2'
train = 'data/Entrenamieto_ECI_2020.csv'
df = load(os.path.join(WORK_DIR, train))
df['Target'] = np.where(df['Stage'] != 'Closed Lost', 1, 0)

def Product_Family():

    gruped_Product_Family = df.groupby('Product_Family').agg({'Target': 'mean', 'Opportunity_ID': 'nunique'})

    smooth = 10  # cantidad de casos que considero como valido para promediar el Target
    n = gruped_Product_Family['Opportunity_ID'] # cantidad de oportunidades que aparece determinada familia de producto

    global_mu = gruped_Product_Family['Target'].mean()  # promedio general de exito de todas las familias
    mu = gruped_Product_Family['Target'] # promedio de la tasa de exito de cada familia de producto
    mu_smoothed = (n * mu + smooth * global_mu) / (n + smooth) # Target Encoding Suavizado

    return mu_smoothed.to_dict()