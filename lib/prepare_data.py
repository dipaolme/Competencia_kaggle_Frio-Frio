import numpy as np
import pandas as pd

from lib.transform import transform_data

def currency_conversion_USD(col, currency):
    # obtenido de analisis exploratorio
    dict_conversion = {'AUD': 0.71, 'EUR': 1.13, 'GBP': 1.32, 'JPY': 0.01, 'USD': 1.0}

    # factor de conversion
    conversion_factor = currency.map(lambda x: dict_conversion[x]).astype('float')

    converted_col = col * conversion_factor

    return converted_col


def completed_cero_Total_Amounts(df):
    df['price_trf'] = df['ASP_(converted)'] * df['TRF'] * 1000000
    # reemplazo nan por cero
    df['price_trf'].replace(np.nan, 0, inplace=True)

    # asigno a cada product_name la mediana
    df['product_median_amount'] = df.groupby('Product_Name')['Total_Amount'].transform('median')
    # reemplazo nan por cero
    df['product_median_amount'].replace(np.nan, 0, inplace=True)

    rows = []

    for idx, row in df.iterrows():
        if row['Total_Amount'] == 0:
            if row['price_trf'] != 0:
                rows.append(row['price_trf'])
            else:
                rows.append(row['product_median_amount'])
        else:
            rows.append(row['Total_Amount'])

    # elimino columnas
    df.drop(['price_trf', 'product_median_amount'], axis=1, inplace=True)

    return pd.Series(rows)


def load(path):

    features_categoricas = ['Region', 'Territory', 'Bureaucratic_Code', 'Source ',
                            'Billing_Country', 'Account_Name', 'Opportunity_Name', 'Account_Owner',
                            'Opportunity_Owner', 'Account_Type', 'Opportunity_Type', 'Quote_Type',
                            'Delivery_Terms', 'Brand', 'Product_Type', 'Size', 'Product_Category_B', 'Currency',
                            'Last_Modified_By', 'Product_Family', 'Product_Name', 'ASP_Currency',
                            'ASP_(converted)_Currency', 'Delivery_Quarter', 'Total_Taxable_Amount_Currency',
                            'Stage', 'Prod_Category_A', 'Total_Amount_Currency']

    features_int = ['ID', 'Opportunity_ID', 'TRF', 'Pricing, Delivery_Terms_Quote_Approval']

    features_float = ['Price', 'ASP', 'ASP_(converted)', 'Total_Amount', 'Total_Taxable_Amount']

    features_bool = ['Pricing, Delivery_Terms_Approved', 'Pricing, Delivery_Terms_Quote_Appr',
                     'Bureaucratic_Code_0_Approval', 'Bureaucratic_Code_0_Approved', 'Submitted_for_Approval']

    features_fecha = ['Account_Created_Date', 'Opportunity_Created_Date', 'Last_Activity',
                      'Quote_Expiry_Date', 'Last_Modified_Date', 'Planned_Delivery_Start_Date',
                      'Planned_Delivery_End_Date', 'Month', 'Delivery_Year', 'Actual_Delivery_Date']

    dict_types = {}

    categoria = ['category']
    entero = ['int32']
    decimal = ['float64']
    booleano = ['bool']

    for k, v in zip(features_int, len(features_int) * entero):
        dict_types[k] = v

    for k, v in zip(features_float, len(features_float) * decimal):
        dict_types[k] = v

    for k, v in zip(features_categoricas, len(features_categoricas) * categoria):
        dict_types[k] = v

    for k, v in zip(features_bool, len(features_bool) * booleano):
        dict_types[k] = v

    df = pd.read_csv(path, dtype=dict_types, na_values=['None', 'none', 'Other'],
                     parse_dates=features_fecha)

    return df


def process(df, train: bool):

    if train:
        # agrego variable target
        df['Target'] = np.where(df['Stage'] != 'Closed Lost', 1, 0)


    ####################### Pre-procesamiento: antes de agrupar #####################

    ##### Variables monto

    # se convierte montos a dolares
    df['Total_Amount'] = currency_conversion_USD(df['Total_Amount'], df['Total_Amount_Currency'])
    df['Total_Taxable_Amount'] = currency_conversion_USD(df['Total_Taxable_Amount'], df['Total_Taxable_Amount_Currency'])

    # se reemplaza nan por cero de la variable 'Total_Amount'
    df['Total_Amount'].replace(np.nan, 0, inplace=True)

    # se completa los ceros de la varible 'Total_Amount'
    df['Total_Amount'] = completed_cero_Total_Amounts(df)

    ##### Variable region

    # hay valores ambiguos, un mismo territorio esta asignado a 2 regiones diferentes (ver tp_1)
    # se actualiza a la Región de mayor cantidad de Oportunidades
    # puede faltar casos
    df.loc[df.Territory == 'Jordan', 'Region'] = 'Middle East'
    df.loc[df.Territory == 'KSA', 'Region'] = 'Middle East'
    df.loc[df.Territory == 'Kuwait', 'Region'] = 'Middle East'
    df.loc[df.Territory == 'SE America', 'Region'] = 'Americas'
    df.loc[df.Territory == 'SW America', 'Region'] = 'Americas'
    df.loc[df.Territory == 'UAE (Dubai)', 'Region'] = 'Middle East'

    ##### Variable Planned_Delivery_End_Date

    # se completa con la variable Month
    df['Planned_Delivery_End_Date'] = df.apply(lambda x: x['Month'] if pd.isnull(x['Planned_Delivery_End_Date']) \
        else x['Planned_Delivery_End_Date'], axis=1)

    # se pasa la variable Month  a formato 1-12
    df['Month'] = df['Month'].dt.month

    ##### Variables booleanas

    features_bool = ['Pricing, Delivery_Terms_Approved', 'Pricing, Delivery_Terms_Quote_Appr',
                     'Bureaucratic_Code_0_Approval',
                     'Bureaucratic_Code_0_Approved', 'Submitted_for_Approval']

    # se pasan a 0-1
    for col in features_bool:
        df[col] = df[col].map(lambda x: 0 if x == False else 1)



    ####################### Se Agrupa por Oportunidad ################################



    if train:

        gruped = df.groupby('Opportunity_ID').agg(Total_Taxable_Amount=('Total_Taxable_Amount', 'first'),
                                                  Total_Amount_sum=('Total_Amount', 'sum'),
                                                  Region=('Region', 'first'),
                                                  Target=('Target', 'first'),
                                                  Opportunity_Created_Date=('Opportunity_Created_Date', 'first'),
                                                  Account_Created_Date=('Account_Created_Date', 'first'),
                                                  Planned_Delivery_End_Date=('Planned_Delivery_End_Date', 'max'),
                                                  Pricing_Delivery_Terms_Approved=(
                                                  'Pricing, Delivery_Terms_Approved', 'first'),
                                                  Pricing_Delivery_Terms_Quote_Appr=(
                                                  'Pricing, Delivery_Terms_Quote_Appr', 'first'),
                                                  Bureaucratic_Code_0_Approval=(
                                                  'Bureaucratic_Code_0_Approval', 'first'),
                                                  Bureaucratic_Code_0_Approved=(
                                                  'Bureaucratic_Code_0_Approved', 'first'),
                                                  Submitted_for_Approval=('Submitted_for_Approval', 'first'),
                                                  Month=('Month', 'max'),
                                                  Product_Family=('Product_Family', lambda x: list(x))).reset_index()

    else:

        gruped = df.groupby('Opportunity_ID').agg(Total_Taxable_Amount=('Total_Taxable_Amount', 'first'),
                                                  Total_Amount_sum=('Total_Amount', 'sum'),
                                                  Region=('Region', 'first'),
                                                  Opportunity_Created_Date=('Opportunity_Created_Date', 'first'),
                                                  Account_Created_Date=('Account_Created_Date', 'first'),
                                                  Planned_Delivery_End_Date=('Planned_Delivery_End_Date', 'max'),
                                                  Pricing_Delivery_Terms_Approved=(
                                                  'Pricing, Delivery_Terms_Approved', 'first'),
                                                  Pricing_Delivery_Terms_Quote_Appr=(
                                                  'Pricing, Delivery_Terms_Quote_Appr', 'first'),
                                                  Bureaucratic_Code_0_Approval=(
                                                  'Bureaucratic_Code_0_Approval', 'first'),
                                                  Bureaucratic_Code_0_Approved=(
                                                  'Bureaucratic_Code_0_Approved', 'first'),
                                                  Submitted_for_Approval=('Submitted_for_Approval', 'first'),
                                                  Month=('Month', 'max'),
                                                  Product_Family=('Product_Family', lambda x: list(x))).reset_index()



    ####################### Post-procesamiento: despues de agrupar #####################


    ##### Variable Total_Taxable_Amount

    # Falta chequear limite de anios

    # se reemplaza nan por cero
    gruped['Total_Taxable_Amount'].replace(np.nan, 0, inplace=True)

    # se completa los valores cero por la suma de Total_Amount previamente completados en e preprocesamiento
    gruped['Total_Taxable_Amount'] = gruped.apply(lambda x: x['Total_Amount_sum'] if x['Total_Taxable_Amount'] == 0 \
        else x['Total_Taxable_Amount'], axis=1)


    ###### Sanity checks Variables Temporales

    # Entrega del producto posterior o igual a la creacion de la cuenta
    #condition_1 => Account_Created_Date <= Planned_Delivery_End_Date

    gruped['Planned_Delivery_End_Date'] = gruped.apply(
        lambda x: x['Opportunity_Created_Date'] if (x['Opportunity_Created_Date'] > x['Planned_Delivery_End_Date']) else
        x['Planned_Delivery_End_Date'], axis=1)


    # Creacion de la cuenta anterior o mismo tiempo que la oportunidad
    #condition_3 => Account_Created_Date <= Opportunity_Created_Date,

    gruped['Account_Created_Date'] = gruped.apply(
        lambda x: x['Opportunity_Created_Date'] if (x['Opportunity_Created_Date'] < x['Account_Created_Date']) else x[
            'Account_Created_Date'], axis=1)



    ######## Feature Engineering : Creacion de Variqables #####################

    ### Variable Antiguedad de la cuenta
    gruped['antiguedad_cuenta'] = round((gruped['Opportunity_Created_Date'] - \
                                         gruped['Account_Created_Date']).dt.days / 30, 2)

    ### Variable Tiempo de entrega maximo
    gruped['tiempo_entrega_max'] = (gruped['Planned_Delivery_End_Date'] - \
                                    gruped['Opportunity_Created_Date']).dt.days

    ###### Feature Transformation #######

    ####  Product_Family: parte II Target Encoding

    # dict con el Target encoding de la variable Product_Family
    code_map = transform_data.Product_Family()

    # si la familia no se encuentra en el dict se usa como default el promedio
    default_value = 0.6218728632446444

    # Se mapea cada familia de acuerdo al target encoding generdo en transform_data
    helper = gruped.assign(Product_Family_Mean=[[code_map[k] if code_map.get(k) else default_value for k in row  ] for row in gruped['Product_Family']])

    # en el caso de que haya mas de un producto de la misma familia se eliminan los duplicados
    helper['Product_Family_Mean_sin_duplicados'] = helper['Product_Family_Mean'].map(lambda x: list(dict.fromkeys(x)))

    # en el caso de que haya mas 1 familia por oportunidad se promedia
    gruped['Product_Family'] = helper['Product_Family_Mean_sin_duplicados'].map(lambda x: sum(x) / len(x))


    # se ordena y selecciona las variable para  el modentrenar el modelo y predecir

    if train:

        gruped = gruped.loc[:, ['Opportunity_ID', 'Total_Taxable_Amount', 'Region', 'antiguedad_cuenta', 'tiempo_entrega_max',
                  'Target', 'Pricing_Delivery_Terms_Approved', 'Pricing_Delivery_Terms_Quote_Appr',
                  'Bureaucratic_Code_0_Approval', 'Bureaucratic_Code_0_Approved', 'Submitted_for_Approval', 'Month', 'Product_Family']]
    else:

        gruped = gruped.loc[:, ['Opportunity_ID', 'Total_Taxable_Amount', 'Region', 'antiguedad_cuenta', 'tiempo_entrega_max',
              'Pricing_Delivery_Terms_Approved', 'Pricing_Delivery_Terms_Quote_Appr',
              'Bureaucratic_Code_0_Approval', 'Bureaucratic_Code_0_Approved', 'Submitted_for_Approval', 'Month', 'Product_Family']]



    # Se genera el archivo procesado

    if train:
        file_name = 'train_prepared.csv'
    else:
        file_name = 'test_prepared.csv'

    gruped.to_csv('/home/dipa/proyectos/7506-Organizacion-de-Datos/tp_2/data/' + file_name )

    return gruped