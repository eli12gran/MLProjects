from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

"""
    Este modulo tiene como fin encargarse del preprocesamiento por el cual tiene que pasar las variables del modelo y los datos 
    en general que se ingresen a través del Postman. Consiste de tres clases: 
    
    1. CleanAndNormalizeData() la cual se encarga de convertir en columnas de tipo float las columnas que tienen valores numéricos 
    como str. Despues de este proceso, se encarga de normalizar todos los datos numéricos usando el StandardScaler() de scikit learn. 
    Esta clase imprime las columnas que recibe al inicio de su proceso y al final del mismo con el fin de llevar un tracking de la
    correcta implementación de la misma sobre las variables.

    2. MapToBinary() la cual se encarga de tomar la variable K, que tiene en su mayoría datos nulos pero que resultó que su existencia es
    relevante para la detección de fraude, y crear una nueva columna llamada `k_bin` la cual tiene como valores 0 si K es cero o nula y 1
    si K tiene un valor diferente de cero. Luego de esto elimina la columna K y concatena k_bin con las anteriores columnas del DataFrame

    3. AssignCountryGroupsAndOneHotEncode() la cual se encarga de la agrupación de países dependiendo de su frecuencia: pone los países que
    tienen una frecuencia de entre 1 y 20 transacciones en el grupo 1; los países que tienen entre 20 y 350 frecuencia de transacciones 
    en el grupo 2; y por último los países que tienen entre 350 y 9400 transacciones serán el grupo 3. Esto puede sonar a mucho sesgo para
    con los países, sin embargo se escogieron esos rangos dado que la frecuencia de los países era acorde para ello. Para mostrar esto, se 
    agregó una tabla al notebook principal llamada "Frecuencia por país" con el fin de que se pueda observar por qué se escogieron estos rangos.
    Despues de realizar esto, la función ya queda con una nueva columna titulada `country_group` que tiene en su interior tres valores únicos:
    1,2,3. A esta columna se le realiza One Hot Encoding y luego se devuelve el dataframe concatenado con las columnas que venían de las 
    clases anteriores.

    4. CustomPipeline() llama a la función `execute_pipeline` el cual se encarga de realizar el Pipeline de forma ordenada para que la 
    asignación de las columnas en cada uno de los pasos se haga de forma correcta
    
"""

class CleanAndNormalizeData(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("\n[CleanAndNormalizeData] Columnas antes de transformar:", X.columns.tolist())

        forced_numeric_columns = ['Monto', 'Q']
        categorical_columns = [col for col in X.columns if col not in forced_numeric_columns and X[col].dtype == 'object']
        numerical_columns = [col for col in X.columns if col in forced_numeric_columns or X[col].dtype in ['float64', 'int64']]
        
        print("[CleanAndNormalizeData] Columnas categóricas detectadas:", categorical_columns)
        print("[CleanAndNormalizeData] Columnas numéricas detectadas:", numerical_columns)
        if numerical_columns:
            for col in numerical_columns:
                X[col] = pd.to_numeric(X[col].replace({r',': ''}, regex=True), errors='coerce')

            scaler = StandardScaler()
            X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

        print("[CleanAndNormalizeData] Columnas después de transformar:", X.columns.tolist())
        return X


class MapToBinary(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("\n[MapToBinary] Columnas antes de transformar:", X.columns.tolist())
        X["k_bin"] = np.where(X["K"] == 0, 0, 1)
        print("[MapToBinary] Columna creada: 'k_bin'")
        del X["K"]
        print("[MapToBinary] Columnas después de transformar:", X.columns.tolist())
        return X


class AssignCountryGroupsAndOneHotEncode(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.country_counts = X['J'].value_counts()
        self.group_mapping = {}  

        # aca registramos los países que caen en cada grupo
        for country, count in self.country_counts.items():
            if count < 20:
                self.group_mapping[country] = 1
            elif 20 <= count < 350:
                self.group_mapping[country] = 2
            elif 350 <= count <= 9400:
                self.group_mapping[country] = 3
            elif count > 9400:
                self.group_mapping[country] = 4
        
        self.categories_ = ['1', '2', '3', '4']
        return self

    def transform(self, X):
        print("\n[AssignCountryGroupsAndOneHotEncode] Columnas antes de transformar:", X.columns.tolist())
        
        # si el pais se vio durante el entrenamiento, usar su grupo
        def assign_group(country):
            if country in self.group_mapping:
                return self.group_mapping[country]
            return 2 # si no se vio, usar el dos
            
        # Asignamos la infoo
        X['country_group'] = X['J'].apply(assign_group).astype(str)
        
        encoder = OneHotEncoder(categories=[self.categories_], sparse_output=False)
        grupo_encoded = encoder.fit_transform(X[['country_group']])
        group_labels = encoder.get_feature_names_out(['country_group'])
        
        grupo_df = pd.DataFrame(grupo_encoded, columns=group_labels, index=X.index)
        X.drop(columns=['country_group', 'J'], inplace=True)
        X = pd.concat([X, grupo_df], axis=1)
        
        print("[AssignCountryGroupsAndOneHotEncode] Columnas después de transformar:", X.columns.tolist())
        return X


def execute_pipeline(data):
    cleaner = CleanAndNormalizeData()
    data_cleaned = cleaner.fit_transform(data)
    
    mapper = MapToBinary()
    data_mapped = mapper.fit_transform(data_cleaned)
    
    country_encoder = AssignCountryGroupsAndOneHotEncode()
    data_final = country_encoder.fit_transform(data_mapped)
    
    desired_columns = ["B", "C", "F", "P", "Q", "S", "Monto", "k_bin", 'country_group_2', "M", "N"]
    for col in desired_columns:
        if col not in data_final.columns:
            data_final[col] = 0  # rellena con ceros si la columna no existe
    
    data_final = data_final[desired_columns]
    print("\nColumnas finales:", data_final.columns.tolist())
    return data_final

class CustomPipeline(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy() 
        return execute_pipeline(X_copy)
