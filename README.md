# python
python src/modelo.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from visualizacao import gerar_graficos

# Dataset
df = pd.read_csv("data/dados_motores.csv")

# Gerar gráficos automaticamente
gerar_graficos(df)

# Separação
X = df[['Temperatura', 'Vibracao', 'Corrente']]
y = df['Falha']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Previsão
y_pred = modelo.predict(X_test)

# Resultado
print("Acurácia:", accuracy_score(y_test, y_pred))
