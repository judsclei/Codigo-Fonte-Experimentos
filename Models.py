#Bibliotecas Utilizadas
from numpy import True_
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import datetime as dt
from pandas_datareader import data
#Normalizaçao 
from sklearn.preprocessing import MinMaxScaler
#Kfolds
from sklearn.model_selection import KFold
#MlpRegressor
from sklearn.neural_network import MLPRegressor
#metricas avaliaçao
from sklearn import metrics

def extraiBaseDeDados(derivativo, FONT, START, END, colunaAtualizada):
    dados = data.DataReader(derivativo, FONT, start= START, end= END)
    dados = dados[dados[colunaAtualizada].notna()]

    dados = removeColunas(dados, ['Close', 'Volume'])

    return dados

def removeColunas(data, colunas):
    data= data.drop(columns=colunas)
    data.tail()
    return data

def printaGraficosDataSet(data):
    plt.subplot(4,1,1)
    data['Open'].plot(label='Abertura',color='blue')
    plt.xlabel('Data')
    plt.ylabel('Abertura')

    plt.subplot(4,1,2)
    data['High'].plot(label='Alta',color='orange')
    plt.xlabel('Data')
    plt.ylabel('Alta')

    plt.subplot(4,1,3)
    data['Low'].plot(label='Baixa',color='green')
    plt.xlabel('Data')
    plt.ylabel('Baixa')

    plt.subplot(4,1,4)
    data['Adj Close'].plot(label='Fechamento Ajustado',color='red')
    plt.xlabel('Data')
    plt.ylabel('Fechamento Ajustado')

    plt.show()

def printaGraficosGeralDataSet(data):
    plt.figure(figsize = (15,10))
    plt.plot(data.index, data['Open'], label='Abertura')
    plt.plot(data.index, data['High'], label='Alta')
    plt.plot(data.index, data['Low'], label='Baixa')
    plt.plot(data.index, data['Adj Close'], label='Fechamento Ajustado')
    plt.legend(loc='best')
    plt.show()

def printaGraficoComConjuntosDeTeste(dataTeste):
    #Plotando grafico com o periodo de teste
    df_teste = pd.DataFrame({'Atual': dataTeste})
    df_teste.head()
    plt.figure(figsize = (15,10))
    plt.title('Representacao em grafico linhas do conjunto de teste')
    plt.ylabel('Preco Fechamento')
    plt.xlabel('Periodo')
    plt.plot(df_teste.index, df_teste['Atual'], label='Atual')
    plt.legend(loc='best')
    plt.show()


def normalizacaoBase(data, tipo):
    #Normaliza

    #Define os valores de X E Y de treino 

    if(tipo =='Treino'):
        arrBaseTreino = np.array(data)
        arrBaseTreino = MinMaxScaler(feature_range=(0, 1)).fit_transform(arrBaseTreino)

        X,Y = arrBaseTreino[:, :-1], arrBaseTreino[:, -1]

        printaGraficoComConjuntosDeTeste(Y)
        return X,Y
    
    else:
        arr = np.array(data)
        arr = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)
        X_val, Y_Val = arr[:, :-1], arr[:, -1]
        printaGraficoComConjuntosDeTeste(Y_Val)
        return  X_val, Y_Val


def kFolds(X,y):
    kf = KFold(n_splits=5)
    kf.get_n_splits(X)
    KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_treino, X_teste = X[train_index], X[test_index]
        Y_treino, Y_teste = y[train_index], y[test_index]

    return X_treino, X_teste, Y_treino, Y_teste

def treinaModeloRegressor(X_treino, Y_treino, X_val, Y_Val):
    #Utilizando a o mlp regressor para fazer o treinmento do modelo 
    regressor = MLPRegressor(
    hidden_layer_sizes=(100), 
    activation='logistic',
    solver='adam',
    alpha = 0.0001,
    batch_size = 'auto',
    learning_rate= 'constant',
    learning_rate_init= 0.005, #0,005
    momentum=0.2,
    max_iter=1000,
    shuffle = False,
    random_state=1,
    beta_1 = 0.9,
    beta_2 = 0.999).fit(X_treino, Y_treino)

    Y_previsto = regressor.predict(X_val)
    score = regressor.score(X_val, Y_Val)

    return Y_previsto, score

def plot(Y_Val,results):

    #scatter
    plt.scatter(Y_Val,results,color = 'blue')
    plt.title('Representacao em grafico scatter')
    plt.xlabel('Y_teste - Conjunto de teste')

    plt.ylabel('Y_previsto - Conjunto de previsto')
    plt.tight_layout()

    #Representaçao em linhas e em barras

    df_temp = pd.DataFrame({'Atual': Y_Val, 'Predito': results})
    df_temp.head()
    plt.figure(figsize = (15,10))
    plt.title('Representacao em grafico linhas ')
    plt.ylabel('Preco Fechamento')
    plt.xlabel('Periodo')
    plt.plot(df_temp.index, df_temp['Atual'], label='Atual')
    plt.plot(df_temp.index, df_temp['Predito'], label='Predito')
    plt.legend(loc='best')
    plt.show()


    df_temp = df_temp.head(30)
    df_temp.plot(kind='bar',figsize=(10,6))
    plt.show()

    plt.close()

def printMetrics(score, Y_Val,y_predict):
    print('Coeficiente de Determinacao (R2): ', score)
    print('Mean Absolute Error:', metrics.mean_absolute_error(Y_Val,y_predict))  
    print('Mean Squared Error:', metrics.mean_squared_error(Y_Val,y_predict))  
    print('Root Mean Squared Error:',  metrics.mean_squared_error(Y_Val,y_predict, squared=False))


def downloadDataTaxas(codigo):
    url = 'http://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=json&dataInicial=02/01/2018&dataFinal=25/02/2022'.format(codigo)
    data = pd.read_json(url)
    data['data'] = pd.to_datetime(data['data'])
    return data

def renomeiaAgrupaDados():
    dataIBOV = downloadDataFrame("^BVSP")
    dataDolar = downloadDataFrame("USDBRL=X")
    dataIBOV.rename({'High':'Maximo','Low':'Minimo','Open':'Abertura','Close':'Fechamento','Volume':'Volume','Adj Close':'Fechamento ajustado'}, axis=1, inplace=True)
    return dataIBOV

def download_data():
    dataIBOV = downloadDataFrame("^BVSP")
    dataDolar = downloadDataFrame("USDBRL=X")
    dataSELIC= downloadDataTaxas(432)
    dataIPCA = downloadDataTaxas(433)
    dataIGPM = downloadDataTaxas(189)
    data = pd.concat([dataIBOV,dataDolar['Adj Close']], axis=1)
    print(dataSELIC.tail(10))
    print(dataDolar.tail(10))
    print(dataIPCA.tail(10))
    print(dataIGPM.tail(10))
    print(dataIBOV.tail(10))
    return print(data.tail(5))


if __name__ == '__main__':
    # carrega data set de treino e de teste 
    #derivativo, FONT, START, END, colunaAtualizada
    ibov = extraiBaseDeDados("^BVSP","yahoo",dt.datetime(2018,1,2),dt.datetime(2022,2,25),'Adj Close')
    printaGraficosDataSet(ibov)
    printaGraficosGeralDataSet(ibov)
    #----
    ibovTeste = extraiBaseDeDados("^BVSP","yahoo",dt.datetime(2022,2,26),dt.datetime(2022,5,24),'Adj Close')
    printaGraficosDataSet(ibovTeste)
    printaGraficosGeralDataSet(ibovTeste)
    #----
    #Normalizaçao 
    X,Y = normalizacaoBase(ibov,'Treino')
    #----
    X_val, Y_Val = normalizacaoBase(ibovTeste,'Teste')
    #----
    #K-folds
    X_treino, X_teste, Y_treino, Y_teste = kFolds(X,Y)
    #-----
    #Treinamento
    Y_previsto, score = treinaModeloRegressor(X_treino, Y_treino, X_val, Y_Val)
    #------
    #plot graficos
    plot(Y_Val,Y_previsto)
    #------
    #printa Metricas
    printMetrics(score,  Y_Val,Y_previsto)
