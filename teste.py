import pandas as pd
from competicao_am.gerar_resultado_teste import gerar_saida_teste
from competicao_am.preprocessamento_atributos_competicao import gerar_atributos_titulo,gerar_atributos_escritores

#altere aqui para o número correspondente ao seu grupo
num_grupo = 0

#leia o dataset fornecido pelo professor (coloquei apenas um exemplo, na entrega, será outro)
df_amostra_teste = pd.read_csv("datasets/movies_amostra_teste_ex.csv")
df_amostra = pd.read_csv("datasets/movies_amostra.csv")

gerar_saida_teste(df_amostra_teste,"genero", num_grupo)