from competicao_am.metodo_competicao import MetodoCompeticao
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

import pandas as pd
def gerar_saida_teste( df_data_to_predict, col_classe, num_grupo):
    """
    Assim como os demais códigos da pasta "competicao_am", esta função 
    só poderá ser modificada na fase de geração da solução. 
    """

    scikit_method = RandomForestClassifier(random_state=2, n_estimators= 100)
    scikit_method_svc = LinearSVC(C=50, random_state=2,dual = True,max_iter=1000)
    ml_method = MetodoCompeticao(scikit_method)
    ml_method_svc = MetodoCompeticao(scikit_method_svc)

    #o treino será sempre o dataset completo - sem nenhum dado a mais e sem nenhum preprocessamento
    #esta função que deve encarregar de fazer o preprocessamento
    df_treino = pd.read_csv("datasets/movies_amostra.csv")

    #gera as representações e seu resultado
    y_to_predict, arr_predictions_ator = ml_method.eval_actors(df_treino, df_data_to_predict, col_classe)
    y_to_predict, arr_predictions_diretor = ml_method.eval_escritores(df_treino, df_data_to_predict, col_classe, "dirigido_por")
    y_to_predict, arr_predictions_escritorum = ml_method.eval_escritores(df_treino, df_data_to_predict, col_classe, "escrito_por_1")
    y_to_predict, arr_predictions_escritordois = ml_method.eval_escritores(df_treino, df_data_to_predict, col_classe, "escrito_por_2")
    y_to_predict, arr_predictions_bow = ml_method.eval_bow(df_treino, df_data_to_predict, col_classe)
    #gera as representações e seu resultado SVC
    y_to_predict, arr_predictions_ator_svc = ml_method.eval_actors(df_treino, df_data_to_predict, col_classe)
    y_to_predict, arr_predictions_diretor_svc = ml_method.eval_escritores(df_treino, df_data_to_predict, col_classe, "dirigido_por")
    y_to_predict, arr_predictions_escritorum_svc = ml_method.eval_escritores(df_treino, df_data_to_predict, col_classe, "escrito_por_1")
    y_to_predict, arr_predictions_escritordois_svc = ml_method.eval_escritores(df_treino, df_data_to_predict, col_classe, "escrito_por_2")
    y_to_predict, arr_predictions_bow_svc = ml_method.eval_bow(df_treino, df_data_to_predict, col_classe)
    
    #combina as duas
    arr_final_predictions = ml_method.combine_predictions([arr_predictions_ator,
                                                           arr_predictions_bow,
                                                           arr_predictions_diretor,
                                                           arr_predictions_escritorum,
                                                           arr_predictions_escritordois,
                                                           arr_predictions_ator_svc,
                                                           arr_predictions_bow_svc,
                                                           arr_predictions_diretor_svc,
                                                           arr_predictions_escritorum_svc,
                                                           arr_predictions_escritordois_svc])

    #grava o resultado obtido
    with open(f"predict_grupo_{num_grupo}.txt","w") as file_predict:
        for predict in arr_final_predictions:
            file_predict.write(ml_method.dic_int_to_nom_classe[predict]+"\n")
