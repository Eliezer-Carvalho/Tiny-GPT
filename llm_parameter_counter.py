import math
import humanize

def PARAM_COUNT (dim_emb, vocab_size, num_blocks):
    
    Token_Embedding_Table = dim_emb * vocab_size

    #Número Total de Parâmetros por Head/Layer
    Query_Key_Value = 3 * pow(dim_emb, 2) # = Embedding_Dimension * (3 * Embedding_Dimension)
    Output_Projection = pow(dim_emb, 2) # = Embedding_Dimension * Embedding_Dimension

    #FNN
    Input_Layer = dim_emb * (4 * dim_emb) #4 Representa o fator de expansão da FNN #Paper
    Output_Layer = (dim_emb * 4) * dim_emb 

    #Language Modelling
    Language_Modelling = dim_emb * vocab_size

    print (f"Parâmetros por Block -> {humanize.intword(Query_Key_Value + Output_Projection + (Input_Layer + Output_Layer))}") #Attention até à saída da FFN
    print (f"Parâmetros de todos os Blocks -> {humanize.intword((Query_Key_Value + Output_Projection + (Input_Layer + Output_Layer)) * num_blocks)}") #Attention até à saída da FFN * Blocks
    print (f"Número de Parâmetros do Modelo -> {humanize.intword((Query_Key_Value + Output_Projection + (Input_Layer + Output_Layer)) * num_blocks + Token_Embedding_Table + Language_Modelling)}") #Total contando com embeddings
