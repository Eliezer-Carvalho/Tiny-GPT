import humanize

#OpenAI Method
#FLOPs são o número de operações de ponto flutuante que o modelo faz
def flops_counter (parameters_non_embedding, batch, sequence_length):

    #Os Transformers não processam todos os tokens de uma vez e sim apenas batches de tokens de cada vez
    tokens = batch * sequence_length

    #Usa-se 6 porque tem em conta Forward (2), Backward (2) e Gradiente (2)
    training_flops = 6 * parameters_non_embedding * tokens  #Apenas 1 step

    #FLOPs por tokens é apenas Forward (2)
    flops_token = 2 * parameters_non_embedding

    #FLOPs usados no momento de inferência
    flops_inferência = flops_token * tokens

    print (f"Número total de FLOPs por Step de Treino -> {humanize.intword(training_flops)} ({(humanize.scientific(training_flops))})")
    #print (f"Número total de FLOPs por Token -> {humanize.intword(flops_token)} ({humanize.scientific(flops_token)})")
    print (f"Número total de FLOPs por Inferência -> {humanize.intword(flops_inferência)} ({humanize.scientific(flops_inferência)})")

flops_counter (450500, 8, 64)

#MFLOPs	10e6
#GFLOPs	10e9
#TFLOPs	10e12
#PFLOPs	10e15