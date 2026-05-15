import humanize


def flops_counter (number_layers, number_heads, emb_dim, seq_len, vocab_size):

    attention_dimension = emb_dim // number_heads
    fnn_dimension = 4 * emb_dim

    attention_QKV = 6 * seq_len * emb_dim**2           
    attention_scores = 2 * number_heads * seq_len**2 * attention_dimension 
    attention_output = 2 * seq_len * emb_dim**2      

    attention_total = attention_QKV + attention_scores + attention_output

    fnn_total = 2 * seq_len * (emb_dim * fnn_dimension + fnn_dimension * emb_dim)

    logits = 2 * seq_len * emb_dim * vocab_size


    total_flops_per_block = number_layers * (attention_total + fnn_total) + logits #Forward
    total_flops_per_step = total_flops_per_block * 3 #aproximação #Forward + Backward (Backprop e update dos pesos)
    total_flops_per_token = total_flops_per_block / seq_len

    print (f"Número de FLOPs por Block (Forward) --> {humanize.intword(total_flops_per_block)}FLOPs")
    print (f"Número de FLOPs por Step de Treino (Forward + Backward) --> {humanize.intword(total_flops_per_step)}FLOPs")
    print (f"Número de FLOPs por Token --> {humanize.intword(total_flops_per_token)}FLOPs")

'''def mfu_counter (model_parameters, flops_per_token, gpu_peak_flops):

    sustained_flops = flops_per_token * tokens_per_second  
    mfu = sustained_flops / total_peak_flops 

'''



flops_counter (6, 16, 256, 128, 78)