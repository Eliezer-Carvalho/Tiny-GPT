from tokenizers import Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

context_length = 8
batch_size = 4
number_embeddings = 32
head_size = 8
mumber_head = 4

torch.manual_seed (1000)

tokenizer = Tokenizer.from_file ("Os Lusiadas Tokenizer.json")

with open ("Os Lusiadas.txt", "r", encoding = "utf-8") as f:
    doc = f.read()
#print (len(doc))

tokens = tokenizer.encode (doc)
vocab_size = tokenizer.get_vocab_size()
#print (tokens.tokens [:100])
#print (tokens.ids [:100])
#print (len(tokens.tokens))

tensor = torch.tensor (tokens.ids, dtype = torch.long)
#print (tensor)
#print (len(tensor))

n = int (0.9 * len(tensor))
dados_treino = tensor [ : n] #90% Dados de Treino
dados_teste = tensor [n : ] #10% Dados de Teste


def get_batch():
    position = torch.randint (0, len(dados_treino) - context_length, (batch_size, ))
    x = torch.stack ([dados_treino [ i : i + context_length] for i in position])
    y = torch.stack ([dados_treino [i + 1 : i + context_length + 1] for i in position])

    return x, y
    
inputs, targets = get_batch()
#print (inputs.shape) (B T)
#print (inputs[0, 0])
#print (targets.shape) (B T)
#print (targets)

'''
embedding = nn.Embedding (vocab_size, number_embeddings)
inputs_embedding = embedding (inputs)
#print (inputs_embedding.shape) # (B T C)
#print (inputs_embedding[0, 0])

positional_encoding = nn.Embedding (context_length, number_embeddings)
inputs_final = inputs_embedding + positional_encoding(torch.arange (context_length))
#print (inputs_final[0, 0])


#----------------------------------------------------------------------------------- Attention ---------------------------------------------------------------------------------------------#

#Single Head Attention

B, T, C = 4, 8, 32 #Batch, Sequence Length, Channels ou Número de Embeddings

Query = nn.Linear (number_embeddings, head_size, bias = False)
Key = nn.Linear (number_embeddings, head_size, bias = False)
Value = nn.Linear (number_embeddings, head_size, bias = False)

Q = Query (inputs_final) # y = inputs_final * Wq
K = Key (inputs_final) # y = inputs_final * Wk
V = Value (inputs_final) # y = inputs_final *Wv
#print (inputs_final.shape) # (B T C)
#print (Q.shape) # (B T T)
#print (K.shape) # (B T T)
#print (V.shape) # (B T T)

weights = Q @ K.transpose (-2, -1) * (head_size ** -0.5) # (B T 8) @ (B 8 T) = (B T T)
#print (inputs_final)
#print (weights)

#Masked Attention - Usado em Transformers Decoders
tril = torch.tril (torch.ones(T, T))
weights = weights.masked_fill (tril == 0, float('-inf'))
#print (tril)
#print (weights)

weights = F.softmax (weights, dim = -1)
#print (weights)
#print (weights.shape)

output = weights @ V
#print (weights.shape)
#print (weights)
'''

class Head (nn.Module):
    def __init__(self):
        super().__init__()
        self.Query = nn.Linear(number_embeddings, head_size, bias = False)
        self.Key   = nn.Linear(number_embeddings, head_size, bias = False)
        self.Value = nn.Linear(number_embeddings, head_size, bias = False)

    def forward(self, inputs_final):
        B, T, C = inputs_final.shape


        Q = self.Query(inputs_final)
        K = self.Key(inputs_final)
        V = self.Value(inputs_final)

        weights = Q @ K.transpose (-2, -1) * (head_size ** -0.5)
        tril = torch.tril (torch.ones(T, T))
        weights = weights.masked_fill (tril == 0, float('-inf'))
        weights = F.softmax (weights, dim = -1)

        return weights @ V


class Multi_Head (nn.Module):
    def __init__(self):
        super().__init__()
        self.Heads = nn.ModuleList ([Head() for _ in range (4)])
        self.out_proj = nn.Linear (number_embeddings, number_embeddings)

    def forward (self, inputs_final):
        output = torch.cat ([h(inputs_final) for h in self.Heads], dim = -1)
        return self.out_proj (output)


class FeedForward_MLP (nn.Module):
    def __init__(self):
        super().__init__()
        self.Neural_Net = nn.Sequential (
            nn.Linear (number_embeddings, 4 * number_embeddings),
            nn.ReLU(),
            nn.Linear (4 * number_embeddings, number_embeddings)
        )

    def forward (self, inputs_final):
        return self.Neural_Net (inputs_final)
    

class Block (nn.Module):
    def __init__(self):
        super().__init__()
        self.Attention = Multi_Head()
        self.FNN = FeedForward_MLP()
        self.LayerNorm = nn.LayerNorm (number_embeddings)
        self.LayerNorm2 = nn.LayerNorm (number_embeddings)
        self.Dropout = nn.Dropout (0.1)

    def forward (self, inputs_final):
        dropout = self.Attention (self.LayerNorm (inputs_final))
        inputs_final = inputs_final + self.Dropout (dropout)

        dropout2 = self.FNN (self.LayerNorm2 (inputs_final))
        inputs_final = inputs_final + self.Dropout (dropout2)

        return inputs_final


class GPT (nn.Module):
    def __init__(self):
        super().__init__()
        self.Token_Embedding = nn.Embedding (vocab_size, number_embeddings)
        self.Positional_Encoding = nn.Embedding (context_length, number_embeddings)
        self.Dropout = nn.Dropout (0.1)

        self.Blocks = nn.Sequential (
            Block(),
            Block(),
            Block(),
            Block()
        )

        self.LayerNormFinal = nn.LayerNorm (number_embeddings)
        self.Language_Modeling_Head = nn.Linear (number_embeddings, vocab_size)


    def forward (self, inputs):
        B, T = inputs.shape


    
        inputs_final = self.Token_Embedding (inputs) + self.Positional_Encoding (torch.arange (T, device = inputs.device))
        inputs_final = self.Dropout (inputs_final)

        inputs_final = self.Blocks(inputs_final)
        inputs_final = self.LayerNormFinal (inputs_final)
        logits = self.Language_Modeling_Head (inputs_final)

        return logits



if __name__ == "__main__":

    model = GPT()
    otimizador = torch.optim.AdamW (model.parameters(), lr = 1e-4)
    loss_fn = nn.CrossEntropyLoss()


    for step in range (1000):

        inputs, targets = get_batch()

        logits = model (inputs)

        B, T, V = logits.shape

        loss = loss_fn (
            logits.view (B * T, V),
            targets.view (B * T)
        )

        otimizador.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_ (model.parameters(), 1.0)
        otimizador.step()

        if step % 100 == 0:
            print (loss.item())


def generate(model, idx, max_new_tokens, temperature=1.0):
    model.eval()

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_length:]

        logits = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_token], dim=1)

    return idx


start = torch.zeros((1, 1), dtype=torch.long)  # token inicial

output = generate(model, start, 100)

text = tokenizer.decode(output[0].tolist())
print(text)



'''
class Single_Head_Attention (nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.Key = nn.Linear (number_embeddings, head_size, bias = False)
        self.Query = nn.Linear (number_embeddings, head_size, bias = False)
        self.Value = nn.Linear (number_embeddings, head_size, bias = False)
        

    def forward(self, inputs_final):
        B, T, C = inputs_final.shape
        K = self.Key (inputs_final)
        Q = self.Query (inputs_final)
        V = self.Value (inputs_final)

        weights = Q @ K.transpose (-2, -1) * (head_size ** -0.5)
        tril = torch.tril(torch.ones(T, T, device = inputs_final.device))
    
        weights = weights.masked_fill(tril == 0, float('-inf'))
        weights = F.softmax (weights, dim = -1)

        output = weights @ V
        return output
    

class Attention_Single_Head_Model (nn.Module):

    def __init__(self):
        super().__init__()
        self.Token_Embedding = nn.Embedding (vocab_size, number_embeddings)
        self.Positional_Encoding = nn.Embedding (context_length, number_embeddings)
        self.SHA = Single_Head_Attention (number_embeddings)
        self.LMA = nn.Linear (number_embeddings, vocab_size)

    def forward(self, inputs, targets = None):
        B, T = inputs.shape

        token_embedding = self.Token_Embedding (inputs)
        positional_encoding_emb = self.Positional_Encoding (torch.arange (context_length, device = inputs.device))
        input_final = token_embedding + positional_encoding_emb
        input_final = self.SHA (input_final)
        logits = self.LMA (input_final)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view (B * T, C)
            targets = targets.view (B * T)
            loss = F.cross_entropy (logits, targets)

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range (max_new_tokens):
            
            indice_max = inputs[:, -context_length:]
            indice_max = indice_max.to(device)

            logits, _ = self(indice_max)

            logits = logits[:, -1, :]

            probs = F.softmax (logits, dim = -1)

            indice_next = torch.multinomial (probs, num_samples = 1) 

            inputs = torch.cat ((inputs, indice_next), dim = 1)
        
        return inputs
    


model = Attention_Single_Head_Model().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):

    x, y = get_batch()
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(step, loss.item())


start = torch.zeros((1, 1), dtype=torch.long, device=device)

out = Attention_Single_Head_Model.generate(model, start, 200)

print(tokenizer.decode(out[0].tolist()))

'''