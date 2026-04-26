from tokenizers import Tokenizer
#from tokenizers.models import BPE
#from tokenizers.pre_tokenizers import Punctuation, Whitespace, Sequence
#from tokenizers.trainers import BpeTrainer
#import matplotlib.pyplot as plt
import torch


tokenizer = Tokenizer.from_file("Os Lusiadas Tokenizer.json")

with open ("Os Lusiadas.txt", "r", encoding = "utf-8") as file:
    texto =  file.read()

output = tokenizer.encode(texto)

dados = torch.tensor(output.ids, dtype = torch.long)
#print (dados.shape, dados.dtype)
#print (dados[:1000])

n = int (0.9 * len(dados))
dados_treino = dados[:n]
dados_teste = dados[n:]

block_size = 8 #context_length

x = dados_treino [: block_size]
y = dados_treino [1: block_size + 1]
for j in range (block_size):
    token_atual = x [: j + 1]
    próximo_token = y [j]
    #print (f"Quando o input é {token_atual} o target: {próximo_token}")


batch_size = 4
torch.manual_seed(1000)

'''
def batch (split):
    dados = dados_treino if split == 'train' else dados_teste
    indice = torch.randint (0, len(dados) - block_size, (batch_size,))
    x = torch.stack ([dados [i : i + block_size] for i in indice])
    y = torch.stack ([dados [i + 1 : i + block_size + 1] for i in indice])
    
    return x, y


xb, yb = batch ('train')

print ('inputs:')
print (xb.shape)
print (xb)

print ('targets:')
print (yb.shape)
print (yb)
'''







'''
tokenizer = Tokenizer(BPE()) #Aqui estamos a selecionar o algoritmo de aprendizagem de tokens (Existem outras opções)
tokenizer.pre_tokenizer = Sequence([Whitespace(), Punctuation()]) #Aqui selecionamos a forma como o texto vai ser pré processado antes do Tokenizer (Existem outras opções) (Pre-Tokenizer)
#Conclusão --> O BPE vai dividir o texto, o PRE_TOKENIZER diz como dividir
trainer = BpeTrainer( #Aqui definimos alguns parâmetros do treino
    vocab_size = 5000, 
    special_tokens = ["<bos>", "<eos>", "<pad>", "<newline>"] #Special_tokens são usados para indicar ao modelo inicio de uma sequencia, fim de uma sequencia, padding
)

tokenizer.train(files = ["Os Lusiadas.txt"], trainer = trainer) #Treino final com o dataset que vai ser usado para o modelo

tokenizer.save("Os Lusiadas Tokenizer.json")



#Maneira mais fácil de entender como funciona um Tokenizer
#Primeiro - Pre-Tokenizer e depois aplicar o Treino


with open ("Os Lusiadas.txt", "r", encoding = "utf-8") as file:
    texto = file.read()

pre_tokenizer = Sequence([Whitespace(), Punctuation(),]).pre_tokenize_str(texto)

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(
    vocab_size = 1000,
    special_tokens = ["<bos>", "<eos>", "<pad>", "<newline>"]
)

tokenizer.train(files = ["Pre-Tokenizer.txt"], trainer = trainer)


tokenization = tokenizer.encode(texto)

print (len(tokenization.tokens))

vocabsize = [10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000]
numtoken = [210876, 211458, 212460, 213181, 214231, 214912, 215623, 217575, 220540, 228023]


plt.figure(figsize = (6, 7))
plt.plot(vocabsize, numtoken, marker = ".")

plt.xlabel("vocab_size", size = 12)
plt.xticks(list(range(0, 11000, 1000)))
plt.ylabel("num tokens", size = 12)
plt.yticks(list(range(200000, 250000, 2500)))

plt.title("Sequence[Whitespace(), Punctuation()] -> BPE")

plt.show()
'''


