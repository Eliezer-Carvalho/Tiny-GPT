from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from tokenizers.trainers import BpeTrainer

'''
tokenizer = Tokenizer(BPE()) #Aqui estamos a selecionar o Tokenizer que vamos utilizar, no fundo estamos a selecionar o algoritmo de aprendizagem de tokens (Existem outras opções)
tokenizer.pre_tokenizer = ByteLevel() #Aqui selecionamos a forma como o texto vai ser pré processado antes do Tokenizer (Existem outras opções)
#Conclusão --> O BPE vai dividir o texto, o PRE_TOKENIZER diz como dividir
trainer = BpeTrainer( #Aqui definimos alguns parâmetros do treino
    special_tokens = ["<bos>", "<eos>", "<pad>", "<newline>"] #Special_tokens são usados para indicar ao modelo inicio de uma sequencia, fim de uma sequencia, padding
)

tokenizer.train(files = ["Os Lusiadas.txt"], trainer = trainer) #Treino final com o dataset que vai ser usado para o modelo


with open ("Os Lusiadas.txt", "r", encoding = "utf-8") as file:
    texto = file.read()

print (len(texto))
print (texto[:1000])

'''

with open ("Os Lusiadas.txt", "r", encoding = "utf-8") as file:
    texto = file.read()

output = Split().pre_tokenize_str(texto)


with open ("Split.txt", "w", encoding = "utf-8") as file2:
    for item in output:
        file2.write(f"{item}\n")

