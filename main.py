from algoritmo_genetico import Item, AlgoritmoGenetico

# Define itens da mochila (peso, valor)
itens = [
    Item(10, 60),
    Item(20, 100),
    Item(30, 120),
    Item(40, 240),
    Item(50, 300),
    Item(35, 150),
    Item(25, 80),
    Item(45, 200)
]

capacidade_mochila = 100

# Executa algoritmo genético
ag = AlgoritmoGenetico(itens, capacidade_mochila)
ag.evoluir()
melhor = ag.obter_melhor()

# Exibe resultado
print("Melhor solução encontrada:")
print("Cromossomo:", melhor.cromossomo)
print("Valor total:", melhor.fitness)
