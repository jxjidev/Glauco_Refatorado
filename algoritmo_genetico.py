import random

class Item:
    def __init__(self, peso, valor):
        self.peso = peso
        self.valor = valor

class Individuo:
    def __init__(self, itens, capacidade_mochila, cromossomo=None):
        self.itens = itens
        self.capacidade_mochila = capacidade_mochila
        self.cromossomo = cromossomo or [random.randint(0, 1) for _ in itens]
        self.avaliar()

    def avaliar(self):
        peso_total = sum(item.peso for i, item in enumerate(self.itens) if self.cromossomo[i])
        valor_total = sum(item.valor for i, item in enumerate(self.itens) if self.cromossomo[i])
        self.fitness = valor_total if peso_total <= self.capacidade_mochila else 0

    def mutar(self, taxa_mutacao):
        for i in range(len(self.cromossomo)):
            if random.random() < taxa_mutacao:
                self.cromossomo[i] = 1 - self.cromossomo[i]
        self.avaliar()

    def cruzar(self, outro):
        ponto_corte = random.randint(1, len(self.cromossomo) - 1)
        filho1 = Individuo(self.itens, self.capacidade_mochila, self.cromossomo[:ponto_corte] + outro.cromossomo[ponto_corte:])
        filho2 = Individuo(self.itens, self.capacidade_mochila, outro.cromossomo[:ponto_corte] + self.cromossomo[ponto_corte:])
        return filho1, filho2

class AlgoritmoGenetico:
    def __init__(self, itens, capacidade_mochila, tamanho_populacao=100, geracoes=1000, taxa_mutacao=0.01):
        self.itens = itens
        self.capacidade_mochila = capacidade_mochila
        self.tamanho_populacao = tamanho_populacao
        self.geracoes = geracoes
        self.taxa_mutacao = taxa_mutacao
        self.populacao = [Individuo(itens, capacidade_mochila) for _ in range(tamanho_populacao)]

    def selecionar_pais(self):
        return random.sample(self.populacao, 2)

    def evoluir(self):
        for _ in range(self.geracoes):
            nova_populacao = []

            while len(nova_populacao) < self.tamanho_populacao:
                pai1, pai2 = self.selecionar_pais()
                filho1, filho2 = pai1.cruzar(pai2)
                filho1.mutar(self.taxa_mutacao)
                filho2.mutar(self.taxa_mutacao)
                nova_populacao.extend([filho1, filho2])

            self.populacao = sorted(nova_populacao, key=lambda ind: ind.fitness, reverse=True)[:self.tamanho_populacao]

    def obter_melhor(self):
        return max(self.populacao, key=lambda ind: ind.fitness)
