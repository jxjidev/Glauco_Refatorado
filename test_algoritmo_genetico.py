import unittest
from algoritmo_genetico import Item, Individuo, AlgoritmoGenetico

class TestAlgoritmoGenetico(unittest.TestCase):

    def setUp(self):
        self.itens = [
            Item(10, 60),
            Item(20, 100),
            Item(30, 120)
        ]
        self.capacidade = 50

    def test_avaliacao_individuo_dentro_limite(self):
        cromossomo = [1, 1, 0]  # Peso total = 30, Valor = 160
        individuo = Individuo(self.itens, self.capacidade, cromossomo)
        self.assertEqual(individuo.fitness, 160)

    def test_avaliacao_individuo_excede_limite(self):
        cromossomo = [1, 1, 1]  # Peso total = 60, passa do limite
        individuo = Individuo(self.itens, self.capacidade, cromossomo)
        self.assertEqual(individuo.fitness, 0)

    def test_mutacao_altera_genes(self):
        cromossomo = [0, 0, 0]
        individuo = Individuo(self.itens, self.capacidade, cromossomo[:])
        individuo.mutar(taxa_mutacao=1.0)  # Força a mutação total
        self.assertNotEqual(individuo.cromossomo, cromossomo)

    def test_cruzamento_gera_filhos(self):
        pai1 = Individuo(self.itens, self.capacidade, [1, 0, 0])
        pai2 = Individuo(self.itens, self.capacidade, [0, 1, 1])
        filho1, filho2 = pai1.cruzar(pai2)
        self.assertEqual(len(filho1.cromossomo), 3)
        self.assertEqual(len(filho2.cromossomo), 3)

    def test_algoritmo_genetico_encontra_melhor(self):
        ag = AlgoritmoGenetico(self.itens, self.capacidade, tamanho_populacao=10, geracoes=50)
        ag.evoluir()
        melhor = ag.obter_melhor()
        self.assertTrue(melhor.fitness > 0)

if __name__ == '__main__':
    unittest.main()
