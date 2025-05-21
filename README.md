# 🧬 Algoritmo Genético para o Problema da Mochila 0/1 (Refatorado)

Este projeto apresenta uma implementação refatorada de um algoritmo genético para resolver o clássico problema da Mochila 0/1 (Knapsack Problem). O código foi separado em módulos para melhor legibilidade, testabilidade e manutenibilidade, seguindo boas práticas de engenharia de software.

---

## 🎯 Objetivo

O problema da Mochila 0/1 consiste em escolher um subconjunto de itens com pesos e valores distintos, de forma a maximizar o valor total sem ultrapassar a capacidade máxima da mochila.

Este projeto utiliza **algoritmos bio-inspirados** — mais especificamente, um **algoritmo genético** — para encontrar soluções aproximadas eficientes.

---

## 📁 Estrutura do Projeto

```
knapsack/
├── algoritmo_genetico.py         # Código principal do algoritmo genético
├── main.py                       # Execução principal com exemplos
├── test_algoritmo_genetico.py   # Testes unitários usando unittest
└── README.md                     # Documentação do projeto
```

---

## 🛠️ Requisitos

- Python 3.8 ou superior
- (Opcional) `pytest` para rodar testes com mais detalhes

Para instalar o `pytest` (caso deseje utilizá-lo):

```bash
pip install pytest
```

---

## 🚀 Como Executar

1. Clone ou baixe o repositório:

```bash
git clone https://github.com/seu-usuario/knapsack-genetico.git
cd knapsack-genetico
```

2. Execute o programa principal:

```bash
python main.py
```

---

## 🧪 Como Rodar os Testes

Utilizando `unittest` (incluso por padrão no Python):

```bash
python test_algoritmo_genetico.py
```

Ou utilizando `pytest`:

```bash
pytest
```

---

## ✅ Melhorias com a Refatoração

- **Modularização**: separação das responsabilidades em arquivos distintos.
- **Testabilidade**: inclusão de testes automatizados.
- **Reutilização**: permite modificar e testar partes específicas facilmente.
- **Clareza**: melhora a leitura e manutenção do código.
