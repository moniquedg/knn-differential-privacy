# Trabalho 3: Privacidade Diferencial – Mecanismo de Laplace em kNN

Disciplina: Privacidade de Dados  
Professor: Javam Machado
Linguagem: Python  

---

## Descrição do Projeto
Este projeto consiste na implementação do algoritmo de classificação **kNN (k-Nearest Neighbors)** do zero (sem recurso a bibliotecas de machine learning como o *scikit-learn*) e da sua variação **diferencialmente privada**, utilizando o **Mecanismo de Laplace** com base na adaptação do Algoritmo 1 de Gursoy et al. (2017).

O objetivo é avaliar o impacto da introdução de ruído de Laplace na acurácia das predições sobre a base de dados *Adult Income Dataset*, comparando o modelo tradicional com o modelo privado em diferentes orçamentos de privacidade ($\epsilon \in [0.5, 1.0, 5.0, 10.0]$).

---

## Pré-processamento dos Dados
Conforme especificado no enunciado, a base de dados passou pelas seguintes etapas de pré-processamento:

1. **Remoção de Atributos Irrelevantes:** Foram eliminados os atributos `fnlwgt`, `education`, `capital-gain`, `capital-loss` e `hours-per-week`.
2. **Tratamento de Valores Faltantes:** Remoção de todas as tuplas contendo o valor `"?"`[cite: 33].
3. **Codificação Categórica (Label Encoding):** Mapeamento de todas as colunas categóricas/não numéricas para valores inteiros[cite: 33].
4. **Divisão Treino/Teste:** A base foi dividida proporcionalmente em **70% para treino** ($X_{train}$) e **30% para teste** ($X_{test}$)[cite: 33].

---

## Fundamentação Teórica da Privacidade Diferencial

* **Mecanismo de Laplace:** Adiciona ruído gerado a partir da distribuição de Laplace com média $0$ e escala $b = \frac{\Delta f}{\epsilon'}$ às contagens de cada classe dentro do raio de vizinhança $r=6$[cite: 5, 8, 33, 34].
* **Sensibilidade Global ($\Delta f = 1$):** A adição ou remoção de uma única tupla no conjunto de dados altera a contagem de vizinhos de uma determinada classe em, no máximo, 1 unidade[cite: 8, 34].
* **Composição Sequencial:** Como o mecanismo é invocado para cada uma das $\vert{}X_{test}\vert{}$ instâncias de teste e para cada uma das $\vert{}\mathcal{L}\vert{}$ classes, o orçamento de privacidade total $\epsilon$ é dividido individualmente[cite: 8, 33]:
  $$\epsilon' = \frac{\epsilon}{\vert{}X_{test}\vert{} \times \vert{}\mathcal{L}\vert{}}$$
[cite: 8, 33]

---

## Estrutura do Repositório

```text
.
├── adult.csv                        # Dataset Adult Income (Kaggle/UCI)
├── trabalho_privacidade.py          # Código-fonte principal em Python
├── .gitignore                       # Ficheiro de exclusão do Git
├── predicoes_knn_tradicional.txt    # Predições do kNN sem ruído (k=10)
├── predicoes_knn_dp_eps_0.5.txt     # Predições privadas com ε = 0.5
├── predicoes_knn_dp_eps_1.0.txt     # Predições privadas com ε = 1.0
├── predicoes_knn_dp_eps_5.0.txt     # Predições privadas com ε = 5.0
├── predicoes_knn_dp_eps_10.0.txt    # Predições privadas com ε = 10.0
├── grafico_acuracia_dp.png          # Gráfico comparativo de acurácia
└── README.md                        # Documentação do projeto