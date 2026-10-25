# Trabalho 3: Privacidade Diferencial – Mecanismo de Laplace em kNN

- Disciplina: Privacidade de Dados
- Professor: Javam Machado
- Linguagem: Python

---

## Descrição do Projeto

Este projeto consiste na implementação do algoritmo de classificação **kNN (k-Nearest Neighbors)** do zero (sem usar bibliotecas de machine learning como o *scikit-learn*) e da sua variação **diferencialmente privada**, utilizando o **Mecanismo de Laplace** com base na adaptação do Algoritmo 1 de Gursoy et al. (2017).

O objetivo é avaliar o impacto da introdução de ruído de Laplace na acurácia das predições sobre a base de dados *Adult Income Dataset*, comparando o modelo tradicional com o modelo privado em diferentes orçamentos de privacidade (ε = 0,5, 1, 5 e 10).

---

## Como Executar

Requisitos: Python 3 e as bibliotecas listadas em `requirements.txt` (numpy, pandas, matplotlib e tqdm).

1. Coloque o arquivo `adult.csv` na pasta do projeto. Caso ele não esteja lá, baixe-o em
   https://www.kaggle.com/datasets/wenruliu/adult-income-dataset/

2. Instale as dependências:

   ```
   pip install -r requirements.txt
   ```

3. Execute o programa (no Windows, use `python` no lugar de `python3`):

   ```
   python3 trabalho_privacidade.py
   ```

A execução leva cerca de 20 segundos e salva todos os resultados na pasta `resultados/`. Ao final, uma janela mostra o gráfico de acurácia, e o programa termina quando ela é fechada. Como a semente é fixa, os resultados são os mesmos a cada execução.

Opções:

- `--detalhar ID`: mostra o passo a passo da tupla de teste de número ID (atributos, os 10 vizinhos mais próximos, os votos, as contagens no raio e as contagens ruidosas de cada ε). Usada na demo, por exemplo: `python3 trabalho_privacidade.py --detalhar 0`
- `--dados CAMINHO`: lê o dataset de outro caminho.
- `--saida PASTA`: salva os resultados em outra pasta (padrão: `resultados/`).
- `--semente N`: altera a semente da divisão treino/teste e do ruído (padrão: 42).

---

## Pré-processamento dos Dados

Conforme especificado no enunciado, a base de dados (48.842 tuplas e 15 atributos) passou pelas seguintes etapas de pré-processamento:

1. **Remoção de Atributos:** Foram eliminados os atributos `fnlwgt`, `education`, `capital-gain`, `capital-loss` e `hours-per-week`.
2. **Tratamento de Valores Faltantes:** Remoção das 3.620 tuplas contendo o valor `"?"`, restando 45.222 tuplas.
3. **Codificação Categórica (Label Encoding):** Em cada coluna não numérica, cada valor distinto recebe um inteiro 0, 1, 2, ... em ordem alfabética (ex.: `gender`: Female = 0, Male = 1; `income`: `<=50K` = 0, `>50K` = 1). As colunas `age` e `educational-num` já são inteiras e foram mantidas. A tabela completa está em `resultados/mapeamento.csv`. Os atributos não foram normalizados, pois o raio r = 6 do enunciado se refere a essa escala.
4. **Divisão Treino/Teste:** A base foi dividida aleatoriamente (semente 42) em **70% para treino** (X_train, 31.656 tuplas) e **30% para teste** (X_test, 13.566 tuplas).

Atributos usados na classificação (9): age, workclass, educational-num, marital-status, occupation, relationship, race, gender e native-country. Classe: income.

---

## kNN Tradicional

- Usa k = 10 e distância euclidiana, implementados sem bibliotecas de machine learning (o NumPy é usado apenas para operações sobre vetores).
- Para cada tupla de teste, calcula-se a distância até todas as tuplas de treino, selecionam-se as 10 mais próximas e escolhe-se a classe mais votada entre elas.
- Desempates: quando várias tuplas empatam na distância do 10º vizinho, entram as de menor índice no treino; quando os votos empatam (5 a 5), vence a classe do vizinho mais próximo. Com isso, o resultado é determinístico.

---

## kNN Diferencialmente Privado

Baseado no Algoritmo 1 de Gursoy et al. (2017):

1. Para cada tupla de teste x, conta-se quantas tuplas de treino de cada classe estão a uma distância menor ou igual ao raio r = 6 de x (a consulta de contagem do classificador r-N do artigo).
2. Cada contagem recebe ruído de Laplace (ver a seção seguinte), gerado pelo método da transformada inversa da CDF.
3. Conforme pedido no enunciado, o Algoritmo 1 foi adaptado para retornar as contagens ruidosas de cada classe, em vez do argmax. A classe prevista é a de maior contagem ruidosa; essa escolha é um pós-processamento e não consome orçamento de privacidade.

---

## Fundamentação Teórica da Privacidade Diferencial

* **Mecanismo de Laplace:** Adiciona ruído gerado a partir da distribuição de Laplace com média 0 e escala b = Δf / ε' às contagens de cada classe dentro do raio de vizinhança r = 6.
* **Sensibilidade Global (Δf = 1):** A adição ou remoção de uma única tupla no conjunto de dados altera a contagem de vizinhos de uma determinada classe em, no máximo, 1 unidade.
* **Composição Sequencial:** Como o mecanismo é invocado para cada uma das |X_test| instâncias de teste e para cada uma das |L| classes, o orçamento de privacidade total ε é dividido entre todas as chamadas:

  ε' = ε / (|X_test| × |L|) = ε / (13.566 × 2) = ε / 27.132

  Assim, a escala do ruído é b = Δf / ε' = 27.132 / ε.

---

## Resultados

Acurácia do kNN tradicional (k = 10): **81,74%**.

| ε   | ε' por chamada | Escala b | Acurácia kNN DP | Concordância com o kNN |
|-----|----------------|----------|-----------------|------------------------|
| 0,5 | 1,84 × 10⁻⁵    | 54.264   | 50,95%          | 50,88%                 |
| 1   | 3,69 × 10⁻⁵    | 27.132   | 50,43%          | 50,63%                 |
| 5   | 1,84 × 10⁻⁴    | 5.426    | 53,92%          | 53,93%                 |
| 10  | 3,69 × 10⁻⁴    | 2.713    | 56,46%          | 56,83%                 |

A concordância é a fração das tuplas de teste em que o kNN privado prevê a mesma classe que o kNN tradicional. O gráfico está em `resultados/grafico_acuracia_dp.png`.

A acurácia do kNN privado fica perto de 50% porque a escala do ruído (de 2.713 a 54.264) é muito maior que a diferença típica entre as contagens das duas classes (mediana de 636). Ela cresce com ε, mas, mesmo sem ruído, classificar pelas contagens no raio daria no máximo 78,11%. É o efeito da divisão do orçamento entre muitas consultas, descrito na Seção 3.1 do artigo.

---

## Estrutura do Repositório

```text
.
├── trabalho_privacidade.py            # Código-fonte principal em Python
├── requirements.txt                   # Dependências do projeto
├── adult.csv                          # Dataset Adult Income (baixar do Kaggle; ignorado pelo Git)
├── resultados/
│   ├── predicoes_knn_tradicional.csv  # Predições do kNN sem ruído (k = 10)
│   ├── predicoes_knn_dp_eps_0.5.csv   # Predições privadas com ε = 0,5
│   ├── predicoes_knn_dp_eps_1.0.csv   # Predições privadas com ε = 1
│   ├── predicoes_knn_dp_eps_5.0.csv   # Predições privadas com ε = 5
│   ├── predicoes_knn_dp_eps_10.0.csv  # Predições privadas com ε = 10
│   ├── acuracias.csv                  # Acurácia e concordância para cada ε
│   ├── mapeamento.csv                 # Codificação dos atributos não numéricos
│   └── grafico_acuracia_dp.png        # Gráfico comparativo de acurácia
├── Readme.txt                         # Documentação do projeto
├── Readme.md                          # Documentação do projeto (mesmo conteúdo, em Markdown)
└── .gitignore                         # Arquivo de exclusão do Git
```

Os arquivos de predições têm uma linha por tupla de teste, com as colunas `id_teste`, `classe_real` e `classe_prevista`. Os do kNN privado trazem também as contagens ruidosas de cada classe (`contagem_ruidosa_<=50K` e `contagem_ruidosa_>50K`).

---

## Referência

Gursoy, M. E.; Inan, A.; Nergiz, M. E.; Saygin, Y. Differentially private nearest neighbor classification. *Data Mining and Knowledge Discovery*, v. 31, p. 1544–1575, 2017. DOI: 10.1007/s10618-017-0532-z.
