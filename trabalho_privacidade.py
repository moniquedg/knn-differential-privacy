import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


# 1. PRÉ-PROCESSAMENTO DO DATASET ADULT INCOME
def preprocess_data(data_path):
    # Carregar o ficheiro CSV
    df = pd.read_csv(data_path, skipinitialspace=True)

    # Remover atributos especificados na especificação do trabalho:
    # fnlwgt, education, capital-gain, capital-loss, hours-per-week
    cols_to_drop = [
        "fnlwgt",
        "education",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Tratar valores faltantes representados por "?"
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)

    # Codificar TODAS as colunas categóricas/não numéricas para inteiros (Label Encoding)
    mapping_dict = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")
            mapping_dict[col] = dict(enumerate(df[col].cat.categories))
            df[col] = df[col].cat.codes

    # A última coluna é a variável alvo ('income')
    target_col = "income" if "income" in df.columns else df.columns[-1]

    X = df.drop(columns=[target_col]).values.astype(float)
    y = df[target_col].values.astype(int)

    return X, y, mapping_dict


# 2. SEPARAÇÃO TREINO/TESTE (70% / 30%)
def train_test_split_custom(X, y, test_size=0.3, random_seed=42):
    np.random.seed(random_seed)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    num_test = int(X.shape[0] * test_size)
    test_idx = indices[:num_test]
    train_idx = indices[num_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# 3. kNN TRADICIONAL (OTIMIZADO COM VETORIZAÇÃO NUMPY, k=10)
def run_knn_traditional(X_train, y_train, X_test, k=10):
    predictions = []
    print("Calculando kNN Tradicional (k=10)...")
    for x_t in tqdm(X_test, desc="kNN Tradicional"):
        # Distância Euclidiana vetorizada para todas as amostras de treino
        distances = np.linalg.norm(X_train - x_t, axis=1)
        # Índices dos k vizinhos mais próximos
        k_indices = np.argpartition(distances, k)[:k]
        k_nearest_labels = y_train[k_indices]

        # Votação por maioria
        counts = np.bincount(k_nearest_labels)
        predictions.append(np.argmax(counts))

    return np.array(predictions)


# 4. kNN DIFERENCIALMENTE PRIVADO (MECANISMO DE LAPLACE, r=6)
# Adaptado do Algoritmo 1 de Gursoy et al. (2017)
def run_knn_dp(X_train, y_train, X_test, r=6.0, epsilon=1.0, num_classes=2):
    num_test = len(X_test)
    # Divisão do orçamento de privacidade devido à composição sequencial
    epsilon_prime = epsilon / (num_test * num_classes)
    scale = 1.0 / epsilon_prime

    predictions = []
    print(f"Calculando kNN DP (epsilon={epsilon})...")
    for x_t in tqdm(X_test, desc=f"kNN DP (eps={epsilon})"):
        # Distâncias para todas as amostras de treino
        distances = np.linalg.norm(X_train - x_t, axis=1)

        # Filtra os vizinhos dentro do raio r
        within_radius = distances <= r
        labels_within = y_train[within_radius]

        # Contagem de instâncias por classe
        counts = np.bincount(labels_within, minlength=num_classes)

        # Adiciona ruído aleatório da distribuição de Laplace
        noise = np.random.laplace(0, scale, size=num_classes)
        noisy_counts = counts + noise

        # Seleciona a classe com a maior contagem ruidosa
        predictions.append(np.argmax(noisy_counts))

    return np.array(predictions)


# 5. FUNÇÃO DE AVALIAÇÃO E EXECUÇÃO PRINCIPAL
def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def main():
    dataset_path = "adult.csv"

    print("Carregando e pré-processando os dados...")
    X, y, mapping = preprocess_data(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, test_size=0.3)

    num_classes = len(np.unique(y))
    print(
        f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras | Classes: {num_classes}\n"
    )

    # --- 1. Executar kNN Tradicional (k=10) ---
    y_pred_trad = run_knn_traditional(X_train, y_train, X_test, k=10)
    acc_trad = calculate_accuracy(y_test, y_pred_trad)
    print(f"Acurácia kNN Tradicional: {acc_trad * 100:.2f}%\n")
    np.savetxt("predicoes_knn_tradicional.txt", y_pred_trad, fmt="%d")

    # --- 2. Executar kNN DP para cada Epsilon ---
    epsilons = [0.5, 1.0, 5.0, 10.0]
    dp_accuracies = []

    for eps in epsilons:
        y_pred_dp = run_knn_dp(
            X_train,
            y_train,
            X_test,
            r=6.0,
            epsilon=eps,
            num_classes=num_classes,
        )
        acc_dp = calculate_accuracy(y_test, y_pred_dp)
        dp_accuracies.append(acc_dp)
        print(f"Acurácia kNN DP (eps={eps}): {acc_dp * 100:.2f}%\n")
        np.savetxt(f"predicoes_knn_dp_eps_{eps}.txt", y_pred_dp, fmt="%d")

    # --- 3. Gerar Gráfico de Acurácia ---
    plt.figure(figsize=(8, 5))
    plt.plot(
        epsilons,
        [acc * 100 for acc in dp_accuracies],
        marker="o",
        linewidth=2,
        label="kNN DP (Mecanismo Laplace)",
    )
    plt.axhline(
        y=acc_trad * 100,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"kNN Sem Privacidade ({acc_trad * 100:.2f}%)",
    )

    plt.title("Acurácia do Classificador: kNN Tradicional vs. kNN DP")
    plt.xlabel("Orçamento de Privacidade (ε)")
    plt.ylabel("Acurácia (%)")
    plt.xscale("log")
    plt.xticks(epsilons, labels=[str(e) for e in epsilons])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("grafico_acuracia_dp.png")
    print("Processo concluído! Arquivos de predição e gráfico gerados com sucesso.")
    plt.show()


if __name__ == "__main__":
    main()