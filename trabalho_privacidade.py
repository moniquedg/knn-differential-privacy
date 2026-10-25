"""Trabalho 3 - Privacidade Diferencial: mecanismo de Laplace no kNN (Adult Income Dataset).

Etapas:
  1. pré-processa a base e divide 70% treino / 30% teste;
  2. roda o kNN tradicional (k = 10, distância euclidiana) e salva as predições;
  3. roda o kNN diferencialmente privado (Algoritmo 1 de Gursoy et al. (2017) adaptado,
     mecanismo de Laplace, raio r = 6) para cada ε e salva as predições;
  4. salva a tabela de acurácias e o gráfico de acurácia por ε.

Uso:
  python trabalho_privacidade.py                roda tudo (lê o adult.csv desta pasta)
  python trabalho_privacidade.py --detalhar 7   roda tudo e mostra o passo a passo da
                                                tupla de teste 7 (útil na demo)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator
from tqdm import tqdm

PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))

COLS_TO_DROP = ["fnlwgt", "education", "capital-gain", "capital-loss", "hours-per-week"]
TARGET_COL = "income"
K = 10
RADIUS = 6.0
EPSILONS = [0.5, 1.0, 5.0, 10.0]

# Sensibilidade global da consulta de contagem: incluir ou remover uma tupla do treino
# muda o número de vizinhos de uma classe dentro do raio em no máximo 1.
SENSITIVITY = 1.0


# 1. PRÉ-PROCESSAMENTO DO DATASET ADULT INCOME
def preprocess_data(data_path):
    # Carregar o arquivo CSV
    df = pd.read_csv(data_path, skipinitialspace=True)
    print(f"Base original: {len(df)} tuplas, {df.shape[1]} atributos")

    # Remover os atributos pedidos no enunciado (dá erro se algum nome não existir na base)
    df = df.drop(columns=COLS_TO_DROP)

    # Remover as tuplas com valores faltantes, representados por "?"
    num_rows = len(df)
    df = df.replace("?", np.nan).dropna().reset_index(drop=True)
    print(f"Tuplas removidas por conter '?': {num_rows - len(df)}")

    # Codificar as colunas não numéricas como inteiros (Label Encoding): cada valor distinto
    # recebe um código 0, 1, 2, ... em ordem alfabética (ex.: gender: Female=0, Male=1;
    # income: <=50K=0, >50K=1). age e educational-num já são inteiros e ficam como estão.
    mapping_dict = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")
            mapping_dict[col] = dict(enumerate(df[col].cat.categories))
            df[col] = df[col].cat.codes

    feature_names = [col for col in df.columns if col != TARGET_COL]
    X = df[feature_names].values.astype(np.int64)
    y = df[TARGET_COL].values.astype(np.int64)
    print(f"Pré-processada: {len(df)} tuplas | atributos: {', '.join(feature_names)}")

    return X, y, feature_names, mapping_dict


# 2. SEPARAÇÃO TREINO/TESTE (70% / 30%)
def train_test_split_custom(X, y, rng, test_size=0.3):
    # Embaralha os índices: os primeiros 30% vão para teste e o resto para treino
    indices = rng.permutation(X.shape[0])

    num_test = int(X.shape[0] * test_size)
    test_idx = indices[:num_test]
    train_idx = indices[num_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# 3. kNN TRADICIONAL (k=10, DISTÂNCIA EUCLIDIANA, SEM BIBLIOTECAS DE ML)
def euclidean_distances(X_train, x):
    # Distância euclidiana de x a cada tupla de treino: raiz(Σ_j (x_j - t_j)²)
    diff = X_train - x
    return np.sqrt(np.sum(diff * diff, axis=1))


def k_nearest_indices(distances, k):
    # Índices dos k vizinhos mais próximos, do mais próximo ao mais distante.
    # Como os atributos são inteiros, é comum várias tuplas empatarem na k-ésima menor
    # distância. Entram todas as que estão mais perto que ela e, entre as empatadas, as de
    # menor índice no treino, então o resultado é o mesmo em qualquer máquina.
    kth_distance = np.partition(distances, k - 1)[k - 1]
    closer = np.flatnonzero(distances < kth_distance)
    tied = np.flatnonzero(distances == kth_distance)[: k - len(closer)]
    k_indices = np.concatenate([closer, tied])

    return k_indices[np.argsort(distances[k_indices], kind="stable")]


def majority_vote(neighbor_labels, num_classes):
    # Classe mais votada entre os vizinhos (ordenados do mais próximo ao mais distante).
    # Empate (ex.: 5 x 5 com k = 10): vence a classe empatada do vizinho mais próximo.
    votes = np.bincount(neighbor_labels, minlength=num_classes)
    tied = votes == votes.max()
    for label in neighbor_labels:
        if tied[label]:
            return label


def run_knn_traditional(X_train, y_train, X_test, k=10, num_classes=2):
    predictions = np.empty(len(X_test), dtype=np.int64)
    for i, x_t in enumerate(tqdm(X_test, desc=f"kNN Tradicional (k={k})")):
        distances = euclidean_distances(X_train, x_t)
        k_indices = k_nearest_indices(distances, k)
        predictions[i] = majority_vote(y_train[k_indices], num_classes)

    return predictions


# 4. CONSULTA DE CONTAGEM NO RAIO r (c_{x,r}(I) DO ARTIGO)
def count_within_radius(X_train, y_train, X_test, r=6.0, num_classes=2):
    # counts[i, c] = quantas tuplas de treino da classe c estão a distância <= r da tupla de
    # teste i. São as respostas exatas da consulta, calculadas uma vez só: o ruído de
    # Laplace é somado depois, para cada ε.
    counts = np.empty((len(X_test), num_classes), dtype=np.int64)
    for i, x_t in enumerate(tqdm(X_test, desc=f"Contagens no raio (r={r:g})")):
        within_radius = euclidean_distances(X_train, x_t) <= r
        counts[i] = np.bincount(y_train[within_radius], minlength=num_classes)

    return counts


# 5. MECANISMO DE LAPLACE E kNN DIFERENCIALMENTE PRIVADO
def laplace_noise(scale, size, rng):
    # Amostras de Lap(0, scale) pela transformada inversa da CDF:
    # se U ~ Uniforme(-1/2, 1/2), então -scale · sinal(U) · ln(1 - 2|U|) ~ Lap(0, scale)
    u = rng.random(size) - 0.5
    return -scale * np.sign(u) * np.log(1 - 2 * np.abs(u))


def laplace_mechanism(true_value, sensitivity, epsilon, rng):
    # M(D) = f(D) + Lap(0, Δf/ε): ruído de média 0 e escala Δf/ε em cada posição
    return true_value + laplace_noise(sensitivity / epsilon, np.shape(true_value), rng)


def algorithm1_noisy_counts(counts, epsilon_query, rng):
    # Algoritmo 1 de Gursoy et al. (2017), adaptado como pede o enunciado: devolve as
    # contagens ruidosas de cada classe, e não o argmax. Cada linha de counts é uma chamada
    # (uma tupla de teste) com orçamento ε'. As L contagens da mesma tupla dividem ε'
    # (composição sequencial), então cada uma usa ε'/L.
    num_classes = counts.shape[1]
    return laplace_mechanism(counts, SENSITIVITY, epsilon_query / num_classes, rng)


def run_knn_dp(counts, epsilon, rng):
    # As |X_test| tuplas de teste consultam a mesma base de treino, então dividem ε
    # (composição sequencial): ε' = ε / |X_test| por tupla, ou seja,
    # ε / (|X_test| · L) em cada chamada do mecanismo de Laplace.
    epsilon_query = epsilon / len(counts)
    noisy_counts = algorithm1_noisy_counts(counts, epsilon_query, rng)

    # Classe prevista = a de maior contagem ruidosa (pós-processamento: não gasta orçamento)
    return noisy_counts.argmax(axis=1), noisy_counts


# 6. AVALIAÇÃO, ARQUIVOS DE SAÍDA E GRÁFICO
def calculate_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


def save_predictions(path, y_true, y_pred, class_names, noisy_counts=None):
    # Uma linha por tupla de teste: id, classe real e classe prevista. Nos arquivos do kNN
    # DP também vão as contagens ruidosas de cada classe (a saída do Algoritmo 1 adaptado).
    df = pd.DataFrame(
        {
            "id_teste": np.arange(len(y_true)),
            "classe_real": class_names[y_true],
            "classe_prevista": class_names[y_pred],
        }
    )
    if noisy_counts is not None:
        for c, name in enumerate(class_names):
            df[f"contagem_ruidosa_{name}"] = noisy_counts[:, c].round(2)
    df.to_csv(path, index=False)


def save_mapping(mapping_dict, path):
    # Tabela de codificação dos atributos não numéricos: atributo, valor original e código
    rows = [
        (col, value, code)
        for col, codes in mapping_dict.items()
        for code, value in codes.items()
    ]
    df = pd.DataFrame(rows, columns=["atributo", "valor_original", "codigo"])
    df.to_csv(path, index=False)


def format_pt(value):
    # Número com uma casa e vírgula decimal, para os rótulos do gráfico
    return f"{value:.1f}".replace(".", ",")


def plot_accuracy(epsilons, dp_accuracies, acc_trad, num_test, path):
    surface, ink, ink_secondary, axis_color, grid_color = (
        "#fcfcfb",
        "#0b0b0b",
        "#52514e",
        "#c3c2b7",
        "#e1e0d9",
    )
    dp_color = "#2a78d6"
    dp_pct = [acc * 100 for acc in dp_accuracies]
    trad_pct = acc_trad * 100

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=surface)
    ax.set_facecolor(surface)

    # Linha de referência: acurácia do kNN sem privacidade
    ax.axhline(
        trad_pct,
        color=ink_secondary,
        linestyle="--",
        linewidth=2,
        label="kNN sem privacidade",
    )
    ax.plot(
        epsilons,
        dp_pct,
        color=dp_color,
        linewidth=2,
        marker="o",
        markersize=9,
        markeredgecolor=surface,
        markeredgewidth=2,
        solid_capstyle="round",
        label="kNN diferencialmente privado (Laplace)",
    )

    # Rótulos diretos só nas pontas da curva e na linha de referência
    for x, value in [(epsilons[0], dp_pct[0]), (epsilons[-1], dp_pct[-1])]:
        ax.annotate(
            f"{format_pt(value)}%",
            (x, value),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            color=ink,
            fontsize=10,
        )
    ax.annotate(
        f"{format_pt(trad_pct)}%",
        (epsilons[-1], trad_pct),
        textcoords="offset points",
        xytext=(0, 7),
        ha="center",
        color=ink,
        fontsize=10,
    )

    ax.set_xscale("log")
    ax.set_xticks(epsilons, labels=[f"{e:g}".replace(".", ",") for e in epsilons])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlim(epsilons[0] / 1.35, epsilons[-1] * 1.35)
    ax.set_ylim(40, 90)

    ax.set_title(
        "Acurácia do kNN com e sem privacidade diferencial",
        loc="left",
        fontsize=13,
        color=ink,
        pad=26,
    )
    num_test_pt = f"{num_test:,}".replace(",", ".")
    ax.text(
        0,
        1.03,
        f"Adult Income · k = {K} · raio r = {RADIUS:g} · {num_test_pt} tuplas de teste",
        transform=ax.transAxes,
        fontsize=10,
        color=ink_secondary,
    )
    ax.set_xlabel("Orçamento de privacidade (ε)", color=ink_secondary)
    ax.set_ylabel("Acurácia no conjunto de teste (%)", color=ink_secondary)

    ax.grid(axis="y", color=grid_color, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis_color)
    ax.tick_params(colors=ink_secondary, length=0, pad=6)
    ax.legend(frameon=False, loc="center right", labelcolor=ink)

    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=surface)


def show_details(
    i,
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    mapping_dict,
    y_pred_trad,
    counts,
    dp_results,
):
    # Passo a passo de uma tupla de teste, para a demo: atributos, vizinhos e votos do kNN,
    # e contagens no raio sem e com ruído para cada ε
    class_names = mapping_dict[TARGET_COL]
    print(f"\n=== Passo a passo da tupla de teste {i} ===")
    for name, value in zip(feature_names, X_test[i]):
        if name in mapping_dict:
            print(f"  {name:<16}{mapping_dict[name][value]} (código {value})")
        else:
            print(f"  {name:<16}{value}")
    print(f"  classe real: {class_names[y_test[i]]}")

    distances = euclidean_distances(X_train, X_test[i])
    k_indices = k_nearest_indices(distances, K)
    print(f"\n{K} vizinhos mais próximos:")
    print("    #  índice treino  distância  classe")
    for rank, j in enumerate(k_indices, start=1):
        print(f"  {rank:>3}  {j:>13}  {distances[j]:>9.2f}  {class_names[y_train[j]]}")
    votes = np.bincount(y_train[k_indices], minlength=len(class_names))
    print(
        "  votos: "
        + ", ".join(f"{class_names[c]} = {v}" for c, v in enumerate(votes))
        + f"  =>  kNN prevê {class_names[y_pred_trad[i]]}"
    )

    print(
        f"\nContagens no raio r = {RADIUS:g} (sem ruído): "
        + ", ".join(f"{class_names[c]} = {n}" for c, n in enumerate(counts[i]))
    )
    for eps, (y_pred_dp, noisy_counts) in dp_results.items():
        print(
            f"  ε = {eps:<4g} contagens ruidosas: "
            + ", ".join(
                f"{class_names[c]} = {v:10.1f}" for c, v in enumerate(noisy_counts[i])
            )
            + f"  =>  kNN DP prevê {class_names[y_pred_dp[i]]}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dados",
        default=os.path.join(PASTA_PROJETO, "adult.csv"),
        help="CSV do Adult Income Dataset (padrão: adult.csv na pasta do projeto)",
    )
    parser.add_argument(
        "--saida",
        default=os.path.join(PASTA_PROJETO, "resultados"),
        help="pasta onde as predições, as tabelas e o gráfico são salvos",
    )
    parser.add_argument(
        "--semente",
        type=int,
        default=42,
        help="semente da divisão treino/teste e do ruído (padrão: 42)",
    )
    parser.add_argument(
        "--detalhar",
        type=int,
        metavar="ID",
        help="mostra o passo a passo de uma tupla de teste (útil na demo)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.saida, exist_ok=True)

    print("Carregando e pré-processando os dados...")
    X, y, feature_names, mapping = preprocess_data(args.dados)
    save_mapping(mapping, os.path.join(args.saida, "mapeamento.csv"))
    class_names = np.array(list(mapping[TARGET_COL].values()))
    num_classes = len(class_names)

    # Semente fixa para os resultados se repetirem na entrega e na demo (num sistema real o
    # ruído não pode ser previsível). O RandomState é o gerador "legado" do NumPy, cuja
    # sequência é congelada: a mesma semente dá os mesmos números em qualquer versão.
    rng = np.random.RandomState(args.semente)
    X_train, X_test, y_train, y_test = train_test_split_custom(X, y, rng, test_size=0.3)
    print(
        f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras | "
        f"Classes (L = {num_classes}): {', '.join(class_names)}\n"
    )
    if args.detalhar is not None and not 0 <= args.detalhar < len(X_test):
        raise SystemExit(f"--detalhar precisa estar entre 0 e {len(X_test) - 1}")

    # --- 1. Executar kNN Tradicional (k=10) ---
    y_pred_trad = run_knn_traditional(
        X_train, y_train, X_test, k=K, num_classes=num_classes
    )
    acc_trad = calculate_accuracy(y_test, y_pred_trad)
    print(f"Acurácia kNN Tradicional: {acc_trad * 100:.2f}%\n")
    save_predictions(
        os.path.join(args.saida, "predicoes_knn_tradicional.csv"),
        y_test,
        y_pred_trad,
        class_names,
    )

    # --- 2. Contagens exatas no raio (a consulta que o kNN DP perturba) ---
    counts = count_within_radius(
        X_train, y_train, X_test, r=RADIUS, num_classes=num_classes
    )
    total = counts.sum(axis=1)
    print(
        f"Vizinhos no raio r={RADIUS:g}: média {total.mean():.0f}, "
        f"mediana {np.median(total):.0f}, "
        f"{np.mean(total == 0) * 100:.1f}% das tuplas de teste sem nenhum"
    )
    # Diferença entre a maior e a segunda maior contagem: é o que o ruído precisa superar
    # para mudar a classe prevista
    sorted_counts = np.sort(counts, axis=1)
    gap = sorted_counts[:, -1] - sorted_counts[:, -2]
    print(f"Diferença entre as duas maiores contagens no raio: mediana {np.median(gap):.0f}")
    acc_radius = calculate_accuracy(y_test, counts.argmax(axis=1))
    print(
        "Referência (não privada): classificar pelas contagens no raio sem ruído dá "
        f"{acc_radius * 100:.2f}%, o teto do kNN DP quando ε → ∞\n"
    )

    # --- 3. Executar kNN DP para cada ε ---
    num_calls = len(X_test) * num_classes
    print(
        f"kNN DP: cada chamada do Laplace usa ε / (|X_test| · L) = ε / {num_calls}, "
        f"com sensibilidade Δf = {SENSITIVITY:g}"
    )
    summary = []
    dp_accuracies = []
    dp_results = {}
    for eps in EPSILONS:
        y_pred_dp, noisy_counts = run_knn_dp(counts, eps, rng)
        dp_results[eps] = (y_pred_dp, noisy_counts)
        save_predictions(
            os.path.join(args.saida, f"predicoes_knn_dp_eps_{eps}.csv"),
            y_test,
            y_pred_dp,
            class_names,
            noisy_counts,
        )

        epsilon_call = eps / num_calls
        scale = SENSITIVITY / epsilon_call
        acc_dp = calculate_accuracy(y_test, y_pred_dp)
        dp_accuracies.append(acc_dp)
        # Comparação entre a predição original (kNN sem ruído) e a predição privada
        agreement = calculate_accuracy(y_pred_trad, y_pred_dp)
        summary.append(
            {
                "epsilon": eps,
                "epsilon_por_chamada": float(f"{epsilon_call:.6g}"),
                "escala_ruido": round(scale, 1),
                "acuracia_knn_dp": round(acc_dp, 4),
                "concordancia_com_knn": round(agreement, 4),
                "acuracia_knn": round(acc_trad, 4),
            }
        )
        print(
            f"  ε = {eps:<4g} ε por chamada = {epsilon_call:.3e}  "
            f"escala do ruído = {scale:8.1f}  acurácia {acc_dp * 100:.2f}%  "
            f"concordância com o kNN {agreement * 100:.2f}%"
        )

    # Só para discussão (NÃO é ε-DP no total): cada tupla de teste usando o ε inteiro, sem
    # dividir por |X_test|. Cada predição isolada seria ε-DP, mas o conjunto de teste
    # inteiro gastaria |X_test| · ε. Usa outro gerador para não mudar os resultados acima.
    rng_discussion = np.random.RandomState(args.semente + 1)
    print("\nSó para discussão: sem dividir ε por |X_test| (NÃO é ε-DP no total)")
    for eps in EPSILONS:
        noisy_counts = algorithm1_noisy_counts(counts, eps, rng_discussion)
        acc = calculate_accuracy(y_test, noisy_counts.argmax(axis=1))
        print(f"  ε = {eps:<4g} acurácia {acc * 100:.2f}%")

    # --- 4. Tabela de acurácias e gráfico ---
    pd.DataFrame(summary).to_csv(
        os.path.join(args.saida, "acuracias.csv"), index=False
    )
    plot_accuracy(
        EPSILONS,
        dp_accuracies,
        acc_trad,
        len(X_test),
        os.path.join(args.saida, "grafico_acuracia_dp.png"),
    )
    print(f"\nPredições, tabelas e gráfico salvos em {args.saida}")

    if args.detalhar is not None:
        show_details(
            args.detalhar,
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            mapping,
            y_pred_trad,
            counts,
            dp_results,
        )

    plt.show()


if __name__ == "__main__":
    main()
