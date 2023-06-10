from sistemas_lineares import gauss_pivot_parcial


def get_J(F, X, h=0.01):
    F_X = F(X)

    lins = len(F_X)
    cols = len(X)

    J = [[0] * cols for _ in range(lins)]

    # para cada coluna de J
    for j in range(cols):
        # aproximar derivadas parciais em relação a xj
        f = F([x + h if k == j else x for k, x in enumerate(X)])
        for i in range(lins):
            J[i][j] = (f[i] - F_X[i]) / h

    return J


def newton_nao_linear(F, X_i, h=0.01):
    """
    F: função que retorna um vetor
    X_i: hipótese de x inicial
    h: fator de aproximação das derivadas parciais
    """

    n = len(X_i)
    d = [1] * n

    erro = 10**-6
    X = list(X_i)

    k = 0
    while sum(abs(v) for v in d) >= erro:
        k += 1

        # sistema: J * d = -F
        # criar matriz aumentada [J -F]
        matriz = get_J(F, X, h)
        for i, x in enumerate(F(X)):
            matriz[i].append(-x)

        print(f"k - {k}", f"x - {X}", "matriz aumentada", *matriz, sep="\n")

        # resolver sistema linear
        d = gauss_pivot_parcial(matriz)

        print(f"d {d}", end="\n\n")

        # x^(k+1) = d + x^k
        for i, di in enumerate(d):
            X[i] += di

    return X


if __name__ == "__main__":
    X_i = [0.5, 0.5, 0.5]
    F = lambda X: (
        X[0] ** 2 + X[1] ** 2 + X[2] ** 2 - 1,
        2 * X[0] ** 2 + X[1] ** 2 - 4 * X[2],
        3 * X[0] ** 2 - 4 * X[1] + X[2] ** 2 - 3,
    )
    x = newton_nao_linear(F, X_i, 0.001)

    print(x)
