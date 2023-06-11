from typing import Callable, List
import matplotlib.pyplot as plt
from math import log, e
import numpy as np

from sistemas_lineares import gauss_pivot_parcial


def polinomio(c: List[float]) -> Callable[[float], float]:
    """
    Retorna uma função que calcula o polinômio com coeficientes
    iguais ao vetor `c`.
    """

    return lambda x: sum(c[i] * x**i for i in range(len(c)))


def ajuste_polinomial(x: List[float], y: List[float], M: int = 2) -> List[float]:
    """
    Retorna o vetor de coeficientes de tamanho `M + 1` do
    polinômio de ajuste.
    """

    n = len(x)

    A = [[0] * (M + 1) for _ in range(M + 1)]

    # gerar matriz
    for i in range(M + 1):
        for j in range(i, M + 1):
            A[i][j] = sum(x[k] ** (i + j) for k in range(n))
            A[j][i] = A[i][j]

    # print("A =", A)

    # calcular b e montar matriz aumentada [A b]
    for i in range(M + 1):
        b = sum(y[k] * x[k] ** i for k in range(n))
        A[i].append(b)

    return gauss_pivot_parcial(A)


def ajuste_exponencial(x: List[float], y: List[float]) -> List[float]:
    """
    Encontra funções de ajuste da forma `y = a * e**(b*x)`.

    Calcula o vetor `z = ln(y)`, linearização de `y`. Encontra os
    coeficientes do polinômio de grau 1, `z = ln(a) + bx`, a
    partir de `x` e `z` utilizando o método de ajuste polinomial.

    Retorna o vetor de coeficientes do polinômio de ajuste `a` e `b`.
    """

    z = [log(y_i) for y_i in y]

    c1, c2 = ajuste_polinomial(x, z, M=1)

    return [e**c1, c2]


def ajuste_potencia(x: List[float], y: List[float]) -> List[float]:
    """
    Encontra funções de ajuste da forma `y = a*x**b`.

    Aplicando o logaritmo natural em ambos os lados da equação acima:
    ```py
    ln(y) = ln(a) + b*ln(x)
    ```

    Transforma-se em uma função linear:

    ```py
    z = ln(a) + b*t, z = ln(y) e t = ln(x)
    ```

    Que pode ser resolvido com um ajuste polinomial de grau 1.


    Retorna o vetor de coeficientes do polinômio de ajuste `a` e `b`.
    """

    z = [log(y_i) for y_i in y]
    t = [log(x_i) for x_i in x]

    c1, c2 = ajuste_polinomial(t, z, M=1)

    return [e**c1, c2]


if __name__ == "__main__":
    x = [1.3, 3.4, 5.1, 6.8, 8]
    y = [2, 5.2, 3.8, 6.1, 5.8]

    c = ajuste_polinomial(x, y, 3)
    g = polinomio(c)

    plt.plot(x, y, "o")

    x_range = np.linspace(x[0], x[-1], 50)
    plt.plot(x_range, [g(xi) for xi in x_range])

    plt.show()
