from typing import Callable, Iterable, List
import matplotlib.pyplot as plt
import numpy as np

from sistemas_lineares import gauss_pivot_parcial


def prod(iterable: Iterable[float]) -> float:
    r = 1
    for x in iterable:
        r *= x

    return r


def interpolacao_sistema(x: List[float], y: List[float]) -> Callable[[float], float]:
    """

    Encontra os coeficientes `(a0, a1, ..., an)` do polinômio
    interpolador dos pontos tabelados por `x` e `y`.

    Resolve-se o seguinte sistema linear:

    ```py
    a0 + x0*a1 + x0*a2 + ... + x0*an = y0
    a0 + x1*a1 + x1*a2 + ... + x1*an = y1
    ...
    a0 + xn*a1 + xn*a2 + ... + xn*an = yn
    ```

    """
    n = len(x)

    # matriz n por n contendo apenas 1s
    A = [[1] * n for _ in range(n)]

    # gerar matriz a partir da segunda coluna
    # pois primeira coluna possui somente 1s
    for i in range(n):
        for j in range(1, n):
            A[i][j] = x[i] ** j

    # calcular b e montar matriz aumentada [A b]
    for i in range(n):
        A[i].append(y[i])

    a = gauss_pivot_parcial(A)
    g = lambda x: sum(a[k] * x**k for k in range(n))

    print(a)

    return g


def interpolacao_lagrange(x: List[float], y: List[float]) -> Callable[[float], float]:
    """
    Encontra o polinômio interpolador utilizando o método de Lagrange.

    ```py
    P(x) = y0*L[n,0](x) + y1*L[n,1](x) + ... + yn*L[n,n](x)

    L[n,j](x) = prod( (x - x[i]) / (x[j] - x[i]) for i in range(n) if i != j)
    ```

    """
    n = len(x)

    def g(xp):
        soma = 0
        for j in range(n):
            L = prod((xp - x[i]) / (x[j] - x[i]) for i in range(n) if i != j)
            soma += y[j] * L

        return soma

    return g


def interpolacao_newton(x: List[float], y: List[float]) -> Callable[[float], float]:
    """
    Encontra o polinômio interpolador dos pontos tabelados por `x` e `y`
    utilizando o método das diferenças divididas de Newton.

    Operador das diferenças divididas de uma função f:
    ```py
    f[xn] = f(xn)
    f[x0, ..., xn] = (f[x1, ..., xn] - f[x0, ..., xn-1]) / (xn - x0)

    an = f[x0, ..., xn]
    ```

    Polinômio interpolador:
    ```py
    P(x) = a0 + a1(x - x0) + a2(x - x0)(x - x1) + ... + an(x - x0)...(x - xn)
    ```
    """

    n = len(x)
    f = {}

    for b in range(n):  # 0, 1, 2, ..., n-1
        for a in range(b, -1, -1):  # b, b-1, b-2, ..., 0
            if a == b:
                f[a, b] = y[a]
            else:
                f[a, b] = (f[a + 1, b] - f[a, b - 1]) / (x[b] - x[a])

    a = [f[0, i] for i in range(n)]
    return lambda xp: sum(a[i] * prod(xp - x[j] for j in range(i)) for i in range(n))


if __name__ == "__main__":
    x = [-3, -2, 1, 3]
    y = [-1, 2, -1, 10]

    f = interpolacao_sistema(x, y)
    g = interpolacao_lagrange(x, y)
    h = interpolacao_newton(x, y)

    plt.plot(x, y, "o")

    x_range = np.linspace(x[0], x[-1], 50)

    plt.plot(x_range, [f(xi) for xi in x_range], "r--")
    plt.plot(x_range, [g(xi) for xi in x_range], "g:")
    plt.plot(x_range, [h(xi) for xi in x_range], "y.")

    plt.show()
