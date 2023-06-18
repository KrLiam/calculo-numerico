from typing import Callable, List
import matplotlib.pyplot as plt
from math import e


def frange(start: float, end: float, step: float):
    n = int((end - start) / step) + 1

    return [start + i * step for i in range(n)]


def edo_por_euler(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    y_a: float,
    # n: float | None = None,
    h: float,  # | None = None,
) -> List[float]:
    """
    Resolve discretamente equações diferenciais de 1ª ordem do tipo `y' = f(x, y)`.

    Argumentos:
    `f`: Função da equação;
    `a, b`: Intervalo tal que `a <= x <= b`;
    `y_a`: Valor de y no ponto inicial `(a, y(a))`;
    `h`: Distância de x entre os pontos calculados;

    Utiliza os 2 primeiros termos da série de Taylor para determinar os
    valores de y partindo do ponto inicial `(a, y(a))`:
    ```
    y(x_i+1) = y(x_i) + h * y'(x_i)
    ```

    Retorna um vetor com `(b - a) / h + 1` elementos, `y(a), y(a+h), y(a+2*h), ..., y(b)`,
    que representa a função y de forma discreta.
    """

    # número de pontos
    x = frange(a, b, h)
    p = len(x)

    y = [0] * p

    # ponto inicial
    y[0] = y_a

    for i in range(p - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])

    return y


def edo_por_runge_kutta_2(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    y_a: float,
    h: float,
) -> List[float]:
    """
    Resolve discretamente equações diferenciais de 1ª ordem do tipo `y' = f(x, y)`.

    Argumentos:
    - `f`: Função da equação;
    - `a, b`: Intervalo tal que `a <= x <= b`;
    - `y_a`: Valor de y no ponto inicial `(a, y(a))`;
    - `h`: Distância de x entre os pontos calculados;
    """

    # número de pontos
    x = frange(a, b, h)
    p = len(x)

    y = [0] * p

    # ponto inicial
    y[0] = y_a

    for i in range(p - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h, y[i] + k1)

        y[i + 1] = y[i] + (k1 + k2) / 2

        # print(f"{x[i]=:.2f}, {k1=:.5f}, {k2=:.5f} => {y[i+1]=:.5f}")

    return y


def edo_por_runge_kutta_4(
    f: Callable[[float, float], float],
    a: float,
    b: float,
    y_a: float,
    h: float,
) -> List[float]:
    """
    Resolve discretamente equações diferenciais de 1ª ordem do tipo `y' = f(x, y)`.

    Argumentos:
    - `f`: Função da equação;
    - `a, b`: Intervalo tal que `a <= x <= b`;
    - `y_a`: Valor de y no ponto inicial `(a, y(a))`;
    - `h`: Distância de x entre os pontos calculados;
    """

    # número de pontos
    x = frange(a, b, h)
    p = len(x)

    y = [0] * p

    # ponto inicial
    y[0] = y_a

    for i in range(p - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # print(f"{x[i]=:.2f}, {k1=:.5f}, {k2=:.5f}, {k3=:.5f}, {k4=:.5f} => {y[i+1]=:.5f}")

    return y


if __name__ == "__main__":
    a = 0
    b = 0.5
    h = 0.1

    f = lambda x, y: -2 * x - y
    y_exato = lambda x: -3 * e**-x - 2 * x + 2

    euler = edo_por_euler(f, a, b, -1, h)
    runge_kutta2 = edo_por_runge_kutta_2(f, a, b, -1, h)
    runge_kutta4 = edo_por_runge_kutta_4(f, a, b, -1, h)

    print(f"euler: {euler}")
    print(f"runge kutta 2 ordem: {runge_kutta2}")
    print(f"runge kutta 4 ordem: {runge_kutta4}")

    x = frange(0, b, h)

    plt.plot(x, euler, "r")
    plt.plot(x, runge_kutta2, "g")
    plt.plot(x, runge_kutta4, "b")
    plt.plot(x, [y_exato(x_i) for x_i in x], "y")

    plt.show()
