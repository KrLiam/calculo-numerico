from typing import Callable

from resources import read_json


def integrar_por_trapezios(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 1,
) -> float:
    """
    Resolve a integral definida `∫ a:b f(x)dx` pelo método dos trapézios.

    Argumentos:

    `f`: Função a ser integrada;
    `a`, `b`: Limites de integração;
    `n`: Número de sub-divisões do intervalo de integração (`n + 1` pontos);

    Subdivide a integral de intervalo `[a, b]` em `n` integrais
    com sub-intervalos de tamanho `h = (b - a) / n`:

    ```
    ∫ a:b f(x)dx = ∫ a:x1 f(x)dx + ∫ x1:x2 f(x)dx + ... + ∫ xn:b f(x)dx
    ```

    Que pode ser aproximado com a formula:

    ```
    ∫ a:b f(x)dx ≃ h/2 * [f(a) + 2(∑ i=1:n-1 f(xi)) + f(b)]
    ```
    """

    h = (b - a) / n

    soma = sum(f(a + i * h) for i in range(1, n))

    return (f(a) + 2 * soma + f(b)) * h / 2


def integrar_por_simpson(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: int = 2,
) -> float:
    """
    Resolve a integral definida `∫ a:b f(x)dx` pelo método de simpson.

    Argumentos:

    `f`: Função a ser integrada;
    `a`, `b`: Limites de integração;
    `n`: Número de sub-divisões do intervalo de integração. Se o
    valor fornecido for ímpar, `n + 1` será utilizado;

    ```
    ∫ a:b f(x)dx ≃ h/3 * [f(a) + 4*f(x1) + 2*f(x2) + 4*f(x3) + ... + 2*f(xp-2) + 4*f(xp-1) + f(b)]
    ```
    """

    n += n % 2
    h = (b - a) / n

    x = [a + i*h for i in range(0, n+1)]

    soma_imps = sum(f(x[i]) for i in range(1, n, 2))
    soma_pares = sum(f(x[i]) for i in range(2, n, 2))

    return (f(a) + 4 * soma_imps + 2 * soma_pares + f(b)) * h / 3



ABCISSAE = read_json("abcissae.json")
WEIGHTS = read_json("weights.json")


def integrar_por_quadratura(
    f: Callable[[float], float],
    a: float,
    b: float,
    n: float = 2,
) -> float:
    """
    Resolve a integral definida `∫ a:b f(x)dx` pelo método da quadratura gaussiana.

    Argumentos:

    `f`: Função a ser integrada;
    `a`, `b`: Limites de integração;
    `n`: Número de pontos da quadratura;

    Padroniza o limite de integração de `[a, b]` para `[-1, 1]` a
    partir da troca de variáveis de `x` para `t`:

    ```
    x = (1/2)(b - a)t + (1/2)(b + a)
    dx = (1/2)(b - a)dt
    ```

    Substitui `x` e `dx` na integral e utiliza um somatório para realizar
    a aproximação:

    ```
    ∫ a:b f(x)dx
    = ∫ -1:1 F(t)dt
    ≃ ∑ i=0:n-1 (A[i] * F(T[i]))
    ```

    Sendo `F(t) = f(x(t)) * (1/2)(b - a)`; `A` e `T` vetores de
    pesos e valores de t aproximados pela quadratura de
    [Legrende-Gauss](https://pomax.github.io/bezierinfo/legendre-gauss.html).
    """

    x = lambda t: ((b - a) * t + (b + a)) / 2
    F = lambda t: f(x(t)) * (b - a) / 2

    A = WEIGHTS[n]
    T = ABCISSAE[n]

    return sum(a * F(t) for a, t in zip(A, T))


if __name__ == "__main__":
    from math import cos, sin

    f = lambda x: x**2 + 5 * cos(x)
    a, b = 0, 4

    F = lambda x: x**3 / 3 + 5 * sin(x)
    print("exato: ", F(4) - F(0), "\n")

    for p in range(3, 14, 2):
        r1 = integrar_por_trapezios(f, a, b, p-1)
        print(f"trapezios, {p} pontos:", r1)

        r2 = integrar_por_simpson(f, a, b, p-1)
        print(f"simpson, {p + 1 - p % 2} pontos:", r2)

        r2 = integrar_por_quadratura(f, a, b, p)
        print(f"quadratura, {p} pontos:", r2, "\n")
