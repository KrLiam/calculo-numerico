from typing import Callable

from resources import read_json

def integrar_por_trapezios(
    f: Callable[[float], float], a: float, b: float, p: int = 1
) -> float:
    h = (b - a) / p

    soma = sum(f(a + i * h) for i in range(1, p))

    return (f(a) + 2 * soma + f(b)) * h / 2


ABCISSAE = read_json("abcissae.json")
WEIGHTS = read_json("weights.json")

def integrar_por_quadratura(
    f: Callable[[float], float], a: float, b: float, n: float
) -> float:
    """
    Resolve a integral definida `∫ a:b f(x)dx` pelo método da quadratura gaussiana.

    Argumentos:
    
    `f`: Função a ser integrada;
    `a`, `b`: Limites de integração;
    `n`: Número de pontos da quadratura;

    Padroniza o limite de integração de `[a, b]` para `[-1, 1]` a
    partir da troca de variáveis de `x` para `t`:

    ```py
    x = (1/2)(b - a)t + (1/2)(b + a)
    dx = (1/2)(b - a)dt
    ```

    Substitui `x` e `dx` na integral e utiliza um somatório para realizar
    a aproximação:

    ```py
    ∫ a:b f(x)dx
    = ∫ -1:1 F(t)dt
    ≃ ∑ i=1:n (A[i] * F(t[i]))
    ```

    Sendo `F(t) = f(x(t)) * (1/2)(b - a)`; `A` e `t` vetores de
    pesos e valores de t aproximados pela quadratura de
    [Lengrende-Gauss](https://pomax.github.io/bezierinfo/legendre-gauss.html).
    """

    x = lambda t: ((b - a)*t + (b + a)) / 2
    F = lambda t: f(x(t)) * (b - a) / 2

    A = WEIGHTS[n]
    T = ABCISSAE[n]

    return sum(a * F(t) for a, t in zip(A, T))


if __name__ == "__main__":
    from math import cos, sin

    f = lambda x: x**2 + 5*cos(x)
    a, b = 0, 4

    F = lambda x: x**3 / 3 + 5*sin(x)
    print("exato: ", F(4) - F(0), "\n")

    for n in range(2, 10):
        r1 = integrar_por_trapezios(f, a, b, n)
        print(f"trapezios, {n} pontos:", r1)

        r2 = integrar_por_quadratura(f, a, b, n)
        print(f"quadratura, {n} pontos:", r2, "\n")
