from typing import Callable


def integrar_por_trapezios(
    f: Callable[[float], float], a: float, b: float, p: int = 1
) -> float:
    h = (b - a) / p

    soma = sum(f(a + i*h) for i in range(1, p))
    
    return (f(a) + 2*soma + f(b)) * h / 2

if __name__ == "__main__":
    r = integrar_por_trapezios(lambda x: x**2, 0, 4, 100_000_000)
    print(r)

