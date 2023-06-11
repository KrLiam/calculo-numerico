#include <iostream>
#include <tgmath.h>
#include <functional>


double integrar_por_trapezios(
    std::function<double(double)> f, double a, double b, int p = 1
) {
    double h = (b - a) / p;

    double soma = 0;
    for (int i = 1; i < p; i++) {
        soma += f(a + i*h);
    }
    
    return (f(a) + 2*soma + f(b)) * h / 2;
}

int main() {
    auto f = [](double x) { return std::pow(x, 2); };

    auto r = integrar_por_trapezios(f, 0, 4, 100000000);
    std::cout << r << std::endl;
}