from typing import List


def gauss_pivot_parcial(matriz: List[List[float]]) -> List[float]:
    cols = len(matriz[0])
    lins = len(matriz)

    # para cada pivot k
    for k in range(cols - 1):
        # achar linha com maior elemento da coluna k
        highest = matriz[k][k]
        pivot_i = k
        for i in range(k + 1, lins):
            if abs(matriz[i][k]) > highest:
                pivot_i = i
                highest = abs(matriz[i][k])

        # trocar linhas k e pivot_i
        if k != pivot_i:
            matriz[k], matriz[pivot_i] = matriz[pivot_i], matriz[k]

        # print(k, *matriz, sep="\n")

        # anular coluna k do pivot para baixo
        for i in range(k + 1, lins):
            scale = -matriz[i][k] / matriz[k][k]
            # operar linha i com linha do pivot k multiplicado por scale
            for j in range(k, cols):
                matriz[i][j] += matriz[k][j] * scale

    x = [0] * lins
    for i in range(lins - 1, -1, -1):
        somatorio = sum(matriz[i][j] * x[j] for j in range(i + 1, lins))
        x[i] = (matriz[i][cols - 1] - somatorio) / matriz[i][i]

    return x
