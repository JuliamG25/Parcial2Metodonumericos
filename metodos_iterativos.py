"""
Métodos iterativos: Jacobi y Gauss-Seidel para sistemas Ax = b.
"""

from __future__ import annotations

from typing import Callable, List, Tuple


def norma_inf_diferencia(a: List[float], b: List[float]) -> float:
    return max(abs(a[i] - b[i]) for i in range(len(a))) if a else 0.0


def jacobi(
    A: List[List[float]],
    b: List[float],
    x0: List[float],
    tol: float,
    max_iter: int,
) -> Tuple[List[float], List[Tuple[int, List[float], float]], str | None]:
    """
    Retorna (solución aproximada, historial [(iter, x, error)], error_msg o None).
    """
    n = len(b)
    if any(A[i][i] == 0 for i in range(n)):
        return x0, [], "Error: hay un cero en la diagonal principal. No se puede aplicar Jacobi."

    x = x0[:]
    historial: List[Tuple[int, List[float], float]] = []

    for k in range(1, max_iter + 1):
        x_nuevo = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_nuevo[i] = (b[i] - s) / A[i][i]
        err = norma_inf_diferencia(x_nuevo, x)
        historial.append((k, x_nuevo[:], err))
        if err < tol:
            return x_nuevo, historial, None
        x = x_nuevo

    return x, historial, None


def gauss_seidel(
    A: List[List[float]],
    b: List[float],
    x0: List[float],
    tol: float,
    max_iter: int,
) -> Tuple[List[float], List[Tuple[int, List[float], float]], str | None]:
    n = len(b)
    if any(A[i][i] == 0 for i in range(n)):
        return x0, [], "Error: hay un cero en la diagonal principal. No se puede aplicar Gauss-Seidel."

    x = x0[:]
    historial: List[Tuple[int, List[float], float]] = []

    for k in range(1, max_iter + 1):
        x_anterior = x[:]
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]
        err = norma_inf_diferencia(x, x_anterior)
        historial.append((k, x[:], err))
        if err < tol:
            return x, historial, None

    return x, historial, None


def ejecutar_metodo(
    nombre: str,
    A: List[List[float]],
    b: List[float],
    x0: List[float],
    tol: float,
    max_iter: int,
) -> Tuple[List[float], List[Tuple[int, List[float], float]], str | None]:
    if nombre == "jacobi":
        return jacobi(A, b, x0, tol, max_iter)
    if nombre == "gauss_seidel":
        return gauss_seidel(A, b, x0, tol, max_iter)
    raise ValueError(f"Método desconocido: {nombre}")
