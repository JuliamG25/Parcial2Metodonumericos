"""
Métodos iterativos: Jacobi y Gauss-Seidel para sistemas Ax = b.

Error absoluto: norma infinito de la diferencia entre dos iterados consecutivos.
Error porcentual: (error absoluto / ‖x nuevo‖∞) × 100, con denominador acotado para evitar división por cero.
"""

from __future__ import annotations

from typing import List, Tuple

# Tipo historial: (iteración, vector x, error absoluto, error porcentual)
HistorialIter = Tuple[int, List[float], float, float]


def norma_inf_diferencia(a: List[float], b: List[float]) -> float:
    return max(abs(a[i] - b[i]) for i in range(len(a))) if a else 0.0


def norma_inf_vector(v: List[float]) -> float:
    return max(abs(t) for t in v) if v else 0.0


def error_porcentual(err_abs: float, x_referencia: List[float], eps: float = 1e-15) -> float:
    """
    Error porcentual respecto a la magnitud del vector actual (‖x‖∞).
    ep = (E_abs / max(‖x‖∞, eps)) × 100
    """
    denom = max(norma_inf_vector(x_referencia), eps)
    return (err_abs / denom) * 100.0


def criterio_convergencia(err_abs: float, err_pct: float, tol_abs: float, tol_pct: float) -> bool:
    return err_abs < tol_abs and err_pct < tol_pct


def jacobi(
    A: List[List[float]],
    b: List[float],
    x0: List[float],
    tol_abs: float,
    tol_pct: float,
    max_iter: int,
) -> Tuple[List[float], List[HistorialIter], str | None]:
    """
    Retorna (solución aproximada, historial, mensaje_error o None).
    """
    n = len(b)
    if any(A[i][i] == 0 for i in range(n)):
        return x0, [], "Error: hay un cero en la diagonal principal. No se puede aplicar Jacobi."

    x = x0[:]
    historial: List[HistorialIter] = []

    for k in range(1, max_iter + 1):
        x_nuevo = [0.0] * n
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_nuevo[i] = (b[i] - s) / A[i][i]
        err_abs = norma_inf_diferencia(x_nuevo, x)
        err_pct = error_porcentual(err_abs, x_nuevo)
        historial.append((k, x_nuevo[:], err_abs, err_pct))
        if criterio_convergencia(err_abs, err_pct, tol_abs, tol_pct):
            return x_nuevo, historial, None
        x = x_nuevo

    return x, historial, None


def gauss_seidel(
    A: List[List[float]],
    b: List[float],
    x0: List[float],
    tol_abs: float,
    tol_pct: float,
    max_iter: int,
) -> Tuple[List[float], List[HistorialIter], str | None]:
    n = len(b)
    if any(A[i][i] == 0 for i in range(n)):
        return x0, [], "Error: hay un cero en la diagonal principal. No se puede aplicar Gauss-Seidel."

    x = x0[:]
    historial: List[HistorialIter] = []

    for k in range(1, max_iter + 1):
        x_anterior = x[:]
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]
        err_abs = norma_inf_diferencia(x, x_anterior)
        err_pct = error_porcentual(err_abs, x)
        historial.append((k, x[:], err_abs, err_pct))
        if criterio_convergencia(err_abs, err_pct, tol_abs, tol_pct):
            return x, historial, None

    return x, historial, None


def ejecutar_metodo(
    nombre: str,
    A: List[List[float]],
    b: List[float],
    x0: List[float],
    tol_abs: float,
    tol_pct: float,
    max_iter: int,
) -> Tuple[List[float], List[HistorialIter], str | None]:
    if nombre == "jacobi":
        return jacobi(A, b, x0, tol_abs, tol_pct, max_iter)
    if nombre == "gauss_seidel":
        return gauss_seidel(A, b, x0, tol_abs, tol_pct, max_iter)
    raise ValueError(f"Método desconocido: {nombre}")
