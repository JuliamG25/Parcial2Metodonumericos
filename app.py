#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz gráfica: Jacobi y Gauss-Seidel para Ax = b.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from metodos_iterativos import ejecutar_metodo


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Métodos numéricos — Jacobi y Gauss-Seidel")
        self.minsize(720, 520)
        self.geometry("900x640")

        self._coef_entries: list[list[tk.Entry]] = []
        self._b_entries: list[tk.Entry] = []
        self._x0_entries: list[tk.Entry] = []

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")

        main = ttk.Frame(self, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        titulo = ttk.Label(
            main,
            text="Sistema lineal Ax = b — Jacobi y Gauss-Seidel",
            font=("Segoe UI", 14, "bold"),
        )
        titulo.pack(anchor=tk.W, pady=(0, 8))

        ayuda = ttk.Label(
            main,
            text="Introduce el tamaño n, pulsa «Generar matriz», rellena coeficientes A, términos independientes b "
            "(lado derecho de las ecuaciones) y, si quieres, el vector inicial x⁽⁰⁾; luego elige método y parámetros.",
            wraplength=860,
        )
        ayuda.pack(anchor=tk.W, pady=(0, 10))

        fila_n = ttk.Frame(main)
        fila_n.pack(fill=tk.X, pady=4)
        ttk.Label(fila_n, text="Número de variables (n):").pack(side=tk.LEFT, padx=(0, 8))
        self.var_n = tk.StringVar(value="3")
        spin = ttk.Spinbox(
            fila_n,
            from_=2,
            to=12,
            width=5,
            textvariable=self.var_n,
        )
        spin.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(fila_n, text="Generar matriz", command=self._generar_matriz).pack(side=tk.LEFT)

        # Panel matriz + opciones
        cuerpo = ttk.Frame(main)
        cuerpo.pack(fill=tk.BOTH, expand=True, pady=8)

        izq = ttk.LabelFrame(cuerpo, text="Matriz A, vector b y x⁽⁰⁾ (opcional)", padding=8)
        izq.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.canvas = tk.Canvas(izq, highlightthickness=0)
        scroll_y = ttk.Scrollbar(izq, orient=tk.VERTICAL, command=self.canvas.yview)
        self._matriz_frame = ttk.Frame(self.canvas)
        self._matriz_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self._matriz_frame, anchor=tk.NW)
        self.canvas.configure(yscrollcommand=scroll_y.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        def _wheel(evt: tk.Event) -> str | None:
            self.canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units")
            return "break"

        self.canvas.bind("<Enter>", lambda _e: self.canvas.bind_all("<MouseWheel>", _wheel))
        self.canvas.bind("<Leave>", lambda _e: self.canvas.unbind_all("<MouseWheel>"))

        der = ttk.LabelFrame(cuerpo, text="Método y parámetros", padding=10)
        der.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        self.var_metodo = tk.StringVar(value="jacobi")
        ttk.Radiobutton(der, text="Jacobi", variable=self.var_metodo, value="jacobi").pack(anchor=tk.W)
        ttk.Radiobutton(
            der,
            text="Gauss-Seidel",
            variable=self.var_metodo,
            value="gauss_seidel",
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(der, text="Tolerancia:").pack(anchor=tk.W)
        self.var_tol = tk.StringVar(value="0.0001")
        ttk.Entry(der, textvariable=self.var_tol, width=18).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(der, text="Iteraciones máximas:").pack(anchor=tk.W)
        self.var_max_iter = tk.StringVar(value="100")
        ttk.Entry(der, textvariable=self.var_max_iter, width=18).pack(anchor=tk.W, pady=(0, 8))

        ttk.Label(der, text="Decimales en resultados:").pack(anchor=tk.W)
        self.var_decimales = tk.StringVar(value="6")
        ttk.Spinbox(der, from_=2, to=15, width=16, textvariable=self.var_decimales).pack(anchor=tk.W, pady=(0, 12))

        ttk.Button(der, text="Calcular solución", command=self._calcular).pack(fill=tk.X, pady=4)
        ttk.Button(der, text="Limpiar resultados", command=self._limpiar_salida).pack(fill=tk.X)

        salida_frame = ttk.LabelFrame(main, text="Resultados e iteraciones", padding=8)
        salida_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self.texto = tk.Text(salida_frame, height=14, wrap=tk.WORD, font=("Consolas", 10))
        sy = ttk.Scrollbar(salida_frame, orient=tk.VERTICAL, command=self.texto.yview)
        self.texto.configure(yscrollcommand=sy.set)
        self.texto.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sy.pack(side=tk.RIGHT, fill=tk.Y)

        self._generar_matriz()

    def _limpiar_entradas_matriz(self) -> None:
        for row in self._coef_entries:
            for e in row:
                e.destroy()
        for e in self._b_entries:
            e.destroy()
        for e in self._x0_entries:
            e.destroy()
        self._coef_entries = []
        self._b_entries = []
        self._x0_entries = []

    def _generar_matriz(self) -> None:
        try:
            n = int(self.var_n.get())
        except ValueError:
            messagebox.showerror("Error", "El número de variables debe ser un entero.")
            return
        if n < 2 or n > 12:
            messagebox.showwarning("Aviso", "Usa n entre 2 y 12.")
            return

        self._limpiar_entradas_matriz()
        f = self._matriz_frame

        eq_font = ("Segoe UI", 13, "bold")
        cab = ttk.Frame(f)
        cab.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(cab, text="Ecuación").grid(row=0, column=0, padx=4)
        for j in range(n):
            ttk.Label(cab, text=f"x{j + 1}").grid(row=0, column=j + 1, padx=4)
        tk.Label(cab, text="=", font=eq_font, fg="#1a1a1a").grid(row=0, column=n + 1, padx=(10, 6))
        ttk.Label(cab, text="b").grid(row=0, column=n + 2, padx=4)
        ttk.Label(cab, text="x⁽⁰⁾ (opc.)").grid(row=0, column=n + 3, padx=8)

        for i in range(n):
            row_e: list[tk.Entry] = []
            ttk.Label(f, text=f"{i + 1}").grid(row=i + 1, column=0, padx=4, pady=2)
            for j in range(n):
                e = ttk.Entry(f, width=10)
                e.grid(row=i + 1, column=j + 1, padx=2, pady=2)
                row_e.append(e)
            self._coef_entries.append(row_e)
            tk.Label(f, text="=", font=eq_font, fg="#1a1a1a").grid(
                row=i + 1, column=n + 1, padx=(10, 6), pady=2
            )
            eb = ttk.Entry(f, width=10)
            eb.grid(row=i + 1, column=n + 2, padx=4, pady=2)
            self._b_entries.append(eb)
            ex = ttk.Entry(f, width=10)
            ex.insert(0, "0")
            ex.grid(row=i + 1, column=n + 3, padx=8, pady=2)
            self._x0_entries.append(ex)

        self.canvas.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _leer_float(self, s: str, nombre: str) -> float:
        s = s.strip().replace(",", ".")
        if s == "":
            return 0.0
        try:
            return float(s)
        except ValueError as exc:
            raise ValueError(f"Valor no numérico en {nombre}: {s!r}") from exc

    def _leer_sistema(self) -> tuple[list[list[float]], list[float], list[float]]:
        n = len(self._b_entries)
        if n == 0:
            raise ValueError("Genera primero la matriz.")
        A: list[list[float]] = []
        b: list[float] = []
        x0: list[float] = []
        for i in range(n):
            fila: list[float] = []
            for j in range(n):
                fila.append(self._leer_float(self._coef_entries[i][j].get(), f"A[{i + 1},{j + 1}]"))
            A.append(fila)
            b.append(self._leer_float(self._b_entries[i].get(), f"b[{i + 1}]"))
            x0.append(self._leer_float(self._x0_entries[i].get(), f"x0[{i + 1}]"))
        return A, b, x0

    def _calcular(self) -> None:
        try:
            A, b, x0 = self._leer_sistema()
            tol = self._leer_float(self.var_tol.get(), "tolerancia")
            max_iter = int(self.var_max_iter.get())
            dec = int(self.var_decimales.get())
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        if tol <= 0:
            messagebox.showerror("Error", "La tolerancia debe ser positiva.")
            return
        if max_iter < 1:
            messagebox.showerror("Error", "Las iteraciones máximas deben ser ≥ 1.")
            return

        metodo = self.var_metodo.get()
        sol, hist, err_msg = ejecutar_metodo(metodo, A, b, x0, tol, max_iter)

        self.texto.delete("1.0", tk.END)
        if err_msg:
            self.texto.insert(tk.END, err_msg + "\n")
            return

        fmt = f"{{:.{dec}f}}"
        nombre = "Jacobi" if metodo == "jacobi" else "Gauss-Seidel"
        self.texto.insert(tk.END, f"Método: {nombre}\n")
        self.texto.insert(tk.END, f"Tolerancia: {tol}  |  Máx. iteraciones: {max_iter}  |  Decimales: {dec}\n\n")

        if hist:
            ultimo_err = hist[-1][2]
            convergio = ultimo_err < tol
            self.texto.insert(
                tk.END,
                "Estado: convergió dentro de la tolerancia.\n\n"
                if convergio
                else "Estado: se alcanzó el máximo de iteraciones sin cumplir la tolerancia (revisa datos o aumenta iteraciones).\n\n",
            )

        self.texto.insert(tk.END, "Solución aproximada x:\n")
        for i, val in enumerate(sol):
            self.texto.insert(tk.END, f"  x{i + 1} = {fmt.format(val)}\n")
        self.texto.insert(tk.END, "\n--- Iteraciones ---\n")
        self.texto.insert(tk.END, f"{'Iter':>6}  {'Error (‖·‖∞)':>18}  ")
        for j in range(len(sol)):
            self.texto.insert(tk.END, f"{'x' + str(j + 1):>14}  ")
        self.texto.insert(tk.END, "\n")

        for it, xv, er in hist:
            self.texto.insert(tk.END, f"{it:6d}  {fmt.format(er):>18}  ")
            for val in xv:
                self.texto.insert(tk.END, f"{fmt.format(val):>14}  ")
            self.texto.insert(tk.END, "\n")

    def _limpiar_salida(self) -> None:
        self.texto.delete("1.0", tk.END)


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
