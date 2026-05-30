#!/usr/bin/env python3
"""Calculadora de terminal com avaliacao segura de expressoes."""

from __future__ import annotations

import ast
import operator
import sys
from typing import Callable


OPERADORES: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

OPERADORES_UNARIOS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def calcular(expressao: str) -> float:
    """Calcula uma expressao matematica sem usar eval."""
    arvore = ast.parse(expressao.replace("^", "**"), mode="eval")
    return _avaliar(arvore.body)


def _avaliar(no: ast.AST) -> float:
    if isinstance(no, ast.Constant) and isinstance(no.value, (int, float)):
        return float(no.value)

    if isinstance(no, ast.BinOp):
        operador = OPERADORES.get(type(no.op))
        if operador is None:
            raise ValueError("Operador nao permitido.")
        esquerda = _avaliar(no.left)
        direita = _avaliar(no.right)
        if isinstance(no.op, (ast.Div, ast.FloorDiv, ast.Mod)) and direita == 0:
            raise ZeroDivisionError("Divisao por zero nao permitida.")
        return operador(esquerda, direita)

    if isinstance(no, ast.UnaryOp):
        operador = OPERADORES_UNARIOS.get(type(no.op))
        if operador is None:
            raise ValueError("Operador unario nao permitido.")
        return operador(_avaliar(no.operand))

    raise ValueError("Expressao invalida. Use apenas numeros e operadores matematicos.")


def formatar_resultado(valor: float) -> str:
    if valor.is_integer():
        return str(int(valor))
    return f"{valor:.10g}"


def modo_interativo() -> None:
    print("Calculadora Python")
    print("Use +, -, *, /, //, %, ** ou ^. Digite 'sair' para encerrar.")

    while True:
        expressao = input(">> ").strip()
        if expressao.lower() in {"sair", "exit", "q", "quit"}:
            print("Encerrando calculadora.")
            return
        if not expressao:
            continue

        try:
            print(formatar_resultado(calcular(expressao)))
        except (SyntaxError, ValueError, ZeroDivisionError) as erro:
            print(f"Erro: {erro}")


def main() -> None:
    if len(sys.argv) > 1:
        expressao = " ".join(sys.argv[1:])
        try:
            print(formatar_resultado(calcular(expressao)))
        except (SyntaxError, ValueError, ZeroDivisionError) as erro:
            print(f"Erro: {erro}")
            raise SystemExit(1) from erro
        return

    modo_interativo()


if __name__ == "__main__":
    main()
