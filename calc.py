import tkinter as tk
import math
import re
from collections import deque
import sys
import random
import inspect

BG_MAIN = "#23252b"
BG_ENTRY = "#181921"
BG_EXPR = "#29494d"        # Новый фон верхнего дисплея (темно-бирюзовый/лазурный)
FG_ENTRY = "#ffdf80"
FG_EXPR = "#7cfced"        # Ярко-голубой цвет текста выражения
BTN_NUM_BG = "#2c3040"
BTN_OP_BG = "#3c3541"
BTN_FN_BG = "#234650"
BTN_CTRL_BG = "#51545f"
BTN_EQ_BG = "#444d2a"
BTN_NUM_FG = "#ece7ff"
BTN_OP_FG = "#ff8323"
BTN_FN_FG = "#5ed1a7"
BTN_CTRL_FG = "#dadada"
BTN_EQ_FG = "#f4ffae"
FONT = ("Segoe UI", 16, "bold")
ENTRY_FONT = ("Consolas", 23, "bold")
EXPR_FONT = ("Consolas", 14, "bold")

BORDER_COLORS = {
    'num': '#444659', 'op': '#806d53', 'fn': '#357c6f',
    'ctrl': '#828393', 'eq': '#7e9e51'
}
TOP_COLORS = {
    'num': '#35383f', 'op': '#46404b', 'fn': '#345058',
    'ctrl': '#5f626b', 'eq': '#506040'
}

class CalculatorLogic:
    def __init__(self):
        self.ans = 0.0
        self._setup_environment()

    def _setup_environment(self):
        self.functions = {
            'sin': lambda x: math.sin(math.radians(x)),
            'cos': lambda x: math.cos(math.radians(x)),
            'tan': lambda x: math.tan(math.radians(x)),
            'cot': lambda x: 1 / math.tan(math.radians(x)),
            'sec': lambda x: 1 / math.cos(math.radians(x)),
            'csc': lambda x: 1 / math.sin(math.radians(x)),
            'arcsin': lambda x: math.degrees(math.asin(x)),
            'arccos': lambda x: math.degrees(math.acos(x)),
            'arctan': lambda x: math.degrees(math.atan(x)),
            'sinh': math.sinh, 'cosh': math.cosh, 'tanh': math.tanh,
            'arcsinh': math.asinh, 'arccosh': math.acosh, 'arctanh': math.atanh,
            'log': math.log, 'log10': math.log10, 'log2': math.log2,
            'ln': math.log, 'exp': math.exp, 'expm1': math.expm1, 'ln1p': math.log1p,
            'sqrt': math.sqrt, 'cbrt': lambda x: math.pow(x, 1/3),
            'abs': abs, 'fact': lambda n: math.gamma(n + 1),
            'gamma': math.gamma, 'sign': lambda x: 1 if x > 0 else -1 if x < 0 else 0,
            'min': min, 'max': max,
            'avg': lambda *args: sum(args) / len(args) if args else 0,
            'deg': math.degrees, 'rad': math.radians,
            'random': random.random,
        }
        self.constants = {
            'pi': math.pi, 'e': math.e, 'phi': (1 + math.sqrt(5)) / 2, 'inf': float('inf')
        }
        self.operators = {
            '+': {'prec': 1, 'assoc': 'L', 'func': lambda a, b: a + b},
            '-': {'prec': 1, 'assoc': 'L', 'func': lambda a, b: a - b},
            '*': {'prec': 2, 'assoc': 'L', 'func': lambda a, b: a * b},
            '/': {'prec': 2, 'assoc': 'L', 'func': lambda a, b: a / b},
            '^': {'prec': 3, 'assoc': 'R', 'func': lambda a, b: a ** b},
        }

    def evaluate(self, expression_str: str):
        try:
            clean_expr = self._prepare_expression(expression_str)
            tokens = self._tokenize(clean_expr)
            rpn_queue = self._shunting_yard(tokens)
            result = self._evaluate_rpn(rpn_queue)
            if abs(result) > 1e-12 and abs(result) < 1e12:
                result = round(result, 12)
            self.ans = result
            return result
        except ZeroDivisionError:
            raise ValueError("Деление на ноль")
        except (SyntaxError, IndexError, TypeError, ValueError) as e:
            if "Неверный аргумент" in str(e):
                raise ValueError("Неверный аргумент") from e
            raise SyntaxError("Ошибка синтаксиса") from e
        except Exception as e:
            raise ValueError("Неизвестная ошибка") from e

    def _prepare_expression(self, expr: str) -> str:
        expr = expr.lower().strip()
        replacements = {
            'ans': str(self.ans), 'π': 'pi', 'ϕ': 'phi', '√': 'sqrt', '∛': 'cbrt',
            '÷': '/', '×': '*', '∞': 'inf',
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        expr = re.sub(r'(\d+\.?\d*)!', r'fact(\1)', expr)
        expr = re.sub(r'(\d|\))(\()', r'\1*\2', expr)
        expr = re.sub(r'(\))(\d)', r'\1*\2', expr)
        expr = re.sub(r'(\d)([a-z(])', r'\1*\2', expr)
        expr = re.sub(r'(!|\))([a-z(])', r'\1*\2', expr)
        return expr

    def _tokenize(self, expr: str) -> list:
        token_regex = re.compile(
            r'([0-9]+\.?[0-9]*|[a-z_][a-z0-9_]*|[+\-*/^()]|,)'
        )
        tokens = token_regex.findall(expr)
        output = []
        for i, token in enumerate(tokens):
            if token == '-' and (i == 0 or tokens[i-1] in self.operators or tokens[i-1] == '('):
                output.extend(['-1', '*'])
            else:
                output.append(token)
        return output

    def _shunting_yard(self, tokens: list) -> deque:
        output_queue = deque()
        operator_stack = []
        for token in tokens:
            if token.replace('.', '', 1).isdigit() or token in self.constants:
                output_queue.append(token)
            elif token in self.functions:
                operator_stack.append(token)
            elif token == ',':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
            elif token in self.operators:
                while (operator_stack and operator_stack[-1] in self.operators and
                       ((self.operators[token]['assoc'] == 'L' and self.operators[token]['prec'] <= self.operators[operator_stack[-1]]['prec']) or
                        (self.operators[token]['assoc'] == 'R' and self.operators[token]['prec'] < self.operators[operator_stack[-1]]['prec']))):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                while operator_stack and operator_stack[-1] != '(':
                    output_queue.append(operator_stack.pop())
                if not operator_stack or operator_stack.pop() != '(':
                    raise SyntaxError("Несбалансированные скобки")
                if operator_stack and operator_stack[-1] in self.functions:
                    output_queue.append(operator_stack.pop())
        while operator_stack:
            op = operator_stack.pop()
            if op == '(':
                raise SyntaxError("Несбалансированные скобки")
            output_queue.append(op)
        return output_queue

    def _evaluate_rpn(self, rpn_queue: deque):
        stack = []
        while rpn_queue:
            token = rpn_queue.popleft()
            if token.replace('.', '', 1).isdigit() or (token.startswith('-') and token[1:].replace('.','',1).isdigit()):
                stack.append(float(token))
            elif token in self.constants:
                stack.append(self.constants[token])
            elif token in self.operators:
                arg2 = stack.pop()
                arg1 = stack.pop()
                stack.append(self.operators[token]['func'](arg1, arg2))
            elif token in self.functions:
                func = self.functions[token]
                try:
                    sig = inspect.signature(func)
                    arg_count = 0
                    for p in sig.parameters.values():
                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default == inspect._empty:
                            arg_count += 1
                    if sig.parameters and any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
                        arg_count = len(stack)
                except Exception:
                    arg_count = 1
                if func in (min, max):
                    arg_count = 2
                if func == self.functions['avg']:
                    arg_count = len(stack)
                if len(stack) < arg_count:
                    raise SyntaxError(f"Недостаточно аргументов для функции {token}")
                args = [stack.pop() for _ in range(arg_count)][::-1]
                stack.append(func(*args))
        if len(stack) != 1:
            raise SyntaxError("Ошибка в выражении")
        return stack[0]

class SciCalcGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.logic = CalculatorLogic()
        self.expression = ""
        self.title("Refactored ScienceCalc")
        self.configure(bg=BG_MAIN)
        self.resizable(False, False)
        self.display_var = tk.StringVar()
        self.expr_var = tk.StringVar()
        self.last_calc = False
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self, bg=BG_MAIN)
        frame.pack(expand=True, fill="both", padx=5, pady=5)

        expr_disp = tk.Entry(frame, textvariable=self.expr_var, font=EXPR_FONT,
                             bg=BG_EXPR, fg=FG_EXPR, insertbackground=FG_EXPR,
                             relief="flat", bd=1, justify="right", state='readonly')
        expr_disp.grid(row=0, column=0, columnspan=8, sticky="nsew", padx=5, pady=(12, 1), ipady=2)

        entry = tk.Entry(frame, textvariable=self.display_var, font=ENTRY_FONT,
                         bg=BG_ENTRY, fg=FG_ENTRY, insertbackground=FG_ENTRY,
                         relief="flat", bd=3, justify="right", state='readonly')
        entry.grid(row=1, column=0, columnspan=8, sticky="nsew", padx=5, pady=(1, 10), ipady=10)

        button_layout = [
            [('C', 'ctrl'), ('⌫', 'ctrl'), ('(', 'op'), (')', 'op'), ('π', 'fn'), ('e', 'fn'), ('ϕ', 'fn'), ('±', 'ctrl')],
            [('sin(', 'fn'), ('cos(', 'fn'), ('tan(', 'fn'), ('cot(', 'fn'), ('sec(', 'fn'), ('csc(', 'fn'), ('abs(', 'fn'), ('sign(', 'fn')],
            [('arcsin(', 'fn'), ('arccos(', 'fn'), ('arctan(', 'fn'), ('log(', 'fn'), ('log10(', 'fn'), ('log2(', 'fn'), ('sinh(', 'fn'), ('cosh(', 'fn')],
            [('tanh(', 'fn'), ('deg(', 'fn'), ('rad(', 'fn'), ('exp(', 'fn'), ('√(', 'fn'), ('∛(', 'fn'), ('!', 'fn'), ('gamma(', 'fn')],
            [('7', 'num'), ('8', 'num'), ('9', 'num'), ('÷', 'op'), ('^', 'op'), ('min(', 'fn'), ('max(', 'fn'), ('avg(', 'fn')],
            [('4', 'num'), ('5', 'num'), ('6', 'num'), ('×', 'op'), ('Rnd', 'ctrl'), (',', 'op'), ('inf', 'fn'), ('Ans', 'ctrl')],
            [('1', 'num'), ('2', 'num'), ('3', 'num'), ('-', 'op')],
            [('0', 'num'), ('.', 'num'), ('=', 'eq'), ('+', 'op')]
        ]

        grid_row = 2
        for row_items in button_layout:
            grid_col = 0
            max_cols = 4 if grid_row >= 8 else 8
            for config in row_items:
                if grid_col >= max_cols: continue
                text, block = config[0], config[1]
                self.create_styled_button(frame, text, block, grid_row, grid_col)
                grid_col += 1
            grid_row += 1

    def create_styled_button(self, parent, text, block, row, col):
        bg_map = {'num': BTN_NUM_BG, 'op': BTN_OP_BG, 'fn': BTN_FN_BG, 'ctrl': BTN_CTRL_BG, 'eq': BTN_EQ_BG}
        fg_map = {'num': BTN_NUM_FG, 'op': BTN_OP_FG, 'fn': BTN_FN_FG, 'ctrl': BTN_CTRL_FG, 'eq': BTN_EQ_FG}
        if text == 'C': cmd = self.clear
        elif text == '⌫': cmd = self.backspace
        elif text == '±': cmd = self.toggle_sign
        elif text == 'Rnd': cmd = lambda: self.add_text(f"{random.random():.5f}")
        elif text == '=': cmd = self.evaluate_expression
        elif text == '√(': cmd = lambda: self.add_text('√(')
        elif text == '∛(': cmd = lambda: self.add_text('∛(')
        else: cmd = lambda t=text: self.add_text(t)

        canvas = tk.Canvas(parent, width=74, height=54, highlightthickness=0, bd=0, bg=parent['bg'])
        canvas.grid(row=row, column=col, padx=6, pady=8, sticky="ew")

        border = BORDER_COLORS.get(block, '#62636b')
        color_top = TOP_COLORS.get(block, '#62636b')

        canvas.create_rectangle(2, 2, 72, 52, outline=border, width=3, fill=color_top)
        canvas.create_rectangle(6, 28, 68, 52, outline='', fill=bg_map[block])
        canvas.create_text(37, 32, text=text, font=FONT, fill=fg_map[block])
        canvas.bind("<Button-1>", lambda event: cmd())

    def add_text(self, value):
        if str(self.display_var.get()).startswith("Ошибка"):
            self.expression = ""
            self.expr_var.set("")
        if self.last_calc:
            self.expression = ""
            self.display_var.set("")
            self.expr_var.set("")
            self.last_calc = False
        self.expression += value
        self.display_var.set(self.expression)

    def clear(self):
        self.expression = ""
        self.expr_var.set("")
        self.display_var.set("")
        self.last_calc = False

    def backspace(self):
        if str(self.display_var.get()).startswith("Ошибка"):
            self.clear()
        else:
            self.expression = self.expression[:-1]
            self.display_var.set(self.expression)

    def toggle_sign(self):
        if not self.expression: return
        if self.expression.startswith('-(') and self.expression.endswith(')'):
            self.expression = self.expression[2:-1]
        else:
            self.expression = f"-({self.expression})"
        self.display_var.set(self.expression)

    def evaluate_expression(self):
        if not self.expression: return
        try:
            orig = self.expression
            result = self.logic.evaluate(self.expression)
            result_str = str(int(result)) if isinstance(result, float) and result.is_integer() else str(result)
            self.expr_var.set(orig)
            self.display_var.set(result_str)
            self.expression = result_str
            self.last_calc = True
        except (ValueError, SyntaxError) as e:
            self.expr_var.set(self.expression)
            self.display_var.set(f"Ошибка: {e}")
            self.expression = ""
            self.last_calc = False
        except Exception as e:
            self.expr_var.set(self.expression)
            self.display_var.set("Неизвестная ошибка")
            self.expression = ""
            self.last_calc = False

if __name__ == "__main__":
    print("[DEBUG] Запуск приложения SciCalc...", file=sys.stderr)
    app = SciCalcGUI()
    app.mainloop()
    print("[DEBUG] Приложение закрыто.", file=sys.stderr)
