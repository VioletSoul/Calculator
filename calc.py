import tkinter as tk
import math
import re
import sys
import traceback
import random

BG_MAIN = "#23252b"
BG_ENTRY = "#181921"
FG_ENTRY = "#ffdf80"
BTN_NUM_BG = "#2c3040"
BTN_OP_BG = "#3c3541"
BTN_FN_BG = "#234650"
BTN_CTRL_BG = "#51545f"
BTN_NUM_FG = "#ece7ff"
BTN_OP_FG = "#ff8323"
BTN_FN_FG = "#5ed1a7"
BTN_CTRL_FG = "#dadada"
BTN_EQ_BG = "#444d2a"
BTN_EQ_FG = "#f4ffae"
FONT = ("Segoe UI", 17, "bold")

BORDER_COLORS = {
    'num': '#444659',
    'op': '#806d53',
    'fn': '#357c6f',
    'ctrl': '#828393',
    'eq': '#7e9e51'
}

def dbg(*args):
    print("[DEBUG]", *args, file=sys.stderr)

def factorial_any(n):
    try:
        if isinstance(n, int) or (isinstance(n, float) and n.is_integer()):
            if n < 0:
                return math.gamma(n+1)
            else:
                return math.factorial(int(n))
        else:
            return math.gamma(n+1)
    except Exception:
        return float('nan')

def sign(x): return 1 if x > 0 else -1 if x < 0 else 0
def avg(*args): return sum(args)/len(args) if args else 0

def replace_factorials(expr):
    def repl(m):
        num = m.group(1)
        if num.startswith("+"): num = num[1:]
        return f"factorial_any({num})"
    return re.sub(r'([+-]?\d*\.?\d+)!', repl, expr)

def insert_mult(expr):
    expr = re.sub(r'(\d|\)|!)(?=[a-zA-Z\(])', r'\1*', expr)
    expr = re.sub(r'(\))(\()', r'\1*\2', expr)
    return expr

def smart_replace_functions(expr):
    expr = re.sub(r'√\(([^\)]*)\)', r'math.sqrt(\1)', expr)
    expr = re.sub(r'∛\(([^\)]*)\)', r'math.pow(\1,1/3)', expr)
    pairs = {
        'sin': 'math.sin(math.radians({}))', 'cos': 'math.cos(math.radians({}))',
        'tan': 'math.tan(math.radians({}))', 'cot': '1/math.tan(math.radians({}))',
        'sec': '1/math.cos(math.radians({}))', 'csc': '1/math.sin(math.radians({}))',
        'arcsin': 'math.degrees(math.asin({}))', 'arccos': 'math.degrees(math.acos({}))',
        'arctan': 'math.degrees(math.atan({}))',
        'sinh': 'math.sinh({})', 'cosh': 'math.cosh({})', 'tanh': 'math.tanh({})',
        'arcsinh': 'math.asinh({})', 'arccosh': 'math.acosh({})', 'arctanh': 'math.atanh({})',
        'log': 'math.log({})', 'log₁₀': 'math.log10({})', 'log₂': 'math.log2({})',
        'ln1p': 'math.log1p({})',
        'min': 'min({})', 'max': 'max({})', 'avg': 'avg({})', 'abs': 'abs({})',
        'sign': 'sign({})', 'deg': 'math.degrees({})', 'rad': 'math.radians({})', 'γ': 'math.gamma({})',
        'exp': 'math.exp({})', 'expm1': 'math.expm1({})'
    }
    for f in sorted(pairs, key=len, reverse=True):
        expr = re.sub(rf'{f}\(([^\)]*)\)', pairs[f].replace('{}',r'\1'), expr)
    return expr

def create_styled_button(parent, text, cmd, bg, fg, row, col, block='num'):
    border = BORDER_COLORS.get(block, '#62636b')
    color_top = "#35383f" if block == "num" else "#345058" if block == "fn" else "#46404b" if block == "op" else "#5f626b"
    if block == "eq": color_top = "#506040"
    btn = tk.Canvas(parent, width=74, height=54, highlightthickness=0, bd=0, bg=parent['bg'])
    btn.create_rectangle(2, 2, 72, 52, outline=border, width=3, fill=color_top)
    btn.create_rectangle(6, 28, 68, 52, outline='', fill=bg)
    btn.create_text(37, 32, text=text, font=FONT, fill=fg)
    btn.grid(row=row, column=col, padx=6, pady=8, sticky="nsew")
    btn.bind("<Button-1>", lambda event: cmd())
    return btn

class SciCalc(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ScienceCalc")
        self.configure(bg=BG_MAIN)
        self.expression = ""
        self.result = ""
        self.last_arg = ""
        self.last_op = ""
        self.display_var = tk.StringVar()
        self.create_widgets()
        self.resizable(False, False)

    def clear(self):
        self.expression = ""
        self.result = ""
        self.last_arg = ""
        self.last_op = ""
        self.display_var.set("")
        dbg("Очистка")

    def backspace(self):
        if self.expression:
            self.expression = self.expression[:-1]
            self.display_var.set(self.expression)
            dbg("Backspace, expression:", self.expression)

    def add(self, val):
        if not self.expression or self.display_var.get() == "Ошибка":
            if val in ['+', '-', '×', '÷', '^']:
                if self.result:
                    self.expression = self.result + val
            else:
                self.expression = val
        else:
            if val in ['+', '-', '×', '÷', '^']:
                if self.expression[-1] in ['+', '-', '×', '÷', '^']:
                    self.expression = self.expression[:-1] + val
                else:
                    self.expression += val
            else:
                self.expression += val
        self.display_var.set(self.expression)
        dbg("Ввод:", val, "Выражение:", self.expression)

    def eval_expr(self):
        expr = self.expression
        if not expr and not self.result:
            self.display_var.set("Введите выражение")
            return
        try:
            last_op = None
            for op in ['+', '-', '×', '÷', '^']:
                if expr.endswith(op):
                    last_op = op
            if last_op:
                if self.result:
                    expr += self.result
                else:
                    parts = re.split('[' + re.escape('+-×÷^') + ']', expr)
                    if len(parts) > 1 and parts[-1]:
                        self.last_arg = parts[-1]
                    else:
                        self.last_arg = '0' if last_op in ['÷', '×'] else '0'
                    expr += self.last_arg
            dbg("Вычисляем:", expr)
            expr = expr.replace("π", "math.pi").replace("e", "math.e") \
                .replace("ϕ", "(1+math.sqrt(5))/2") \
                .replace("γ(", "math.gamma(") \
                .replace("c", "299792458") \
                .replace("∞", "float('inf')") \
                .replace("^", "**").replace("÷", "/").replace("×", "*").replace(",", ".")
            expr = smart_replace_functions(expr)
            expr = replace_factorials(expr)
            expr = insert_mult(expr)
            if expr.count('(') != expr.count(')'):
                raise ValueError("Проблема со скобками")
            env = {
                'math': math, 'abs': abs, 'gamma': math.gamma, 'factorial_any': factorial_any,
                'sign': sign, 'avg': avg, 'min': min, 'max': max, 'random': random.random
            }
            result = eval(expr, env)
            dbg("Результат:", result)
            self.display_var.set(f"{expr.replace('**', '^').replace('/', '÷').replace('*', '×')}={result}")
            self.result = str(result)
            self.expression = ""
            self.last_arg = re.sub(r'.*[+×÷^×/-]', '', expr) if re.search(r'[+×÷^×/-]', expr) else str(result)
            self.last_op = last_op if last_op else ''
        except Exception as e:
            dbg("Ошибка вычисления:", e)
            traceback.print_exc(file=sys.stderr)
            self.display_var.set("Ошибка")
            self.expression = ""
            self.result = ""
            self.last_arg = ""
            self.last_op = ""

    def repeat_operation(self):
        if self.last_op and self.result:
            self.add(self.last_op)
            self.add(self.result)

    def plusminus(self):
        if not self.expression: return
        if re.search(r'[+\-×÷^]$', self.expression): return
        if not re.search(r'[0-9πeϕc∞.]$', self.expression): return
        expr = re.sub(r'([0-9πeϕc∞.]+)$', lambda m: str(-float(m.group(1))), self.expression, 1)
        self.expression = expr
        self.display_var.set(self.expression)
        dbg("Плюс-минус, выражение:", self.expression)

    def create_widgets(self):
        frame = tk.Frame(self, bg=BG_MAIN)
        frame.pack(expand=True, fill="both")
        entry = tk.Entry(frame, textvariable=self.display_var, font=("Consolas", 23, "bold"),
                         bg=BG_ENTRY, fg=FG_ENTRY, insertbackground=FG_ENTRY, relief="flat", bd=3, justify="right")
        entry.grid(row=0, column=0, columnspan=8, sticky="nsew", padx=12, pady=12, ipady=8)
        btns = [
            (1, 0, "C", "ctrl"), (1, 1, "⌫", "ctrl"), (1, 2, "(", "ctrl"), (1, 3, ")", "ctrl"),
            (1, 4, "π", "fn"), (1, 5, "e", "fn"), (1, 6, "ϕ", "fn"), (1, 7, "±", "ctrl"),
            (2, 0, "sin(", "fn"), (2, 1, "cos(", "fn"), (2, 2, "tan(", "fn"), (2, 3, "cot(", "fn"),
            (2, 4, "sec(", "fn"), (2, 5, "csc(", "fn"), (2, 6, "abs(", "fn"), (2, 7, "sign(", "fn"),
            (3, 0, "arcsin(", "fn"), (3, 1, "arccos(", "fn"), (3, 2, "arctan(", "fn"), (3, 3, "arcsinh(", "fn"),
            (3, 4, "arccosh(", "fn"), (3, 5, "arctanh(", "fn"), (3, 6, "sinh(", "fn"), (3, 7, "cosh(", "fn"),
            (4, 0, "tanh(", "fn"), (4, 1, "deg(", "fn"), (4, 2, "rad(", "fn"), (4, 3, "exp(", "fn"),
            (4, 4, "10^(", "fn"), (4, 5, "2^(", "fn"), (4, 6, "expm1(", "fn"), (4, 7, "ln1p(", "fn"),
            (5, 0, "min(", "fn"), (5, 1, "max(", "fn"), (5, 2, "avg(", "fn"), (5, 3, "log(", "fn"),
            (5, 4, "log₁₀(", "fn"), (5, 5, "log₂(", "fn"), (5, 6, "√(", "fn"), (5, 7, "∛(", "fn"),
            (6, 0, "7", "num"), (6, 1, "8", "num"), (6, 2, "9", "num"), (6, 3, "÷", "op"),
            (6, 4, "^", "op"), (6, 5, "γ(", "fn"), (6, 6, "c", "fn"), (6, 7, "∞", "fn"),
            (7, 0, "4", "num"), (7, 1, "5", "num"), (7, 2, "6", "num"), (7, 3, "×", "op"),
            (7, 4, "Rnd", "ctrl"), (7, 5, "", "num"), (7, 6, "", "num"), (7, 7, "", "num"),
            (8, 0, "1", "num"), (8, 1, "2", "num"), (8, 2, "3", "num"), (8, 3, "-", "op"),
            (8, 4, "", "num"), (8, 5, "", "num"), (8, 6, "", "num"), (8, 7, "", "num"),
            (9, 0, "0", "num"), (9, 1, ".", "num"), (9, 2, "=", "eq"), (9, 3, "+", "op"),
            (9, 4, "!", "fn"), (9, 5, "", "num"), (9, 6, "", "num"), (9, 7, "", "num"),
        ]
        for i in range(8):
            frame.grid_columnconfigure(i, weight=1)
        for i in range(10):
            frame.grid_rowconfigure(i, weight=1)
        for row, col, txt, block in btns:
            if txt == "": continue
            if txt == "=":
                make = lambda t=txt: self.eval_expr()
            elif txt == "C":
                make = lambda t=txt: self.clear()
            elif txt == "⌫":
                make = lambda t=txt: self.backspace()
            elif txt == "±":
                make = lambda t=txt: self.plusminus()
            elif txt == "Rnd":
                make = lambda: self.add(str(round(random.random(), 5)))
            else:
                make = lambda t=txt: self.add(t)
            bg, fg = BTN_NUM_BG, BTN_NUM_FG
            if block == "fn": bg, fg = BTN_FN_BG, BTN_FN_FG
            if block == "op": bg, fg = BTN_OP_BG, BTN_OP_FG
            if block == "ctrl": bg, fg = BTN_CTRL_BG, BTN_CTRL_FG
            if block == "eq": bg, fg = BTN_EQ_BG, BTN_EQ_FG
            create_styled_button(frame, txt, make, bg, fg, row, col, block)

dbg("Запуск приложения")
app = SciCalc()
app.mainloop()
