import tkinter as tk
import math
import re
import sys
import traceback

BG_MAIN      = "#23252b"
BG_ENTRY     = "#181921"
FG_ENTRY     = "#ffdf80"
BTN_NUM_BG   = "#2c3040"
BTN_OP_BG    = "#3c3541"
BTN_FN_BG    = "#234650"
BTN_CTRL_BG  = "#51545f"
BTN_NUM_FG   = "#ece7ff"
BTN_OP_FG    = "#ff8323"
BTN_FN_FG    = "#5ed1a7"
BTN_CTRL_FG  = "#dadada"
BTN_EQ_BG    = "#444d2a"
BTN_EQ_FG    = "#f4ffae"
FONT         = ("Segoe UI", 17, "bold")

BORDER_COLORS = {
    'num':  '#444659',
    'op':   '#806d53',
    'fn':   '#357c6f',
    'ctrl': '#828393',
    'eq':   '#7e9e51'
}

def dbg(*args):
    print("[DEBUG]", *args, file=sys.stderr)

def factorial_safe(expr):
    def repl(m):
        num = m.group(1)
        if '.' in num or '-' in num:
            raise ValueError("Факториал только для целых неотрицательных чисел")
        return f"math.factorial({num})"
    result = re.sub(r'(\d+)!', repl, expr)
    return result

def create_styled_button(parent, text, cmd, bg, fg, row, col, block='num'):
    border = BORDER_COLORS.get(block, '#62636b')
    color_top = "#35383f" if block == "num" else "#345058" if block == "fn" else "#46404b" if block == "op" else "#5f626b"
    if block == "eq":
        color_top = "#506040"
    btn = tk.Canvas(parent, width=74, height=54, highlightthickness=0, bd=0, bg=parent['bg'])
    btn.create_rectangle(2, 2, 72, 52, outline=border, width=3, fill=color_top)
    btn.create_rectangle(6, 28, 68, 52, outline='', fill=bg)
    btn.create_text(37, 32, text=text, font=FONT, fill=fg)
    btn.grid(row=row, column=col, padx=6, pady=8, sticky="nsew")
    btn.bind("<Button-1>", lambda event: (dbg(f"Нажата кнопка '{text}'"), cmd()))
    btn.bind("<ButtonRelease-1>", lambda event: dbg(f"Отпущена кнопка '{text}'"))
    return btn

class SciCalc(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ScienceCalc")
        self.configure(bg=BG_MAIN)
        self.geometry("470x610")
        self.resizable(False, False)
        self.expression = ""
        dbg("Приложение инициализировано")
        self.create_widgets()

    def add(self, val):
        dbg(f"Добавление в выражение: {val} (до: '{self.expression}')")
        self.expression += str(val)
        self.display_var.set(self.expression)
        dbg(f"Текущее выражение: '{self.expression}'")

    def clear(self):
        dbg("Сброс выражения")
        self.expression = ""
        self.display_var.set("")

    def backspace(self):
        dbg(f"Backspace (до: '{self.expression}')")
        self.expression = self.expression[:-1]
        self.display_var.set(self.expression)
        dbg(f"Текущее выражение: '{self.expression}'")

    def eval_expr(self):
        dbg("Вычисление выражения:", self.expression)
        try:
            expr = self.expression \
                .replace("π", "math.pi") \
                .replace("÷", "/") \
                .replace("×", "*") \
                .replace("^", "**") \
                .replace("√(", "math.sqrt(") \
                .replace(",", ".") \
                .replace("sin(", "math.sin(math.radians(") \
                .replace("cos(", "math.cos(math.radians(") \
                .replace("tan(", "math.tan(math.radians(") \
                .replace("log10(", "math.log10(") \
                .replace("log(", "math.log(") \
                .replace("ln(", "math.log(") \
                .replace("abs(", "abs(")
            dbg("Преобразованное выражение:", expr)
            expr = factorial_safe(expr)
            dbg("Выражение после проверки факториалов:", expr)
            result = eval(expr, {'math': math, 'abs': abs})
            dbg(f"Результат вычисления: {result}")
            self.display_var.set(result)
            self.expression = str(result)
        except Exception:
            tb = traceback.format_exc()
            dbg(f"Ошибка вычисления!\n{tb}")
            self.display_var.set("Ошибка")
            self.expression = ""

    def plusminus(self):
        dbg("Смена знака (до:", self.expression, ")")
        if self.expression.startswith('-'):
            self.expression = self.expression[1:]
        elif self.expression != "":
            self.expression = '-' + self.expression
        self.display_var.set(self.expression)
        dbg("Текущее выражение:", self.expression)

    def create_widgets(self):
        frame = tk.Frame(self, bg=BG_MAIN)
        frame.place(x=0, y=0, width=470, height=610)
        self.display_var = tk.StringVar()
        entry = tk.Entry(frame, textvariable=self.display_var, font=("Consolas", 23, "bold"),
                         bg=BG_ENTRY, fg=FG_ENTRY, insertbackground=FG_ENTRY,
                         relief="flat", bd=3, justify="right")
        entry.grid(row=0, column=0, columnspan=5, sticky="nsew", padx=16, pady=15, ipady=14)
        for i in range(5):
            frame.grid_columnconfigure(i, minsize=90, weight=1)
        for i in range(7):
            frame.grid_rowconfigure(i, minsize=52, weight=1)
        create_styled_button(frame, "C", self.clear, BTN_CTRL_BG, BTN_CTRL_FG, 1, 0, "ctrl")
        create_styled_button(frame, "⌫", self.backspace, BTN_CTRL_BG, BTN_CTRL_FG, 1, 1, "ctrl")
        create_styled_button(frame, "(", lambda: self.add("("), BTN_CTRL_BG, BTN_CTRL_FG, 1, 2, "ctrl")
        create_styled_button(frame, ")", lambda: self.add(")"), BTN_CTRL_BG, BTN_CTRL_FG, 1, 3, "ctrl")
        create_styled_button(frame, "π", lambda: self.add("π"), BTN_FN_BG, BTN_FN_FG, 1, 4, "fn")
        create_styled_button(frame, "sin(", lambda: self.add("sin("), BTN_FN_BG, BTN_FN_FG, 2, 0, "fn")
        create_styled_button(frame, "cos(", lambda: self.add("cos("), BTN_FN_BG, BTN_FN_FG, 2, 1, "fn")
        create_styled_button(frame, "tan(", lambda: self.add("tan("), BTN_FN_BG, BTN_FN_FG, 2, 2, "fn")
        create_styled_button(frame, "log(", lambda: self.add("log("), BTN_FN_BG, BTN_FN_FG, 2, 3, "fn")
        create_styled_button(frame, "ln(", lambda: self.add("ln("), BTN_FN_BG, BTN_FN_FG, 2, 4, "fn")
        buttons = [
            ("7", 3, 0, BTN_NUM_BG, BTN_NUM_FG, "num"), ("8", 3, 1, BTN_NUM_BG, BTN_NUM_FG, "num"),
            ("9", 3, 2, BTN_NUM_BG, BTN_NUM_FG, "num"), ("÷", 3, 3, BTN_OP_BG, BTN_OP_FG, "op"),
            ("^", 3, 4, BTN_OP_BG, BTN_OP_FG, "op"),
            ("4", 4, 0, BTN_NUM_BG, BTN_NUM_FG, "num"), ("5", 4, 1, BTN_NUM_BG, BTN_NUM_FG, "num"),
            ("6", 4, 2, BTN_NUM_BG, BTN_NUM_FG, "num"), ("×", 4, 3, BTN_OP_BG, BTN_OP_FG, "op"),
            ("√(", 4, 4, BTN_FN_BG, BTN_FN_FG, "fn"),
            ("1", 5, 0, BTN_NUM_BG, BTN_NUM_FG, "num"), ("2", 5, 1, BTN_NUM_BG, BTN_NUM_FG, "num"),
            ("3", 5, 2, BTN_NUM_BG, BTN_NUM_FG, "num"), ("-", 5, 3, BTN_OP_BG, BTN_OP_FG, "op"),
            ("!", 5, 4, BTN_FN_BG, BTN_FN_FG, "fn"),
            ("0", 6, 0, BTN_NUM_BG, BTN_NUM_FG, "num"), (".", 6, 1, BTN_NUM_BG, BTN_NUM_FG, "num"),
            ("±", 6, 2, BTN_CTRL_BG, BTN_CTRL_FG, "ctrl"), ("+", 6, 3, BTN_OP_BG, BTN_OP_FG, "op"),
            ("=", 6, 4, BTN_EQ_BG, BTN_EQ_FG, "eq"),
        ]
        for bt in buttons:
            text, row, col, bg, fg, block = bt
            if text == "=":
                cmd = self.eval_expr
            elif text == "±":
                cmd = self.plusminus
            elif text == "√(":
                cmd = lambda: self.add("√(")
            elif text == "!":
                cmd = lambda: self.add("!")
            elif text in ("÷", "×", "-", "+", "^"):
                cmd = lambda t=text: self.add(t)
            else:
                cmd = lambda t=text: self.add(t)
            create_styled_button(frame, text, cmd, bg, fg, row, col, block=block)
if __name__ == "__main__":
    dbg("Запуск приложения")
    SciCalc().mainloop()
