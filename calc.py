# Импорт библиотек
import tkinter as tk              # GUI для калькулятора
import math                       # Математические функции и константы
import re                         # Регулярные выражения для анализа строк
from collections import deque     # Очередь для обратной польской нотации (RPN)
import sys                        # Для вывода сообщений и отладки
import random                     # Генерация случайных чисел
import inspect                    # Для анализа функций (сколько аргументов требуется)

# Цветовая схема интерфейса (основные цвета и цвета кнопок)
BG_MAIN = "#23252b"
BG_ENTRY = "#181921"
BG_EXPR = "#29494d"
FG_ENTRY = "#ffdf80"
FG_EXPR = "#7cfced"
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

# Логика научного калькулятора. Выделена в отдельный класс.
class CalculatorLogic:
    def __init__(self):
        self.ans = 0.0
        self._setup_environment()  # Подготовим функции, операторы и константы

    def _setup_environment(self):
        # Все поддерживаемые функции калькулятора
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
            'gamma': math.gamma,
            'sign': lambda x: 1 if x > 0 else -1 if x < 0 else 0,
            'min': min, 'max': max,
            'avg': lambda *args: sum(args) / len(args) if args else 0,
            'deg': math.degrees, 'rad': math.radians,
            'random': random.random,
        }
        # Известные константы
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'phi': (1 + math.sqrt(5)) / 2,
            'inf': float('inf')
        }
        # Поддерживаемые операторы
        self.operators = {
            '+': {'prec': 1, 'assoc': 'L', 'func': lambda a, b: a + b},
            '-': {'prec': 1, 'assoc': 'L', 'func': lambda a, b: a - b},
            '*': {'prec': 2, 'assoc': 'L', 'func': lambda a, b: a * b},
            '/': {'prec': 2, 'assoc': 'L', 'func': lambda a, b: a / b},
            '^': {'prec': 3, 'assoc': 'R', 'func': lambda a, b: a ** b},
        }

    # Главная функция вычисления строки-выражения
    def evaluate(self, expression_str: str):
        try:
            clean_expr = self._prepare_expression(expression_str)  # Подготовка (замены, подстановка ans и т.д.)
            tokens = self._tokenize(clean_expr)  # Разбиваем на токены (числа, операторы, функции)
            rpn_queue = self._shunting_yard(tokens)  # Переводим в обратную польскую нотацию (RPN)
            result = self._evaluate_rpn(rpn_queue)  # Вычисляем по RPN
            # Ограничиваем значение по верхней/нижней границе и округляем
            if abs(result) > 1e-12 and abs(result) < 1e12:
                result = round(result, 12)
            self.ans = result  # Сохраняем последний результат в ans
            return result
        except ZeroDivisionError:
            raise ValueError("Деление на ноль")
        except (SyntaxError, IndexError, TypeError, ValueError) as e:
            if "Неверный аргумент" in str(e):
                raise ValueError("Неверный аргумент") from e
            raise SyntaxError("Ошибка синтаксиса") from e
        except Exception as e:
            raise ValueError("Неизвестная ошибка") from e

    # Подстановка ans, замены символов и явное умножение
    def _prepare_expression(self, expr: str) -> str:
        expr = expr.lower().strip()
        # Замена символов на имена функций и констант
        replacements = {
            'ans': str(self.ans), 'π': 'pi', 'ϕ': 'phi', '√': 'sqrt', '∛': 'cbrt',
            '÷': '/', '×': '*', '∞': 'inf',
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        # n! -> fact(n)
        expr = re.sub(r'(\d+\.?\d*|\bpi\b|\be\b|\bphi\b|\([^\(\)]+\))!', r'fact(\1)', expr)
        # Вставляем '*' между скобками и числами, функциями
        expr = re.sub(r'(\d|\))(\()', r'\1*\2', expr)    # 2(3) -> 2*(3)
        expr = re.sub(r'(\))(\d)', r'\1*\2', expr)       # )(3 -> )*3
        expr = re.sub(r'(\d)([a-z(])', r'\1*\2', expr)   # 3sin( -> 3*sin(
        expr = re.sub(r'(!|\))([a-z(])', r'\1*\2', expr) # )sin( -> )*sin(
        return expr

    # Разбиение на токены (операторы, функции, числа и т.д.)
    def _tokenize(self, expr: str) -> list:
        token_regex = re.compile(
            r'([0-9]+\.?[0-9]*|[a-z_][a-z0-9_]*|[+\-*/^()]|,)'
        )
        tokens = token_regex.findall(expr)
        output = []
        # Обработка унарного минуса — превращается в умножение на -1
        for i, token in enumerate(tokens):
            if token == '-' and (i == 0 or tokens[i-1] in self.operators or tokens[i-1] == '('):
                output.extend(['-1', '*'])
            else:
                output.append(token)
        return output

    # Преобразование в RPN по алгоритму Дейкстры (shunting-yard)
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
                # Выгружаем операторы с большим/равным приоритетом
                while (operator_stack and operator_stack[-1] in self.operators and
                       ((self.operators[token]['assoc'] == 'L' and self.operators[token]['prec'] <= self.operators[operator_stack[-1]]['prec']) or
                        (self.operators[token]['assoc'] == 'R' and self.operators[token]['prec'] < self.operators[operator_stack[-1]]['prec']))):
                    output_queue.append(operator_stack.pop())
                operator_stack.append(token)
            elif token == '(':
                operator_stack.append(token)
            elif token == ')':
                # Выгружаем все операторы до открывающей скобки
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

    # Логика вычисления выражения, записанного в RPN
    def _evaluate_rpn(self, rpn_queue: deque):
        stack = []
        while rpn_queue:
            token = rpn_queue.popleft()
            # Числа (в том числе отрицательные)
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
                    # Определяем, сколько аргументов требуется функции
                    sig = inspect.signature(func)
                    arg_count = 0
                    for p in sig.parameters.values():
                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default == inspect._empty:
                            arg_count += 1
                    # Для функций с *args — забираем всё из стека!
                    if sig.parameters and any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
                        arg_count = len(stack)
                except Exception:
                    arg_count = 1 # если не смогли определить —
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

# Класс графического интерфейса
class SciCalcGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.logic = CalculatorLogic()
        self.expression = ""  # строка текущего ввода
        self.title("Refactored ScienceCalc")
        self.configure(bg=BG_MAIN)
        self.resizable(False, False)
        self.display_var = tk.StringVar()  # строка результата
        self.expr_var = tk.StringVar()     # строка выражения
        self.last_calc = False             # был ли только что произведён расчет
        self.create_widgets()              # построение интерфейса

    def create_widgets(self):
        # Основной контейнер
        frame = tk.Frame(self, bg=BG_MAIN)
        frame.pack(expand=True, fill="both", padx=5, pady=5)

        # Поле истории/выражения
        expr_disp = tk.Entry(frame, textvariable=self.expr_var, font=EXPR_FONT,
                             bg=BG_EXPR, fg=FG_EXPR, insertbackground=FG_EXPR,
                             relief="flat", bd=1, justify="right", state='readonly')
        expr_disp.grid(row=0, column=0, columnspan=8, sticky="nsew", padx=5, pady=(12, 1), ipady=2)

        # Основное поле вывода результата
        entry = tk.Entry(frame, textvariable=self.display_var, font=ENTRY_FONT,
                         bg=BG_ENTRY, fg=FG_ENTRY, insertbackground=FG_ENTRY,
                         relief="flat", bd=3, justify="right", state='readonly')
        entry.grid(row=1, column=0, columnspan=8, sticky="nsew", padx=5, pady=(1, 10), ipady=10)

        # Разметка кнопок: основной список всех кнопок и их типов (для цветовой схемы)
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

        # Построение кнопок по сетке
        grid_row = 2
        for row_items in button_layout:
            grid_col = 0
            # Определяем: последние строки — по 4 кнопки, остальные по 8 (ширина строчки)
            max_cols = 4 if grid_row >= 8 else 8
            for config in row_items:
                if grid_col >= max_cols: continue
                text, block = config[0], config[1]
                self.create_styled_button(frame, text, block, grid_row, grid_col)
                grid_col += 1
            grid_row += 1

    # Вспомогательная функция создания красивых кнопок
    def create_styled_button(self, parent, text, block, row, col):
        bg_map = {'num': BTN_NUM_BG, 'op': BTN_OP_BG, 'fn': BTN_FN_BG, 'ctrl': BTN_CTRL_BG, 'eq': BTN_EQ_BG}
        fg_map = {'num': BTN_NUM_FG, 'op': BTN_OP_FG, 'fn': BTN_FN_FG, 'ctrl': BTN_CTRL_FG, 'eq': BTN_EQ_FG}
        # Определяем действие для каждой кнопки
        if text == 'C': cmd = self.clear
        elif text == '⌫': cmd = self.backspace
        elif text == '±': cmd = self.toggle_sign
        elif text == 'Rnd': cmd = lambda: self.add_text(f"{random.random():.5f}")
        elif text == '=': cmd = self.evaluate_expression
        elif text == '√(': cmd = lambda: self.add_text('√(')
        elif text == '∛(': cmd = lambda: self.add_text('∛(')
        else: cmd = lambda t=text: self.add_text(t)

        # Создаём прямоугольную область под кнопку
        canvas = tk.Canvas(parent, width=74, height=54, highlightthickness=0, bd=0, bg=parent['bg'])
        canvas.grid(row=row, column=col, padx=6, pady=8, sticky="ew")

        # Цвета отрисовки кнопки
        border = BORDER_COLORS.get(block, '#62636b')
        color_top = TOP_COLORS.get(block, '#62636b')

        # Рисуем "градиент" кнопки — верх и низ разного цвета
        canvas.create_rectangle(2, 2, 72, 52, outline=border, width=3, fill=color_top)
        canvas.create_rectangle(6, 28, 68, 52, outline='', fill=bg_map[block])
        # Надпись на кнопке
        canvas.create_text(37, 32, text=text, font=FONT, fill=fg_map[block])
        # Привязываем обработчик клика
        canvas.bind("<Button-1>", lambda event: cmd())

    # Добавить текст в текущее выражение (при нажатии кнопки)
    def add_text(self, value):
        # Если выводилась ошибка — обнуляем всё
        if str(self.display_var.get()).startswith("Ошибка"):
            self.expression = ""
            self.expr_var.set("")
        # Если только что был расчет — начинаем новое выражение
        if self.last_calc:
            self.expression = ""
            self.display_var.set("")
            self.expr_var.set("")
            self.last_calc = False
        self.expression += value
        self.display_var.set(self.expression)

    # Очистить выражение
    def clear(self):
        self.expression = ""
        self.expr_var.set("")
        self.display_var.set("")
        self.last_calc = False

    # Удалить последний символ (по Backspace)
    def backspace(self):
        if str(self.display_var.get()).startswith("Ошибка"):
            self.clear()
        else:
            self.expression = self.expression[:-1]
            self.display_var.set(self.expression)

    # Переключение на отрицательное/положительное число
    def toggle_sign(self):
        if not self.expression: return
        if self.expression.startswith('-(') and self.expression.endswith(')'):
            self.expression = self.expression[2:-1]
        else:
            self.expression = f"-({self.expression})"
        self.display_var.set(self.expression)

    # Вычислить выражение и показать результат
    def evaluate_expression(self):
        if not self.expression: return
        try:
            orig = self.expression
            result = self.logic.evaluate(self.expression)
            # Корректно выводим целое без .0, иначе — просто округлённое число
            result_str = str(int(result)) if isinstance(result, float) and result.is_integer() else str(result)
            self.expr_var.set(orig)
            self.display_var.set(result_str)
            self.expression = result_str  # Можно дальше считать с этим значением
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

# Запуск калькулятора
if __name__ == "__main__":
    print("[DEBUG] Запуск приложения SciCalc...", file=sys.stderr)
    app = SciCalcGUI()
    app.mainloop()
    print("[DEBUG] Приложение закрыто.", file=sys.stderr)
