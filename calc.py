# Импорт необходимых библиотек
import tkinter as tk  # GUI библиотека для создания окон и виджетов
import math  # Математические функции
import re  # Регулярные выражения
from collections import deque  # Двусторонняя очередь для RPN
import sys  # Для вывода отладочной информации
import random  # Генератор случайных чисел
import inspect  # Модуль для анализа сигнатур функций

# Настройки цветов для интерфейса
BG_MAIN = "#23252b"        # Цвет фона окна
BG_ENTRY = "#181921"       # Цвет поля ввода результата
BG_EXPR = "#29494d"        # Цвет истории выражения
FG_ENTRY = "#ffdf80"       # Цвет текста поля результата
FG_EXPR = "#7cfced"        # Цвет текста поля выражения
BTN_NUM_BG = "#2c3040"     # Цвет фона числовых кнопок
BTN_OP_BG = "#3c3541"      # Цвет операционных кнопок
BTN_FN_BG = "#234650"      # Цвет функциональных кнопок
BTN_CTRL_BG = "#51545f"    # Цвет управляющих кнопок
BTN_EQ_BG = "#444d2a"      # Цвет кнопки равно
BTN_NUM_FG = "#ece7ff"     # Цвет текста числовых кнопок
BTN_OP_FG = "#ff8323"      # Цвет текста операторов
BTN_FN_FG = "#5ed1a7"      # Цвет текста функций
BTN_CTRL_FG = "#dadada"    # Цвет текста управления
BTN_EQ_FG = "#f4ffae"      # Цвет текста кнопки "="

# Шрифты
FONT = ("Segoe UI", 16, "bold")
ENTRY_FONT = ("Consolas", 23, "bold")
EXPR_FONT = ("Consolas", 14, "bold")

# Цвета рамок кнопок
BORDER_COLORS = {
    'num': '#444659',
    'op': '#806d53',
    'fn': '#357c6f',
    'ctrl': '#828393',
    'eq': '#7e9e51'
}

# Цвета верхней части кнопок
TOP_COLORS = {
    'num': '#35383f',
    'op': '#46404b',
    'fn': '#345058',
    'ctrl': '#5f626b',
    'eq': '#506040'
}

# Логика калькулятора
class CalculatorLogic:
    def __init__(self):
        self.ans = 0.0  # Последний результат
        self._setup_environment()  # Настроить окружение с функциями и константами

    def _setup_environment(self):
        # Поддерживаемые математические функции
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
            'sinh': math.sinh,
            'cosh': math.cosh,
            'tanh': math.tanh,
            'arcsinh': math.asinh,
            'arccosh': math.acosh,
            'arctanh': math.atanh,
            'log': math.log,
            'log10': math.log10,
            'log2': math.log2,
            'ln': math.log,
            'exp': math.exp,
            'expm1': math.expm1,
            'ln1p': math.log1p,
            'sqrt': math.sqrt,
            'cbrt': lambda x: math.pow(x, 1/3),
            'abs': abs,
            'fact': lambda n: math.gamma(n + 1),
            'gamma': math.gamma,
            'sign': lambda x: 1 if x > 0 else -1 if x < 0 else 0,
            'min': min,
            'max': max,
            'avg': lambda *args: sum(args) / len(args) if args else 0,
            'deg': math.degrees,
            'rad': math.radians,
            'random': random.random,
        }

        # Константы
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'phi': (1 + math.sqrt(5)) / 2,
            'inf': float('inf')
        }

        # Операторы с приоритетом и ассоциативностью
        self.operators = {
            '+': {'prec': 1, 'assoc': 'L', 'func': lambda a, b: a + b},
            '-': {'prec': 1, 'assoc': 'L', 'func': lambda a, b: a - b},
            '*': {'prec': 2, 'assoc': 'L', 'func': lambda a, b: a * b},
            '/': {'prec': 2, 'assoc': 'L', 'func': lambda a, b: a / b},
            '^': {'prec': 3, 'assoc': 'R', 'func': lambda a, b: a ** b},
        }

    def evaluate(self, expression_str: str):
        try:
            # Подготовка и парсинг выражения
            clean_expr = self._prepare_expression(expression_str)
            tokens = self._tokenize(clean_expr)
            rpn_queue = self._shunting_yard(tokens)
            result = self._evaluate_rpn(rpn_queue)

            # Ограничение по диапазону для вывода
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
        # Заменяем символы вроде π, √ и т.д.
        expr = expr.lower().strip()
        replacements = {
            'ans': str(self.ans), 'π': 'pi', 'ϕ': 'phi', '√': 'sqrt', '∛': 'cbrt',
            '÷': '/', '×': '*', '∞': 'inf',
        }
        for old, new in replacements.items():
            expr = expr.replace(old, new)

        # Обработка факториала — заменяем `n!` на `fact(n)`
        expr = re.sub(r'(\d+\.?\d*|\bpi\b|\be\b|\bphi\b|\([^\(\)]+\))!', r'fact(\1)', expr)

        # Явное умножение — между числом и скобкой или функцией добавляем '*'
        expr = re.sub(r'(\d|\))(\()', r'\1*\2', expr)
        expr = re.sub(r'(\))(\d)', r'\1*\2', expr)
        expr = re.sub(r'(\d)([a-z(])', r'\1*\2', expr)
        expr = re.sub(r'(!|\))([a-z(])', r'\1*\2', expr)
        return expr

    def _tokenize(self, expr: str) -> list:
        # Разбиение строки выражения на токены (числа, функции, операторы и т.п.)
        token_regex = re.compile(r'([0-9]+\.?[0-9]*|[a-z_][a-z0-9_]*|[+\-*/^()]|,)')
        tokens = token_regex.findall(expr)
        output = []

        for i, token in enumerate(tokens):
            # Обработка унарного минуса: заменяем '-x' на '-1 * x'
            if token == '-' and (i == 0 or tokens[i-1] in self.operators or tokens[i-1] == '('):
                output.extend(['-1', '*'])
            else:
                output.append(token)
        return output

    def _shunting_yard(self, tokens: list) -> deque:
        # Преобразование выражения в обратную польскую нотацию (RPN) по алгоритму Дейкстры
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
        # Вычисление результата по RPN
        stack = []
        while rpn_queue:
            token = rpn_queue.popleft()
            if token.replace('.', '', 1).isdigit() or (token.startswith('-') and token[1:].replace('.', '', 1).isdigit()):
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
                    # Определяем необходимое количество аргументов
                    sig = inspect.signature(func)
                    arg_count = 0
                    for p in sig.parameters.values():
                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default == inspect._empty:
                            arg_count += 1
                    if sig.parameters and any(p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()):
                        arg_count = len(stack)
                except Exception:
                    arg_count = 1  # По умолчанию одна переменная

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

# GUI калькулятор на tkinter
class SciCalcGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.logic = CalculatorLogic()
        self.expression = ""
        self.title("Refactored ScienceCalc")
        self.configure(bg=BG_MAIN)
        self.resizable(False, False)
        self.display_var = tk.StringVar()  # Отображение результата
        self.expr_var = tk.StringVar()     # Отображение введённого выражения
        self.last_calc = False
        self.create_widgets()

    def create_widgets(self):
        # Создаём текстовые поля и кнопки
        ...

    def create_styled_button(self, parent, text, block, row, col):
        # Рисует стилизованные кнопки с помощью Canvas
        ...

    def add_text(self, value):
        # Добавление текста в поле выражения
        ...

    def clear(self):
        # Очистка выражения и поля результата
        ...

    def backspace(self):
        # Удаление последнего символа
        ...

    def toggle_sign(self):
        # Переключение знака числа
        ...

    def evaluate_expression(self):
        # Вычисление выражения и вывод результата
        ...

# Запуск приложения
if __name__ == "__main__":
    print("[DEBUG] Запуск приложения SciCalc...", file=sys.stderr)
    app = SciCalcGUI()
    app.mainloop()
    print("[DEBUG] Приложение закрыто.", file=sys.stderr)
