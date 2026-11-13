"""Собирает учебный корпус ЛР2.

Скрипт создаёт три JSONL-файла:
* ``assignments.jsonl`` — задания и разрешённые шаблоны;
* ``submissions.jsonl`` — все сдачи (для потокового чтения и порождения пар);
* ``pairs.jsonl`` — 40 размеченных пар с происхождением варианта.

Корпус синтетический и учебный. Он не содержит реальных студенческих работ и
не используется для утверждений о точности на реальных данных.
"""

from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "corpus"

TEMPLATE = '''from typing import Iterable

LIMIT = 3


def solve(words: Iterable[str]) -> int:
    """Вернуть количество слов длиннее LIMIT символов."""
    return count_long(words)
'''

ASSIGNMENTS = [
    {
        "assignment_id": "hw_loops",
        "title": "Сумма чётных чисел",
        "allowed_template": "",
        "requirements": {"skills": ["for", "if", "sum"], "language": "python"},
    },
    {
        "assignment_id": "hw_strings",
        "title": "Нормализация слов",
        "allowed_template": "",
        "requirements": {"skills": ["str", "re", "string"], "language": "python"},
    },
    {
        "assignment_id": "hw_recursion",
        "title": "Рекурсивное разворачивание списка",
        "allowed_template": "",
        "requirements": {"skills": ["recursion", "list"], "language": "python"},
    },
    {
        "assignment_id": "hw_template",
        "title": "Подсчёт длинных слов (разрешённый шаблон)",
        "allowed_template": TEMPLATE,
        "requirements": {"skills": ["iterable", "template"], "language": "python"},
    },
]

SOLUTIONS: dict[str, list[str]] = {
    "hw_loops": [
        '''def sum_even(values):
    total = 0
    for value in values:
        if value % 2 == 0:
            total += value
    return total
''',
        '''def sum_even(values):
    return sum(value for value in values if value % 2 == 0)
''',
        '''def sum_even(values):
    result = 0
    index = 0
    while index < len(values):
        if values[index] % 2 == 0:
            result += values[index]
        index += 1
    return result
''',
        '''def sum_even(values):
    filtered = filter(lambda item: item % 2 == 0, values)
    return sum(filtered)
''',
        '''def sum_even(values):
    return sum([item for item in values if not item % 2])
''',
        '''def sum_even(values):
    total = 0
    for index in range(len(values)):
        item = values[index]
        if item % 2 == 0:
            total = total + item
    return total
''',
        '''def sum_even(values):
    evens = []
    for item in values:
        if item % 2 == 0:
            evens.append(item)
    return sum(evens)
''',
        '''from functools import reduce


def sum_even(values):
    return reduce(lambda acc, item: acc + item if item % 2 == 0 else acc, values, 0)
''',
        '''def sum_even(values):
    total = 0
    for item in values:
        total += item * (item % 2 == 0)
    return total
''',
        '''def sum_even(values):
    return sum(item for item in values if item & 1 == 0)
''',
    ],
    "hw_strings": [
        '''import string


def normalize_words(text):
    cleaned = text.lower()
    for mark in string.punctuation:
        cleaned = cleaned.replace(mark, " ")
    return cleaned.split()
''',
        '''import re


def normalize_words(text):
    return re.findall(r"[a-zа-яё0-9]+", text.lower())
''',
        '''def normalize_words(text):
    result = []
    word = ""
    for char in text.lower():
        if char.isalnum():
            word += char
        elif word:
            result.append(word)
            word = ""
    if word:
        result.append(word)
    return result
''',
        '''import re


def normalize_words(text):
    words = re.sub(r"[^\\w\\s]", " ", text.lower())
    return words.split()
''',
        '''def normalize_words(text):
    return "".join(c if c.isalnum() else " " for c in text.lower()).split()
''',
        '''import string


def normalize_words(text):
    table = str.maketrans({mark: " " for mark in string.punctuation})
    return text.lower().translate(table).split()
''',
        '''def normalize_words(text):
    words = []
    current = []
    for symbol in text.lower():
        if symbol.isalpha() or symbol.isdigit():
            current.append(symbol)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return words
''',
        '''import re


def normalize_words(text):
    pattern = re.compile(r"\\w+")
    return [match.group(0) for match in pattern.finditer(text.lower())]
''',
        '''def normalize_words(text):
    return [word.strip(".,!?;:-_()[]{}\\"'") for word in text.lower().split()]
''',
        '''import string


def normalize_words(text):
    lowered = text.lower()
    result = []
    for word in lowered.split():
        result.append(word.strip(string.punctuation))
    return [word for word in result if word]
''',
    ],
    "hw_recursion": [
        '''def flatten(items):
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
''',
        '''def flatten(items):
    if not items:
        return []
    head, *tail = items
    if isinstance(head, list):
        return flatten(head) + flatten(tail)
    return [head] + flatten(tail)
''',
        '''def flatten(items):
    result = []
    for element in items:
        if type(element) is list:
            result = result + flatten(element)
        else:
            result = result + [element]
    return result
''',
        '''def flatten(items):
    result = []
    stack = list(reversed(items))
    while stack:
        current = stack.pop()
        if isinstance(current, list):
            stack.extend(reversed(current))
        else:
            result.append(current)
    return result
''',
        '''def flatten(items):
    return sum((flatten(item) if isinstance(item, list) else [item] for item in items), [])
''',
        '''def flatten(items):
    output = []

    def walk(part):
        for value in part:
            if isinstance(value, list):
                walk(value)
            else:
                output.append(value)

    walk(items)
    return output
''',
        '''def flatten(items):
    result = []
    for value in items:
        if isinstance(value, (list, tuple)):
            result += flatten(list(value))
        else:
            result.append(value)
    return result
''',
        '''def flatten(items):
    if len(items) == 0:
        return []
    first = items[0]
    rest = items[1:]
    prefix = flatten(first) if isinstance(first, list) else [first]
    return prefix + flatten(rest)
''',
        '''def flatten(items):
    result = []
    for value in items:
        if isinstance(value, list):
            for nested in flatten(value):
                result.append(nested)
        else:
            result.append(value)
    return result
''',
        '''def flatten(items):
    result = []
    stack = [items]
    while stack:
        part = stack.pop()
        for value in part:
            if isinstance(value, list):
                stack.append(value)
            else:
                result.append(value)
    result.reverse()
    return result
''',
    ],
    "hw_template": [
        '''def count_long(words):
    total = 0
    for word in words:
        if len(word) > LIMIT:
            total += 1
    return total
''',
        '''def count_long(words):
    return sum(1 for word in words if len(word) > LIMIT)
''',
        '''def count_long(words):
    return len([word for word in words if len(word) > LIMIT])
''',
        '''def count_long(words):
    total = 0
    index = 0
    words = list(words)
    while index < len(words):
        if len(words[index]) > LIMIT:
            total += 1
        index += 1
    return total
''',
        '''def count_long(words):
    return sum(map(lambda word: len(word) > LIMIT, words))
''',
        '''def count_long(words):
    filtered = filter(lambda item: len(item) > LIMIT, words)
    return len(list(filtered))
''',
        '''def count_long(words):
    total = 0
    for word in words:
        total += int(len(word) > LIMIT)
    return total
''',
        '''def count_long(words):
    result = []
    for word in words:
        if len(word) > LIMIT:
            result.append(word)
    return len(result)
''',
        '''def count_long(words):
    words = list(words)
    return sum(len(word) > LIMIT for word in words)
''',
        '''def count_long(words):
    total = 0
    for word in words:
        if len(word) <= LIMIT:
            continue
        total += 1
    return total
''',
    ],
}


FAMILIES: list[dict[str, Any]] = [
    {"family_id": "F01", "assignment_id": "hw_loops", "split": "tuning", "base": 0, "alts": (1, 2), "relations": ("exact_copy", "formatting")},
    {"family_id": "F02", "assignment_id": "hw_loops", "split": "evaluation", "base": 3, "alts": (4, 5), "relations": ("renaming", "formatting")},
    {"family_id": "F03", "assignment_id": "hw_loops", "split": "tuning", "base": 6, "alts": (7, 8), "relations": ("exact_copy", "renaming")},
    {"family_id": "F04", "assignment_id": "hw_loops", "split": "evaluation", "base": 1, "alts": (2, 3), "relations": ("formatting", "independent")},
    {"family_id": "F05", "assignment_id": "hw_loops", "split": "tuning", "base": 8, "alts": (9, 0), "relations": ("independent", "independent")},
    {"family_id": "F06", "assignment_id": "hw_strings", "split": "tuning", "base": 2, "alts": (4, 6), "relations": ("exact_copy", "renaming")},
    {"family_id": "F07", "assignment_id": "hw_strings", "split": "tuning", "base": 0, "alts": (5, 7), "relations": ("formatting", "independent")},
    {"family_id": "F08", "assignment_id": "hw_strings", "split": "evaluation", "base": 3, "alts": (1, 8), "relations": ("renaming", "independent")},
    {"family_id": "F09", "assignment_id": "hw_strings", "split": "evaluation", "base": 6, "alts": (9, 2), "relations": ("exact_copy", "formatting")},
    {"family_id": "F10", "assignment_id": "hw_strings", "split": "evaluation", "base": 4, "alts": (0, 7), "relations": ("independent", "independent")},
    {"family_id": "F11", "assignment_id": "hw_recursion", "split": "tuning", "base": 0, "alts": (1, 3), "relations": ("exact_copy", "formatting")},
    {"family_id": "F12", "assignment_id": "hw_recursion", "split": "tuning", "base": 2, "alts": (4, 5), "relations": ("renaming", "independent")},
    {"family_id": "F13", "assignment_id": "hw_recursion", "split": "evaluation", "base": 6, "alts": (7, 8), "relations": ("formatting", "renaming")},
    {"family_id": "F14", "assignment_id": "hw_recursion", "split": "evaluation", "base": 1, "alts": (0, 9), "relations": ("independent", "independent")},
    {"family_id": "F15", "assignment_id": "hw_recursion", "split": "evaluation", "base": 3, "alts": (6, 2), "relations": ("exact_copy", "renaming")},
    {"family_id": "F16", "assignment_id": "hw_template", "split": "tuning", "base": 0, "alts": (1, 2), "relations": ("template_only", "template_only")},
    {"family_id": "F17", "assignment_id": "hw_template", "split": "tuning", "base": 3, "alts": (4, 5), "relations": ("template_only", "template_only")},
    {"family_id": "F18", "assignment_id": "hw_template", "split": "evaluation", "base": 6, "alts": (7, 8), "relations": ("template_only", "template_only")},
    {"family_id": "F19", "assignment_id": "hw_template", "split": "evaluation", "base": 9, "alts": (0, 3), "relations": ("template_only", "template_only")},
    {"family_id": "F20", "assignment_id": "hw_template", "split": "evaluation", "base": 2, "alts": (5, 9), "relations": ("template_only", "template_only")},
]

RELATION_NOTES = {
    "exact_copy": "Точная копия кода; происхождение примера, не вердикт.",
    "formatting": "Изменено форматирование/комментарии без изменения логики.",
    "renaming": "Переименованы функции и локальные переменные.",
    "template_only": "Обе работы используют разрешённый шаблон; собственная логика независима.",
    "independent": "Независимое решение той же задачи.",
}


def rename_identifiers(code: str) -> str:
    """AST-переименование пользовательских имён, имитирующее обход антиплагиата."""

    tree = ast.parse(code)
    protected = set(dir(builtins)) | {"self", "cls"}
    mapping: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            mapping[node.name] = f"{node.name}_renamed"
        elif isinstance(node, ast.arg):
            if node.arg not in protected:
                mapping.setdefault(node.arg, f"arg_{node.arg}")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id not in protected:
                mapping.setdefault(node.id, f"var_{node.id}")

    class Renamer(ast.NodeTransformer):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
            node.name = mapping.get(node.name, node.name)
            self.generic_visit(node)
            return node

        def visit_arg(self, node: ast.arg) -> ast.AST:
            node.arg = mapping.get(node.arg, node.arg)
            return node

        def visit_Name(self, node: ast.Name) -> ast.AST:
            node.id = mapping.get(node.id, node.id)
            return node

    tree = Renamer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def reformat(code: str) -> str:
    """Меняет форматирование и добавляет комментарии, сохраняя логику."""

    formatted = ast.unparse(ast.parse(code))
    header = (
        "# Вариант с другим форматированием.\n"
        "# Комментарии и пустые строки не должны влиять на сходство.\n\n"
    )
    return header + formatted + "\n"


def wrap_template(solution: str) -> str:
    return TEMPLATE.rstrip() + "\n\n\n" + solution.lstrip("\n")


def submission(
    *,
    student_id: str,
    submission_id: str,
    assignment_id: str,
    family_id: str,
    split: str,
    code: str,
    variant_role: str,
) -> dict[str, Any]:
    return {
        "student_id": student_id,
        "submission_id": submission_id,
        "assignment_id": assignment_id,
        "code": code,
        "text": "",
        "timestamp": "2026-10-20T12:00:00+03:00",
        "family_id": family_id,
        "split": split,
        "variant_role": variant_role,
    }


def build() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    submissions: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []

    for family_index, family in enumerate(FAMILIES, 1):
        family_id = family["family_id"]
        assignment_id = family["assignment_id"]
        split = family["split"]
        solutions = SOLUTIONS[assignment_id]
        is_template_assignment = assignment_id == "hw_template"

        base_solution = solutions[family["base"]]
        base_code = wrap_template(base_solution) if is_template_assignment else base_solution
        base_student = f"s{1000 + family_index}"
        base_submission = submission(
            student_id=base_student,
            submission_id=f"sub-{family_id}-a",
            assignment_id=assignment_id,
            family_id=family_id,
            split=split,
            code=base_code,
            variant_role="base",
        )
        submissions.append(base_submission)

        for pair_index, relation in enumerate(family["relations"], 1):
            alt_solution = solutions[family["alts"][pair_index - 1]]
            if relation == "exact_copy":
                right_code = base_code
                role = "exact_copy"
            elif relation == "formatting":
                right_code = reformat(base_code)
                role = "formatted"
            elif relation == "renaming":
                right_code = rename_identifiers(base_code)
                role = "renamed"
            elif relation == "independent":
                right_code = wrap_template(alt_solution) if is_template_assignment else alt_solution
                role = "independent"
            elif relation == "template_only":
                right_code = wrap_template(alt_solution)
                role = "template_variant"
            else:  # pragma: no cover - защита от опечатки в FAMILIES
                raise ValueError(f"Неизвестное отношение: {relation}")

            right_student = f"s{2000 + family_index * 2 + pair_index - 1}"
            right_submission = submission(
                student_id=right_student,
                submission_id=f"sub-{family_id}-{'bc'[pair_index - 1]}",
                assignment_id=assignment_id,
                family_id=family_id,
                split=split,
                code=right_code,
                variant_role=role,
            )
            submissions.append(right_submission)
            pairs.append(
                {
                    "pair_id": f"{family_id}-p{pair_index}",
                    "assignment_id": assignment_id,
                    "family_id": family_id,
                    "split": split,
                    "derivation_relation": relation,
                    "template_only": relation == "template_only",
                    "left": base_submission,
                    "right": right_submission,
                    "notes": RELATION_NOTES[relation],
                }
            )

    return ASSIGNMENTS, submissions, pairs


def main() -> None:
    assignments, submissions, pairs = build()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    def dump(path: Path, records: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    dump(DATA_DIR / "assignments.jsonl", assignments)
    dump(DATA_DIR / "submissions.jsonl", submissions)
    dump(DATA_DIR / "pairs.jsonl", pairs)

    assert len(pairs) == 40, len(pairs)
    assert len(submissions) == 60, len(submissions)
    assert sum(1 for pair in pairs if pair["template_only"]) >= 5
    assert len({pair["family_id"] for pair in pairs}) == 20
    families_by_split: dict[str, set[str]] = {"tuning": set(), "evaluation": set()}
    for pair in pairs:
        families_by_split[pair["split"]].add(pair["family_id"])
    assert families_by_split["tuning"].isdisjoint(families_by_split["evaluation"])

    print(f"assignments: {len(assignments)}")
    print(f"submissions: {len(submissions)}")
    print(f"pairs: {len(pairs)}")
    print(f"template pairs: {sum(1 for pair in pairs if pair['template_only'])}")
    print(
        "tuning/evaluation families: "
        f"{len(families_by_split['tuning'])}/{len(families_by_split['evaluation'])}"
    )


if __name__ == "__main__":
    main()
