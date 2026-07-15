"""triad — упаковка held-out тройки донор / цель / цель-зонд.

Тройка (как в worlds/comp*, worlds/hard*):
  donor.yaml         донор — проживается ЦЕЛИКОМ (зонд + использование),
                     форма и карта использования выучиваются начисто.
  target.yaml        цель — та же (проверяемая) форма, свои слова; несёт
                     экзамен «зонд A вопрос B».
  target_probe.yaml  цель, где живёт ТОЛЬКО зонд A: карта использования не
                     выучивается → экзамен проверяет ВВОЗ карты донора, а не
                     локальное доучивание.

target_probe выводится из target механически: убрать действие-вопрос B.
Всё валидируется worldkit.Spec — брак ловится компилятором, не глазами.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from worldkit import Spec, parse_text   # noqa: E402


def _exam_actions(spec):
    """(зонд, вопрос) первого экзамена — их роли в тройке."""
    if not spec.exams:
        raise ValueError("в цели нет экзамена 'зонд A вопрос B' — тройка "
                         "не имеет что проверять")
    _, probe, ask, _ = spec.exams[0]
    return probe, ask


def strip_block(text, key, indent):
    """Вырезать блок ключа `key` с данным отступом. Блок = от строки-заголовка
    'key:' до следующей непустой строки с отступом <= indent (сосед того же
    уровня или новая секция). Хвостовые пустые строки блока тоже убираются."""
    lines = text.splitlines()
    out, i, n = [], 0, len(lines)
    head = " " * indent + f"{key}:"
    while i < n:
        stripped = lines[i].split("#")[0].rstrip()
        if stripped == head or stripped.startswith(head):
            i += 1
            while i < n:
                ln = lines[i]
                if ln.strip() == "":
                    i += 1
                    continue
                if (len(ln) - len(ln.lstrip())) <= indent:
                    break
                i += 1                      # тело вырезаемого блока
            while out and out[-1].strip() == "":
                out.pop()
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def make_probe_text(target_text):
    """Из текста цели сделать текст цели-зонда: убрать действие-вопрос (отступ 2)
    и секцию экзамена (отступ 0) — цель-зонд лишь проживается, не экзаменуется."""
    spec = Spec(parse_text(target_text))
    _, ask = _exam_actions(spec)
    probe_text = strip_block(target_text, ask, 2)
    probe_text = strip_block(probe_text, "экзамены", 0)
    Spec(parse_text(probe_text))            # должен остаться валиден
    return probe_text


def validate(text, role):
    """Скомпилировать мир; вернуть Spec или бросить ValueError с ролью."""
    try:
        return Spec(parse_text(text))
    except ValueError as e:
        raise ValueError(f"[{role}] {e}") from e


def build(donor_text, target_text):
    """Собрать и провалидировать тройку из текстов донора и цели.
    Возвращает dict {donor, target, target_probe: (text, Spec)} или бросает."""
    d_spec = validate(donor_text, "донор")
    t_spec = validate(target_text, "цель")
    probe, ask = _exam_actions(t_spec)
    for a in (probe, ask):
        if a not in d_spec.obj_actions:
            raise ValueError(f"[тройка] действие экзамена {a!r} есть в цели, "
                             f"но не в доноре — донор не умеет того, что "
                             f"проверяем (нечего ввозить)")
    probe_text = make_probe_text(target_text)
    p_spec = validate(probe_text, "цель-зонд")
    return {"donor": (donor_text, d_spec),
            "target": (target_text, t_spec),
            "target_probe": (probe_text, p_spec),
            "exam": (probe, ask)}


def write(triad, out_dir):
    """Записать тройку в каталог (donor.yaml, target.yaml, target_probe.yaml)."""
    os.makedirs(out_dir, exist_ok=True)
    for role, fname in (("donor", "donor.yaml"),
                        ("target", "target.yaml"),
                        ("target_probe", "target_probe.yaml")):
        with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as fh:
            fh.write(triad[role][0])
    return out_dir


if __name__ == "__main__":
    # само-проверка: восстановить target_probe из target каждой тройки и
    # сверить с существующим файлом (регенерация должна совпасть по действиям).
    ok = 0
    triads = ["comp1", "comp2", "comp3", "comp4", "hard1", "hard2", "hard3"]
    for name in triads:
        d = os.path.join(_ROOT, "worlds", name)
        with open(os.path.join(d, "target.yaml"), encoding="utf-8") as fh:
            tgt = fh.read()
        gen = Spec(parse_text(make_probe_text(tgt)))
        with open(os.path.join(d, "target_probe.yaml"), encoding="utf-8") as fh:
            ref = Spec(parse_text(fh.read()))
        same = set(gen.obj_actions) == set(ref.obj_actions)
        ok += same
        mark = "✓" if same else "✗"
        print(f"  {mark} {name}: регенер. зонд-действия {sorted(gen.obj_actions)}"
              f" vs эталон {sorted(ref.obj_actions)}")
    print(f"\nитог: {ok}/{len(triads)} target_probe регенерируются верно")
