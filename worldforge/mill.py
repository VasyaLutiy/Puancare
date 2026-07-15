"""mill — СТАНОК миров. Оператор крутит ручки, станок печатает валидные тройки.

Оператор задаёт: модель (OpenRouter), температуру, сколько троек, акцент формы.
Станок: собрать промпт (grammar + novelty_spec) → запрос модели → разобрать
донор/цель → скомпилировать (triad.build ловит брак) → повтор при браке →
записать во входной каталог для harness.

СЛЕПОТА СОХРАНЕНА: на панель станок печатает только ФОРМУ (k, темп, топология,
маршрут гейта) — НЕ слова мира. Оператор крутит ручки, глядя на распределение
форм, и не читает содержимого. Слова увидит harness-манифест не раньше
заморозки, а автор — не раньше вердикта.

  python3 worldforge/mill.py --n 8 --model x-ai/grok-3-mini --temp 1.0 \
                             --out worldforge/in
"""

import os
import sys
import argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import triad                                  # noqa: E402
import novelty_declared as nd                 # noqa: E402
from client import OpenRouter, list_models_hint  # noqa: E402

_PROMPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")

# Акценты формы — ручка «покрой пространство». Про ФОРМУ, не про дыры ядра
# (слепота): станок ротирует их по партии, чтобы не штамповать одну форму.
EMPHASES = [
    "лестница вниз, 3 ступени, медленный темп; цель — изоморф донора",
    "тумблер (обратимый), быстрый темп; цель — возмущённый родич (сдвинь порог)",
    "глубокая цепь, 4+ ступеней; цель — та же глубина, свои слова",
    "кольцо s0→s1→s2→s0; цель — изоморф",
    "сток (состояние-ловушка, из которого не уходят); цель — родич",
    "грубый зонд (2 исхода на 3 состояния — сливает ступени); цель — изоморф",
]

_OUTPUT_RULE = """
ФОРМАТ ОТВЕТА — строго два блока, без markdown-заборов, без пояснений:

===DONOR===
<полный yaml донора>
===TARGET===
<полный yaml цели>
===END===

Ничего не пиши после ===END===. Донор и цель обязаны делить по ИМЕНИ
действие-зонд и действие-вопрос из секции экзамены. Слова скрытого/исходов/
видимого — у цели свои.
"""


def _system_prompt():
    with open(os.path.join(_PROMPTS, "grammar.txt"), encoding="utf-8") as fh:
        grammar = fh.read()
    with open(os.path.join(_PROMPTS, "novelty_spec.txt"), encoding="utf-8") as fh:
        spec = fh.read()
    return grammar + "\n\n" + spec + "\n\n" + _OUTPUT_RULE


def _split(text):
    """Разобрать ответ на (donor_yaml, target_yaml). Терпим к ```-заборам и
    хвостовой болтовне модели (обрезаем по ===END===)."""
    t = text.replace("```yaml", "").replace("```", "")
    if "===DONOR===" not in t or "===TARGET===" not in t:
        raise ValueError("нет маркеров ===DONOR===/===TARGET=== в ответе")
    _, rest = t.split("===DONOR===", 1)
    donor, target = rest.split("===TARGET===", 1)
    target = target.split("===END===", 1)[0]        # отсечь хвост после цели
    return donor.strip() + "\n", target.strip() + "\n"


def _form(sig):
    topo = ("лестница" if sig["directed"] >= 0.99 else
            "тумблер" if sig["directed"] <= 0.01 else "кольцо/смеш")
    return (f"k={sig['k']:.0f} темп={sig['tempo']:.2f} {topo} "
            f"зонд-карт={sig['cards']:.0f}")


def _clean(e):
    """Санитизация ошибки для панели (ТЗ §8.1): только класс/слова ДО первой
    кавычки — str(e) компилятора цитирует слова миров, это утечка слепоты."""
    msg = str(e)
    for q in ("'", '"', "«"):
        i = msg.find(q)
        if i != -1:
            msg = msg[:i]
    return msg.strip().rstrip(":") or type(e).__name__


def _next_batch(out_dir):
    """Следующий свободный batch-NN (ТЗ §8.2): станок никогда не перезаписывает
    существующие тройки."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    while os.path.exists(os.path.join(out_dir, f"batch-{n:02d}")):
        n += 1
    return os.path.join(out_dir, f"batch-{n:02d}")


def mill(n, model, temp, out_dir, max_tries=None, env_path=None):
    cli = OpenRouter(model=model, env_path=env_path)
    lib = _library()
    w = nd.calibrate(lib.values())
    rad = nd.radius(lib, w)
    system = _system_prompt()

    batch_dir = _next_batch(out_dir)                 # своя партия, без перезаписи
    max_tries = max_tries or n * 3
    made, tries, braks = [], 0, {}
    print(f"=== СТАНОК: модель {cli.model}, темп {temp}, цель {n} троек ===")
    print(f"партия: {batch_dir}")
    print(f"панель показывает ФОРМУ, не слова (слепота). радиус={rad:.2f}\n")

    while len(made) < n and tries < max_tries:
        idx = tries
        tries += 1
        emphasis = EMPHASES[idx % len(EMPHASES)]
        user = (f"Тройка #{idx + 1}. Акцент формы: {emphasis}\n"
                f"Порождай форму, потому что она интересна как форма.")
        try:
            raw = cli.chat(system, user, temperature=temp)
            donor_txt, target_txt = _split(raw)
            tri = triad.build(donor_txt, target_txt)      # компиляция + тройка
        except (ValueError, RuntimeError) as e:
            key = _clean(e)[:40]
            braks[key] = braks.get(key, 0) + 1
            print(f"  ✗ #{idx + 1}: брак — {_clean(e)}")
            continue

        ds = nd.declared_signature(tri["donor"][1])
        ts = nd.declared_signature(tri["target"][1])
        novel, why, d, near = nd.classify_triad(ds, ts, lib, w, rad)
        name = f"m{len(made) + 1:02d}"
        dst = os.path.join(batch_dir, name)
        triad.write(tri, dst)
        made.append((name, novel, ds, ts, d))
        route = "EXAM" if novel else "REGRESS"
        print(f"  ◆ {name}: {route:8s} донор[{_form(ds)}] "
              f"цель[{_form(ts)}] d={d:.2f}")

    _report(made, tries, braks, batch_dir)
    return made


def _library():
    import repertoire
    return {name: nd.sig_of_text(text)
            for name, text in repertoire.REPERTOIRE.items()}


def _report(made, tries, braks, out_dir):
    n_exam = sum(1 for _, nv, *_ in made if nv)
    print(f"\n=== ВЫХОД СТАНКА ===")
    print(f"  годных троек: {len(made)} из {tries} попыток "
          f"(выход {100 * len(made) // max(tries, 1)}%)")
    print(f"  форм-распределение: экзамен {n_exam} | регресс {len(made) - n_exam}")
    ks = {}
    for _, _, ds, *_ in made:
        ks[ds["k"]] = ks.get(ds["k"], 0) + 1
    print(f"  доноры по k: { {int(k): v for k, v in sorted(ks.items())} }")
    if braks:
        print(f"  брак по причинам: {braks}")
    print(f"\n  тройки записаны в {out_dir}/ — НЕ читай содержимое.")
    print(f"  дальше: python3 worldforge/harness.py {out_dir} "
          f"--out worldforge/out")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="станок миров WorldForge")
    ap.add_argument("--n", type=int, default=6, help="сколько годных троек")
    ap.add_argument("--model", default=None, help="модель OpenRouter")
    ap.add_argument("--temp", type=float, default=0.9, help="температура")
    ap.add_argument("--out", default="worldforge/in", help="каталог вывода")
    ap.add_argument("--max-tries", type=int, default=None)
    ap.add_argument("--models", action="store_true", help="показать памятку моделей")
    a = ap.parse_args()
    if a.models:
        print("типовые модели (ручка --model):")
        for m in list_models_hint():
            print(f"  {m}")
        sys.exit(0)
    mill(a.n, a.model, a.temp, a.out, a.max_tries)
