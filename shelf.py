"""shelf — шкаф: постоянная версионируемая полка карточек форм.

Заменяет для ПОТОКА (станок worldforge) два неподходящих места: сундок
(chest.py — одна жизнь целиком, тащит сырьё) и in-memory репертуар
(умирает с процессом). Шкаф = JSON-файл карточек + монотонная версия, от
которой штампуются вердикты.

Пороги — не константы: веса и радиус считаются из САМОЙ полки (как
koopman.calibrate / repertoire.radius), пересчёт на каждое изменение (O(N²)
по десятку чисел — дёшево до тысяч; не оптимизировать заранее, ТЗ §5).

Узнавание (семантика v3, exam_heldout3.py — не менять, только источник
библиотеки):
  * сравнение через ЛИНЗУ ЗОНДА карточки (card.recog_sig);
  * ВВОЗ возможен только от ДОНОРА текущего экзамена (общий словарь
    токенов); у соседей-карточек словарь чужой — их карта не встанет →
    «нет карты». Донор в узнавании участвует всегда (эфемерно), на полку
    ложится только если нов (ТЗ реш. #1).
"""

import os
import json
import tempfile

import koopman
import card as card_mod
from repertoire import radius as _radius

_DEF_PATH = "shelf/shelf.json"
THIN = 5                          # тоньше — радиус ненадёжен (ТЗ D3, предупр.)


class Verdict(dict):
    """dict-вердикт: decision, near, d, d_donor, radius, version, thin."""
    @property
    def decision(self):
        return self["decision"]


class Shelf:
    def __init__(self, path, cards):
        self.path = path
        self._cards = cards
        self._version = max((c.get("added_v") or 0 for c in cards), default=0)
        self._w = None            # кэш весов/радиуса, инвалидируется на add
        self._rad = None

    # ------------------------------------------------------ загрузка/создание
    @classmethod
    def open(cls, path=_DEF_PATH):
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                cards = json.load(fh)
        else:
            cards = []
        return cls(path, cards)

    @property
    def version(self):
        return self._version

    @property
    def cards(self):
        return tuple(self._cards)            # только чтение

    # ------------------------------------------------------ веса/радиус полки
    def _recog_sigs(self, extra=()):
        return [card_mod.recog_sig(c) for c in self._cards] + \
               [card_mod.recog_sig(c) for c in extra]

    def weights(self, extra=()):
        if extra:
            return koopman.calibrate(self._recog_sigs(extra))
        if self._w is None:
            self._w = koopman.calibrate(self._recog_sigs())
        return self._w

    def radius(self, extra=()):
        if extra:
            w = self.weights(extra)
            lib = {i: s for i, s in enumerate(self._recog_sigs(extra))}
            return _radius(lib, w)
        if self._rad is None:
            lib = {i: s for i, s in enumerate(self._recog_sigs())}
            self._rad = _radius(lib, self.weights())
        return self._rad

    # ------------------------------------------------------ добавление
    def add(self, c):
        """+карточка: выдать id, added_v = новая версия, атомарно сохранить."""
        self._version += 1
        c = dict(c)
        c["id"] = f"f{len(self._cards):04d}"
        c["added_v"] = self._version
        self._cards.append(c)
        self._w = self._rad = None           # инвалидировать кэш порога
        self._save()
        return c["id"]

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".",
                                   suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self._cards, fh, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)           # атомарно

    def quarantine(self, c, reason):
        """Нестабильную карточку-кандидата — в карантин с причиной, НЕ на полку."""
        qdir = os.path.join(os.path.dirname(self.path) or ".", "quarantine")
        os.makedirs(qdir, exist_ok=True)
        stamp = c.get("origin", "unknown").replace("/", "_")
        path = os.path.join(qdir, f"{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"reason": reason, "card": c}, fh, ensure_ascii=False,
                      indent=1)
        return path

    @staticmethod
    def stable(sig1, sig2, w, rad):
        """Критерий карантина (ТЗ §7/D2): подпись стабильна по двум сидам,
        если dist(sig1, sig2) <= радиус текущей полки."""
        return koopman.dist(sig1, sig2, w) <= rad

    # ------------------------------------------------------ узнавание (v3+ворота)
    @staticmethod
    def _jitter(sig, w):
        """Дрожь подписи в единицах дистанции: взвешенная сумма измеренной
        внутримировой болтанки компонент (поле noise, из половин данных)."""
        n = sig.get("noise") or {}
        return sum(w[c] * n.get(c, 0.0) for c in koopman.COMPS)

    @staticmethod
    def pair_import(sig, donor_sig):
        """ПАРНЫЙ тест ввоза (арка ворот): «запрос неотличим от донора с
        точностью до измеренного шума пары» — покомпонентно, в собственных
        единицах компоненты, БЕЗ весов полки и БЕЗ радиуса.

        Почему не радиус: вопрос «встанет ли карта донора» — свойство ПАРЫ,
        а глобальная калибровка полки глушит ровно ту компоненту, которой
        пара различается (batch-05: полка спектрально тесна → вес spec=0 →
        кольцо-чужак невидимо, ложный ввоз b03-m02 d=0.51 при rad=1.27).
        Требуются поля noise у обеих подписей; без них — None (фолбэк на
        радиусные ворота)."""
        if not (sig.get("noise") and donor_sig.get("noise")):
            return None
        for c in koopman.COMPS:
            d = koopman.comp_dist(sig, donor_sig, c)
            lim = (sig["noise"].get(c, 0.0) + donor_sig["noise"].get(c, 0.0))
            if d > lim:
                return False
        return True

    def recognize(self, sig, donor=None):
        """Вердикт по подписи запроса (уже через свою линзу зонда). donor —
        эфемерная карточка донора текущего экзамена: входит в сравнение и
        единственный источник ввоза. Возвращает Verdict со shelf.version.

        Ворота (доводка после batch-05, ноль констант, всё из данных):
          ВВОЗ      — парный тест pair_import(q, донор): d_c <= шум_c(q) +
                      шум_c(донора) по каждой компоненте (фолбэк для legacy-
                      подписей без noise: d_donor <= радиус+дрожь).
          нет карты — форма знакома полке: d1 <= радиус + дрожь(q). Дрожь
                      запроса РАСШИРЯЕТ ворота знакомости: короткая жизнь
                      (300) даёт большую болтанку — b05-m01 ОТКАЗ@300 при
                      честной дрожи 0.37 против радиуса 0.16 был казнью за
                      бедность данных, не за чужую форму.
          ОТКАЗ     — иначе (новая форма)."""
        extra = (donor,) if donor is not None else ()
        w = self.weights(extra)
        rad = self.radius(extra)
        jit = self._jitter(sig, w)
        gate = rad + jit
        comparison = list(self._cards) + list(extra)
        if not comparison:
            return Verdict(decision="ОТКАЗ", near=None, d=float("inf"),
                           d_donor=float("inf"), radius=gate,
                           radius_shelf=rad, jitter=jit,
                           version=self._version, thin=True)
        ranked = sorted((koopman.dist(sig, card_mod.recog_sig(c), w),
                         c.get("id") or c.get("origin")) for c in comparison)
        d1, near = ranked[0]
        d_donor = (koopman.dist(sig, card_mod.recog_sig(donor), w)
                   if donor is not None else float("inf"))
        pair = (self.pair_import(sig, card_mod.recog_sig(donor))
                if donor is not None else False)
        if pair is None:                             # legacy: радиусные ворота
            pair = d_donor <= gate
        if pair:
            decision = "ВВОЗ"                       # пара неразличима → карта встанет
        elif d1 <= gate:
            decision = "нет карты"                  # форма знакома, словарь чужой
        else:
            decision = "ОТКАЗ"                       # новая форма
        return Verdict(decision=decision, near=near, d=d1, d_donor=d_donor,
                       radius=gate, radius_shelf=rad, jitter=jit,
                       version=self._version,
                       thin=len(comparison) < THIN)

    # ------------------------------------------------------ засев архетипов
    def seed_archetypes(self, budget=1500):
        """Залить 4 архетипа repertoire.py: открыть медленной дорогой, карточки
        как у всех. Одноразово, при пустой полке."""
        import runworld
        from organism2 import Organism
        from repertoire import REPERTOIRE, glue_of as glue_of_text
        runworld.Organism = Organism
        for name, text in REPERTOIRE.items():
            glue = glue_of_text(text)
            org = runworld.live(glue, False, budget)
            c = card_mod.build(org, glue, origin=f"repertoire:{name}",
                               discovery={"budget": budget, "seed": 0})
            self.add(c)
        return self._version
