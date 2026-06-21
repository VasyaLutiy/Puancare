#!/usr/bin/env python3
"""
Step 3 — Проба асимметрии = KILL-TEST (ResearchPuancare.md §8.3, §7)

Независимый сигнал (НЕ из графа WordNet): GPT-2 logprob обоих направлений.
    gpt2_asym(A,B) = logP("<a> A is <a> B.") - logP("<a> B is <a> A.")

Наш предиктор (заморожен в Шаге 2): направленная cost-асимметрия
    our(A,B) = cost(B→A) - cost(A→B) = h_ic(A) - h_ic(B)
  (для предложения "A is a B": >0 => A конкретнее => "A is a B" естественнее).

Бейзлайны (§7, обязаны быть побиты):
    freq_diff  = log f(A) - log f(B)     — ЕДИНСТВЕННЫЙ направленный конкурент (риск!)
    path_len   = длина пути A-B          — СИММЕТРИЧЕН => на направленный сигнал ≈ 0
    emb_cos    = косинус эмбеддингов A,B  — СИММЕТРИЧЕН => на направленный сигнал ≈ 0

КРИТЕРИЙ УБИЙСТВА:
  1. sign-match our vs gpt2  >> sign-match freq vs gpt2  (направление)
  2. partial Spearman(our, gpt2 | freq) сохраняется > 0  (сигнал СВЕРХ частоты)
  Если (2) схлопывается ~0 — мерили частоту, хороним.
"""

import sys
import math
import random
from collections import deque

import torch
from scipy import stats
from nltk.corpus import wordnet as wn

from step0_tda_gate import ensure_wordnet, collect_subtree
from step2_cost import build_h_ic, ancestors, cost as cost_fn

SEED = 0


# ---------- датасет пар (held-out, авто из WordNet) ----------

def good_word(synset):
    """Однословная, буквенная, не слишком короткая лемма — чтобы GPT-2 не терялся."""
    name = synset.lemma_names()[0]
    if "_" in name or "-" in name or not name.isalpha() or len(name) < 3:
        return None
    return name.lower()


def descendants_map(nodes, parents):
    children = {n: [] for n in nodes}
    for c, ps in parents.items():
        for p in ps:
            children[p].append(c)
    desc = {}
    sys.setrecursionlimit(1 << 20)

    def dfs(u):
        if u in desc:
            return desc[u]
        s = {u}
        for v in children[u]:
            s |= dfs(v)
        desc[u] = s
        return s

    for n in nodes:
        dfs(n)
    return desc, children


def build_pairs(nodes, parents, n_isa=120, n_cousin=120):
    rng = random.Random(SEED)
    desc, children = descendants_map(nodes, parents)
    words = {n: good_word(n) for n in nodes}
    # ДЕТЕРМИНИЗМ: nodes — set, порядок итерации плавает между процессами.
    # Сортируем по имени синсета => RNG воспроизводим, пары стабильны (пред-регистрация).
    def srt(seq):
        return sorted(seq, key=lambda s: s.name())
    usable = srt(n for n in nodes if words[n])

    isa = []
    tries = 0
    while len(isa) < n_isa and tries < n_isa * 60:
        tries += 1
        s = rng.choice(usable)
        anc = srt(a for a in ancestors(s, parents) if a != s and words[a])
        if not anc:
            continue
        g = rng.choice(anc)
        if words[s] != words[g]:
            isa.append((s, g))  # s конкретнее, g общее ; "s is a g" истинно

    cousins = []
    tries = 0
    usable_set = set(usable)
    while len(cousins) < n_cousin and tries < n_cousin * 80:
        tries += 1
        x = rng.choice(usable)
        anc_x = srt(a for a in ancestors(x, parents) if a != x)
        if not anc_x:
            continue
        c = rng.choice(anc_x)                       # общий предок
        cand = srt(d for d in desc[c] if d in usable_set
                   and d not in ancestors(x, parents)
                   and x not in ancestors(d, parents)
                   and words[d] != words[x])
        if not cand:
            continue
        y = rng.choice(cand)                        # кузен x: общий предок c, не is-a
        cousins.append((x, y))

    return isa, cousins, words


# ---------- GPT-2 scoring ----------

def art(w):
    return "an" if w[0] in "aeiou" else "a"


def sentence(a, b):
    return f"{art(a).capitalize()} {a} is {art(b)} {b}."


class GPT2Scorer:
    def __init__(self, name="gpt2"):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        torch.manual_seed(SEED)
        self.tok = GPT2TokenizerFast.from_pretrained(name)
        self.model = GPT2LMHeadModel.from_pretrained(name).eval()
        self.emb = self.model.transformer.wte.weight.detach()  # static embeddings (baseline)

    @torch.no_grad()
    def logprob(self, text):
        ids = self.tok(text, return_tensors="pt").input_ids
        out = self.model(ids)
        logits = out.logits[:, :-1, :]
        tgt = ids[:, 1:]
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return tok_lp.sum().item()

    def word_vec(self, word):
        ids = self.tok(" " + word).input_ids
        return self.emb[ids].mean(0)


def cos(u, v):
    return float(torch.dot(u, v) / (u.norm() * v.norm() + 1e-9))


# ---------- partial Spearman ----------

def spearman(x, y):
    return stats.spearmanr(x, y).correlation


def partial_spearman(x, y, z):
    """Корреляция x,y при контроле z (через ранги: формула частной корреляции)."""
    rxy = spearman(x, y)
    rxz = spearman(x, z)
    ryz = spearman(y, z)
    denom = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / denom


def lemma_freq(synset):
    return sum(l.count() for l in synset.lemmas()) + 1  # SemCor counts + сглаживание


def main():
    root_name = sys.argv[1] if len(sys.argv) > 1 else "animal.n.01"
    ensure_wordnet()
    root, nodes, edges_und, parents = collect_subtree(root_name)
    h_ic, ndesc = build_h_ic(root, nodes, parents)

    # глубина для path baseline
    depth = {root: 0}
    children = {n: [] for n in nodes}
    for c, ps in parents.items():
        for p in ps:
            children[p].append(c)
    q = deque([root])
    while q:
        u = q.popleft()
        for v in children[u]:
            if v not in depth:
                depth[v] = depth[u] + 1
                q.append(v)

    isa, cousins, words = build_pairs(nodes, parents)
    pairs = [("isa", a, b) for a, b in isa] + [("cousin", a, b) for a, b in cousins]
    print(f"[data] is-a пар={len(isa)}  cousin пар={len(cousins)}  всего={len(pairs)}")

    print("[gpt2] loading ...", file=sys.stderr)
    scorer = GPT2Scorer("gpt2")

    rows = []
    for kind, a, b in pairs:
        wa, wb = words[a], words[b]
        lp_ab = scorer.logprob(sentence(wa, wb))   # "A is a B"
        lp_ba = scorer.logprob(sentence(wb, wa))   # "B is a A"
        gpt2_asym = lp_ab - lp_ba

        our = h_ic[a] - h_ic[b]                     # cost-асимметрия (Шаг 2)
        freq_diff = math.log(lemma_freq(a)) - math.log(lemma_freq(b))
        # path: симметричная длина пути (через LCA-расстояние по depth)
        cab, lab = cost_fn(a, b, parents, h_ic)
        path_len = (depth[a] - depth[lab]) + (depth[b] - depth[lab]) if lab else 0
        emb_cos = cos(scorer.word_vec(wa), scorer.word_vec(wb))  # симметричен

        rows.append(dict(kind=kind, a=wa, b=wb, gpt2=gpt2_asym, our=our,
                         freq=freq_diff, path=path_len, cos=emb_cos))

    G = [r["gpt2"] for r in rows]
    O = [r["our"] for r in rows]
    F = [r["freq"] for r in rows]
    P = [r["path"] for r in rows]
    C = [r["cos"] for r in rows]

    def signmatch(pred):
        m = [1 for p, g in zip(pred, G) if (p > 0) == (g > 0) and g != 0]
        n = sum(1 for g in G if g != 0)
        return len(m) / n if n else float("nan")

    print("=" * 72)
    print(f"  KILL-TEST  —  subtree {root_name}  (N={len(rows)} пар, сигнал=GPT-2 logprob-асим)")
    print("=" * 72)
    print("  Spearman корреляция предиктора с GPT-2 logprob-асимметрией:")
    print(f"    НАШ  h_ic-diff (направленный) : {spearman(O, G):+.3f}")
    print(f"    freq_diff      (направленный) : {spearman(F, G):+.3f}   <- конкурент")
    print(f"    path_len       (СИММЕТРИЧЕН)  : {spearman(P, G):+.3f}   <- structurally ~0")
    print(f"    emb_cos        (СИММЕТРИЧЕН)  : {spearman(C, G):+.3f}   <- structurally ~0")
    print("-" * 72)
    print("  Точность предсказания ЗНАКА асимметрии (направление 'это'):")
    print(f"    НАШ  h_ic-diff : {signmatch(O):.1%}")
    print(f"    freq_diff      : {signmatch(F):.1%}")
    print(f"    path / cos     : неопределён (симметричны, знака нет)")
    print("-" * 72)
    ps = partial_spearman(O, G, F)
    print("  *** КРИТЕРИЙ УБИЙСТВА ***")
    print(f"    partial Spearman(НАШ, GPT-2 | freq) = {ps:+.3f}")
    if abs(ps) < 0.1:
        print("    -> СХЛОПНУЛОСЬ. Мерили частоту, не структуру. ХОРОНИМ честно. ⚰️")
    else:
        print("    -> ВЫЖИЛО. Направленная структура несёт сигнал СВЕРХ частоты. ✅")
    print("=" * 72)

    # на is-a и cousin отдельно
    for k in ("isa", "cousin"):
        idx = [i for i, r in enumerate(rows) if r["kind"] == k]
        if len(idx) > 5:
            Ok = [O[i] for i in idx]; Gk = [G[i] for i in idx]
            print(f"  [{k:6s}] Spearman(НАШ, GPT-2) = {spearman(Ok, Gk):+.3f}  (n={len(idx)})")
    print("=" * 72)


if __name__ == "__main__":
    main()
