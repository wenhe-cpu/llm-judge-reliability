#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
50 对扩展实验（v2，修正版）

相对第一轮的修正：
  1. 不截断：max_tokens=800，要求自然收尾；撞上限会重生成并标记
  2. 消混淆：同一温度、不同随机种子；记录"哪个回答赢"（身份）而非"A/B"（标签）
  3. 加对照：等长 / 不等长 两组 × 正序 / 换序 —— 检验"判据替换"假设

等长条件用「截短到较短者」实现（不引入外部文字）；
不等长条件用「中性段落把一侧撑到 1.5 倍」实现。

用法：
  .venv/bin/python pilot/run50.py --limit 3
  .venv/bin/python pilot/run50.py
"""

import os, sys, json, csv, re, time, argparse

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")

FILLER = [
    "需要说明的是，这个问题在不同情境下会有不同的侧重点，上述讨论只能视为一般性的梳理。",
    "此外，相关领域的研究仍在推进，部分细节结论可能随新证据的出现而调整。",
    "从实践角度看，理解这一点有助于在遇到类似问题时保持审慎，并在必要时补充更多信息。",
    "总体而言，以上内容覆盖了主要方面，但并未穷尽所有可能情形，读者可结合具体背景判断。",
]

JUDGE = """你是一名严格的评测员。下面针对同一个问题给出了两个回答。

【问题】
{q}

【回答 A】
{a}

【回答 B】
{b}

请判断哪一个回答更好。只输出两行，格式严格如下：
胜者: A
理由: <一句话>

第一行只能是「胜者: A」「胜者: B」或「胜者: 平局」。"""

GEN_SUFFIX = "\n\n请完整作答，控制在 250 字左右，务必以一个完整的句子结束，不要在句子中间断开。"


def ntok(tok, s):
    return len(tok.encode(s))


def truncate_to(tok, text, target):
    """按句子边界截到不超过 target 个 token；截不动就原样返回"""
    if ntok(tok, text) <= target:
        return text
    parts = re.split(r"(?<=[。！？])", text)
    out = ""
    for p in parts:
        if ntok(tok, out + p) > target:
            break
        out += p
    return out.strip() or text


def pad_to(tok, text, target):
    out, i = text, 0
    while ntok(tok, out) < target and i < 40:
        out = out + "\n\n" + FILLER[i % len(FILLER)]
        i += 1
    return out


def complete(tok, s, cap):
    t = s.rstrip()
    if not t:
        return False
    if ntok(tok, t) >= cap * 0.97:
        return False
    return t[-1] in "。！？…”」）).!?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "results50.csv"))
    ap.add_argument("--model", default=DEFAULT_MODEL, help="MLX 模型路径或 HuggingFace/ModelScope 仓库名")
    args = ap.parse_args()

    print(f"加载模型：{args.model}")
    model, tok = load(args.model)

    def gen(prompt, temp=0.8, seed=0, max_tokens=800):
        mx.random.seed(seed)
        p = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, tokenize=False)
        return generate(model, tok, prompt=p, max_tokens=max_tokens,
                        sampler=make_sampler(temp=temp), verbose=False).strip()

    def judge(q, first, second):
        raw = gen(JUDGE.format(q=q, a=first, b=second), temp=0.0, seed=0, max_tokens=90)
        m = re.search(r"胜者\s*[:：]\s*(A|B|平局)", raw)
        r = re.search(r"理由\s*[:：]\s*(.+)", raw)
        return (m.group(1) if m else "解析失败"), (r.group(1).strip()[:100] if r else "")

    qs = json.load(open(os.path.join(HERE, "questions50.json"), encoding="utf-8"))["questions"]
    if args.limit:
        qs = qs[:args.limit]

    fields = ["id", "type", "len_x", "len_y", "len_eq_x", "len_eq_y", "len_infl",
              "c1_win", "c2_win", "c3_win", "c4_win",
              "c1_label", "c2_label", "c3_label", "c4_label",
              "reason_eq", "reason_infl", "truncated", "answer_x", "answer_y"]

    rows = []
    if os.path.exists(args.out):
        rows = list(csv.DictReader(open(args.out, encoding="utf-8-sig")))
        done = {r["id"] for r in rows}
        print(f"已有 {len(done)} 条，继续。")
    else:
        done = set()

    t0 = time.time()
    for i, item in enumerate(qs, 1):
        qid = item["id"]
        if qid in done:
            continue
        print(f"[{i}/{len(qs)}] {qid}  {item['q'][:26]}…", flush=True)

        x = ""
        for k in range(3):
            x = gen(item["q"] + GEN_SUFFIX, temp=0.8, seed=101 + k)
            if complete(tok, x, 800):
                break
        y = ""
        for k in range(3):
            y = gen(item["q"] + GEN_SUFFIX, temp=0.8, seed=202 + k)
            if complete(tok, y, 800):
                break
        trunc = not (complete(tok, x, 800) and complete(tok, y, 800))

        lx, ly = ntok(tok, x), ntok(tok, y)

        # 等长：都截到较短者的长度
        base = min(lx, ly)
        x_eq = truncate_to(tok, x, base)
        y_eq = truncate_to(tok, y, base)
        lx_eq, ly_eq = ntok(tok, x_eq), ntok(tok, y_eq)

        # 不等长：把 y 一侧撑到 1.5 倍
        y_inf = pad_to(tok, y_eq, int(max(lx_eq, ly_eq) * 1.5))
        len_infl = ntok(tok, y_inf)

        c1_lab, r1 = judge(item["q"], x_eq, y_eq)      # 等长 正序
        c2_lab, _ = judge(item["q"], y_eq, x_eq)       # 等长 换序
        c3_lab, r3 = judge(item["q"], x_eq, y_inf)     # 不等长 正序
        c4_lab, _ = judge(item["q"], y_inf, x_eq)      # 不等长 换序

        def ident(label, first_name, second_name):
            return {"A": first_name, "B": second_name}.get(label, "平局")

        c1 = ident(c1_lab, "X", "Y")
        c2 = ident(c2_lab, "Y", "X")
        c3 = ident(c3_lab, "X", "Y")
        c4 = ident(c4_lab, "Y", "X")

        print(f"    等长 {lx_eq}/{ly_eq}  C1={c1}({c1_lab}) C2={c2}({c2_lab})"
              f"  |  不等长 {lx_eq}/{len_infl}  C3={c3}({c3_lab}) C4={c4}({c4_lab})",
              flush=True)

        rows.append({
            "id": qid, "type": item["type"],
            "len_x": lx, "len_y": ly, "len_eq_x": lx_eq, "len_eq_y": ly_eq,
            "len_infl": len_infl,
            "c1_win": c1, "c2_win": c2, "c3_win": c3, "c4_win": c4,
            "c1_label": c1_lab, "c2_label": c2_lab, "c3_label": c3_lab, "c4_label": c4_lab,
            "reason_eq": r1, "reason_infl": r3,
            "truncated": int(trunc), "answer_x": x, "answer_y": y,
        })

        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    n = len(rows)
    eq_cons = sum(1 for r in rows if r["c1_win"] == r["c2_win"])
    in_cons = sum(1 for r in rows if r["c3_win"] == r["c4_win"])
    a_rate = sum(1 for r in rows for k in ("c1_label", "c2_label", "c3_label", "c4_label") if r[k] == "A")
    tie = sum(1 for r in rows for k in ("c1_label", "c2_label", "c3_label", "c4_label") if r[k] == "平局")
    pad_win = sum(1 for r in rows if r["c3_win"] == "Y")

    print("\n" + "=" * 64)
    print(f"样本数                  : {n}")
    print(f"等长  顺序一致率        : {eq_cons}/{n}  ({eq_cons/n:.0%})")
    print(f"不等长 顺序一致率       : {in_cons}/{n}  ({in_cons/n:.0%})")
    print(f"选 A 标签的比例         : {a_rate}/{4*n}  ({a_rate/(4*n):.0%})")
    print(f"平局率                  : {tie}/{4*n}  ({tie/(4*n):.0%})")
    print(f"加长一侧(C3)获胜        : {pad_win}/{n}  ({pad_win/n:.0%})")
    print(f"被截断样本              : {sum(int(r['truncated']) for r in rows)}")
    print(f"耗时                    : {(time.time()-t0)/60:.1f} 分钟")
    print(f"结果                    : {args.out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
