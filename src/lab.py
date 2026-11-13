#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动实验台 v2

思路：先把回答「存」起来，再「选」两篇来比。不用粘贴、不会搞混。
所有回答都存在 pilot/answers/ 下，编号从 1 开始。

用法：
    .venv/bin/python pilot/lab.py
"""

import os, re, glob
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_MODEL = os.environ.get("JUDGE_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit")
ADIR = os.path.join(HERE, "answers")
os.makedirs(ADIR, exist_ok=True)

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


def paths():
    return sorted(glob.glob(os.path.join(ADIR, "*.txt")))


def save(text):
    n = len(paths()) + 1
    p = os.path.join(ADIR, f"{n:04d}.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return n, p


def load_one(n):
    p = os.path.join(ADIR, f"{n:04d}.txt")
    if not os.path.exists(p):
        return None
    return open(p, encoding="utf-8").read()


def preview(t, k=46):
    t = " ".join(t.split())
    return t[:k] + ("…" if len(t) > k else "")


def read_block(hint):
    print(f"\n{hint}")
    print("（粘贴内容；粘贴完后，单独输入一个英文句点 . 回车）")
    out = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        out.append(line)
    return "\n".join(out).strip()


def main():
    model_id = os.environ.get("JUDGE_MODEL", DEFAULT_MODEL)
    print(f"加载模型：{model_id}")
    model, tok = load(model_id)
    print("就绪。\n")

    def gen(prompt, temp=0.8, max_tokens=400):
        p = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                    add_generation_prompt=True, tokenize=False)
        return generate(model, tok, prompt=p, max_tokens=max_tokens,
                        sampler=make_sampler(temp=temp), verbose=False).strip()

    last = {}

    def show_list():
        ps = paths()
        if not ps:
            print("\n（还没有保存的回答）")
            return
        print("\n已保存的回答：")
        for p in ps:
            n = int(os.path.basename(p)[:4])
            t = open(p, encoding="utf-8").read()
            print(f"  #{n:<3} {len(t):>4} 字  {preview(t)}")

    def compare(na, nb, tag):
        q = last.get("q") or input("\n这次比较对应的问题是？（回车跳过）: ").strip()
        a, b = load_one(na), load_one(nb)
        if a is None or b is None:
            print("\n编号不存在。")
            return
        print(f"\n正在判定：A = #{na}（{len(a)}字 / {len(tok.encode(a))} tok）"
              f"　B = #{nb}（{len(b)}字 / {len(tok.encode(b))} tok）")
        raw = gen(JUDGE.format(q=q or "（未提供问题）", a=a, b=b), temp=0.0, max_tokens=90)
        v = re.search(r"胜者\s*[:：]\s*(A|B|平局)", raw)
        r = re.search(r"理由\s*[:：]\s*(.+)", raw)
        print("-" * 58)
        print(f"[{tag}]  胜者：{v.group(1) if v else '解析失败'}")
        if r:
            print(f"        理由：{r.group(1).strip()}")
        print("-" * 58)
        last.update(q=q, na=na, nb=nb)

    while True:
        print("""
┌─ 手动实验台 v2 ──────────────────────────┐
│ 1  生成一个回答（自动保存）              │
│ 2  粘贴一篇自定义回答（自动保存）        │
│ 3  比较两篇已保存的回答                  │
│ 4  把上一次的比较交换顺序再判一次        │
│ 5  查看已保存的回答                      │
│ 6  退出                                  │
└──────────────────────────────────────────┘""")
        c = input("选择: ").strip()

        if c == "1":
            q = input("\n问题（单行，回车确认）：").strip()
            if not q:
                continue
            last["q"] = q
            print("\n生成中…（10~20 秒）")
            ans = gen(q)
            n, _ = save(ans)
            print("\n" + "=" * 58)
            print(ans)
            print("=" * 58)
            print(f"已保存为 #{n}")

        elif c == "2":
            t = read_block("粘贴你的回答：")
            if not t:
                continue
            n, _ = save(t)
            print(f"\n已保存为 #{n}（{len(t)} 字 / {len(tok.encode(t))} tok）")

        elif c == "3":
            show_list()
            try:
                na = int(input("\n第一个回答的编号（作为 A）: ").strip())
                nb = int(input("第二个回答的编号（作为 B）: ").strip())
            except ValueError:
                print("\n请输入数字编号。")
                continue
            compare(na, nb, "正序")

        elif c == "4":
            if "na" not in last:
                print("\n还没有做过比较，请先做 3。")
                continue
            compare(last["nb"], last["na"], "交换后")

        elif c == "5":
            show_list()

        elif c == "6":
            break
        else:
            print("\n输入 1~6。")


if __name__ == "__main__":
    main()
