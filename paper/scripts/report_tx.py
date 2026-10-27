"""Text routing layer. Every text block the report builder writes passes through tx_segs(): a block is a list of
(text, flags) segments (flags: b = bold, i = italic, sup, sub).
  mode 'log'   : blocks are recorded in the order they are written;
  mode 'apply' : each block is rebuilt from the list of new words computed by make_h_report.align(), keeping the formatting of
                 the word each new word was aligned with."""
import re


class _TX:
    def __init__(self):
        self.reset("off")

    def reset(self, mode, new=None):
        self.mode = mode          # "off" | "log" | "apply"
        self.blocks = []
        self.counter = 0
        self.new = new or {}      # bid -> [(word, owner_index or None, is_equal)]
        self.ctx = None           # (table id, row, col) while a table cell is being written
        self.table_counter = 0
        self.front_start = 0


TX = _TX()


def _flags_key(fl):
    return (fl.get("b"), fl.get("i"), bool(fl.get("sup")), bool(fl.get("sub")))


def _rebuild(segs, words):
    text = "".join(t for t, _ in segs)
    char_fl = []
    for t, fl in segs:
        char_fl.extend([fl] * len(t))
    toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    lead = text[:len(text) - len(text.lstrip())]
    trail = text[len(text.rstrip()):]
    out = []                      # list of (text, flags)

    def push(t, fl):
        if out and _flags_key(out[-1][1]) == _flags_key(fl):
            out[-1] = (out[-1][0] + t, out[-1][1])
        else:
            out.append((t, fl))

    plain = {}
    prev_fl = segs[0][1] if segs else {}
    if lead:
        push(lead, prev_fl)
    for n, (w, ti, eq) in enumerate(words):
        if n:
            push(" ", {k: v for k, v in prev_fl.items() if k in ("b", "i")})
        reuse = False
        if ti is not None and 0 <= ti < len(toks):
            s, e = toks[ti]
            orig = text[s:e]
            if eq and w == orig:
                for ch_i in range(s, e):
                    push(text[ch_i], char_fl[ch_i])
                prev_fl = char_fl[e - 1]
                reuse = True
            else:
                fl0 = {k: v for k, v in char_fl[s].items() if k in ("b", "i")}
                push(w, fl0)
                prev_fl = fl0
        else:
            fl0 = {k: v for k, v in prev_fl.items() if k in ("b", "i")}
            push(w, fl0)
            prev_fl = fl0
    if trail:
        push(trail, {k: v for k, v in prev_fl.items() if k in ("b", "i")})
    return out


def tx_segs(segs, kind="p"):
    if TX.mode == "off":
        return segs
    text = "".join(t for t, _ in segs)
    if not text.strip():
        return segs
    bid = TX.counter
    TX.counter += 1
    if TX.mode == "log":
        TX.blocks.append(dict(bid=bid, text=text, kind=kind, ctx=TX.ctx))
        return segs
    new = TX.new.get(bid)
    if new is None:
        return segs
    return _rebuild(segs, new)


def tx_text(text, kind="p"):
    """Single-segment convenience wrapper; returns the text to write."""
    if TX.mode == "off" or not text or not text.strip():
        return text
    out = tx_segs([(text, {})], kind)
    return "".join(t for t, _ in out)
