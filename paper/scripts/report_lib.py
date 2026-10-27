"""Helpers for building the BCSE497J project report on top of the VIT template (python-docx)."""
import copy
import io
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph
from report_tx import TX, tx_segs, tx_text

FONT = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
BODY_PT = 12


def set_run_font(run, size=None, bold=None, italic=None, name=FONT, color=None):
    run.font.name = name
    rpr = run._r.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts"); rpr.insert(0, rfonts)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(a), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)


def clear_paragraph(p):
    for child in list(p._p):
        if child.tag != qn("w:pPr"):
            p._p.remove(child)


def delete_paragraph(p):
    el = p._p if hasattr(p, "_p") else p
    el.getparent().remove(el)


def shade_cell(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


class Report:
    def __init__(self, doc, refs: dict, fig_labels: dict, tab_labels: dict):
        self.doc = doc
        self.refs = refs
        self.cite_order: list[str] = []
        self.ch = 0
        self.fig_in_ch = 0
        self.tab_in_ch = 0
        self.eq_in_ch = 0
        self.fig_labels = fig_labels     # key -> expected label "4.1"
        self.tab_labels = tab_labels
        self.figures: list[tuple[str, str]] = []     # (label, caption)
        self.tables: list[tuple[str, str]] = []
        self.headings: list[tuple[int, str]] = []    # (level, text) for TOC; level 0 = chapter
        self.used_figs, self.used_tabs = set(), set()
        self.bm_counter = 1000

    def _bookmark(self, p):
        self.bm_counter += 1
        name = f"_RefApo{self.bm_counter}"
        start = OxmlElement("w:bookmarkStart"); start.set(qn("w:id"), str(self.bm_counter)); start.set(qn("w:name"), name)
        end = OxmlElement("w:bookmarkEnd"); end.set(qn("w:id"), str(self.bm_counter))
        pPr = p._p.pPr
        if pPr is not None:
            pPr.addnext(start)
        else:
            p._p.insert(0, start)
        p._p.append(end)
        return name

    # ------------------------------------------------------------ inline markup
    def _cite_text(self, keys):
        nums = []
        for k in keys:
            if k not in self.refs:
                raise KeyError(f"unknown reference key {k}")
            if k not in self.cite_order:
                self.cite_order.append(k)
            nums.append(self.cite_order.index(k) + 1)
        nums = sorted(set(nums))
        return "".join(f"[{n}]" for n in nums)

    def runs(self, p, text, size=BODY_PT, bold=None, italic=None, color=None):
        text = re.sub(r"\{cite:([^}]+)\}", lambda m: self._cite_text([k.strip() for k in m.group(1).split(",")]), text)
        text = re.sub(r"(?<![A-Za-z_])([A-Za-z])_ℓ", r"\1_{ℓ}", text)
        text = re.sub(r"\b(s|N)_k\b", r"\1_{k}", text)
        text = re.sub(r"\bd_model\b", "d_{model}", text)
        text = re.sub(r"\{fig:([a-z_0-9]+)\}", lambda m: "Figure " + self.fig_labels[m.group(1)], text)
        text = re.sub(r"\{tab:([a-z_0-9]+)\}", lambda m: "Table " + self.tab_labels[m.group(1)], text)
        segs = []
        for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]+\}|_\{[^}]+\})", text):
            if not tok:
                continue
            fl = {"b": bold, "i": italic}
            if tok.startswith("**"):
                tok, fl["b"] = tok[2:-2], True
            elif tok.startswith("*"):
                tok, fl["i"] = tok[1:-1], True
            elif tok.startswith("^{"):
                tok, fl["sup"] = tok[2:-1], True
            elif tok.startswith("_{"):
                tok, fl["sub"] = tok[2:-1], True
            segs.append((tok, fl))
        for tok, fl in tx_segs(segs, "cell" if TX.ctx else "p"):
            if not tok:
                continue
            r = p.add_run(tok)
            set_run_font(r, size=size, bold=fl.get("b"), italic=fl.get("i"), color=color)
            if fl.get("sup"):
                r.font.superscript = True
            if fl.get("sub"):
                r.font.subscript = True
        return p

    # ------------------------------------------------------------ paragraphs
    def _para(self, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6, space_before=0, line=1.15, keep_next=False, indent=None,
              first=None):
        p = self.doc.add_paragraph()
        pf = p.paragraph_format
        p.alignment = align
        pf.space_after = Pt(space_after)
        pf.space_before = Pt(space_before)
        pf.line_spacing = line
        pf.keep_with_next = keep_next
        if indent is not None:
            pf.left_indent = Inches(indent)
        if first is not None:
            pf.first_line_indent = Inches(first)
        return p

    @staticmethod
    def _outline(p, level):
        pPr = p._p.get_or_add_pPr()
        el = OxmlElement("w:outlineLvl"); el.set(qn("w:val"), str(level))
        pPr.insert_element_before(el, "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange")

    def chapter(self, num, title, new_page=True):
        self.ch = num
        self.fig_in_ch = self.tab_in_ch = self.eq_in_ch = 0
        p = self._para(WD_ALIGN_PARAGRAPH.CENTER, space_after=4, line=1.15, keep_next=True)
        p.paragraph_format.page_break_before = new_page
        r = p.add_run(f"CHAPTER {num}"); set_run_font(r, 16, bold=True)
        self._outline(p, 0)
        bm = self._bookmark(p)
        p2 = self._para(WD_ALIGN_PARAGRAPH.CENTER, space_after=12, line=1.15, keep_next=True)
        r = p2.add_run(tx_text(title.upper(), "h")); set_run_font(r, 16, bold=True)
        self.headings.append((0, f"{num}|{title.upper()}", bm))
        return p

    def h2(self, text):
        p = self._para(WD_ALIGN_PARAGRAPH.LEFT, space_after=6, space_before=10, line=1.15, keep_next=True)
        r = p.add_run(tx_text(text, "h")); set_run_font(r, 14, bold=True)
        self._outline(p, 1)
        self.headings.append((1, text, self._bookmark(p)))
        return p

    def h3(self, text):
        p = self._para(WD_ALIGN_PARAGRAPH.LEFT, space_after=4, space_before=8, line=1.15, keep_next=True)
        r = p.add_run(tx_text(text, "h")); set_run_font(r, 12, bold=True)
        self._outline(p, 2)
        self.headings.append((2, text, self._bookmark(p)))
        return p

    def h4(self, text):
        p = self._para(WD_ALIGN_PARAGRAPH.LEFT, space_after=3, space_before=6, line=1.15, keep_next=True)
        r = p.add_run(text); set_run_font(r, 12, bold=True, italic=True)
        return p

    def p(self, text, indent_first=False, space_after=6, keep_next=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
        para = self._para(align, space_after=space_after, keep_next=keep_next, first=0.3 if indent_first else None)
        self.runs(para, text)
        if not para.text.strip():
            delete_paragraph(para)
        return para

    def bullets(self, items, numbered=False, indent=0.35):
        for n, it in enumerate(items, 1):
            para = self._para(WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=3, indent=indent, first=-0.25)
            mark = para.add_run(f"{n}.\t" if numbered else "\u2022\t"); set_run_font(mark, BODY_PT)
            self.runs(para, it)
            if not para.text.replace("\t", "").replace("\u2022", "").strip() and not any(ch.isalnum() for ch in para.text):
                delete_paragraph(para)
                continue
            para.paragraph_format.tab_stops.add_tab_stop(Inches(indent))
        sp = self._para(space_after=2, line=0.5)

    # ------------------------------------------------------------ equations
    def equation(self, latex, fontsize=13):
        self.eq_in_ch += 1
        fig = plt.figure(figsize=(0.1, 0.1))
        fig.text(0, 0, f"${latex}$", fontsize=fontsize)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=330, bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)
        buf.seek(0)
        from PIL import Image
        w_px, h_px = Image.open(buf).size
        buf.seek(0)
        t = self.doc.add_table(rows=1, cols=3)
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for c, w in zip(t.rows[0].cells, (Inches(0.5), Inches(5.3), Inches(0.6))):
            c.width = w
        mid = t.rows[0].cells[1].paragraphs[0]
        mid.alignment = WD_ALIGN_PARAGRAPH.CENTER
        mid.add_run().add_picture(buf, width=Inches(min(w_px / 330.0, 5.2)))
        mid.paragraph_format.space_after = Pt(0)
        num = t.rows[0].cells[2].paragraphs[0]
        num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r = num.add_run(f"({self.ch}.{self.eq_in_ch})"); set_run_font(r, BODY_PT)
        for c in t.rows[0].cells[1:]:
            c.vertical_alignment = 1
        sp = self.doc.add_paragraph(); sp.paragraph_format.space_after = Pt(2); sp.paragraph_format.line_spacing = 0.5
        return f"{self.ch}.{self.eq_in_ch}"

    # ------------------------------------------------------------ figures and tables
    def figure(self, key, path, caption, width=6.0):
        self.fig_in_ch += 1
        label = f"{self.ch}.{self.fig_in_ch}"
        exp = self.fig_labels.get(key)
        assert exp in (None, "?.?", label), f"figure {key}: expected {exp} but placed at {label}"
        self.fig_labels[key] = label
        self.used_figs.add(key)
        para = self._para(WD_ALIGN_PARAGRAPH.CENTER, space_after=3, space_before=4, line=1.0, keep_next=True)
        para.add_run().add_picture(str(path), width=Inches(width))
        cap = self._para(WD_ALIGN_PARAGRAPH.CENTER, space_after=10, line=1.0)
        r = cap.add_run(f"Figure {label} "); set_run_font(r, 10, bold=True)
        self.runs(cap, caption, size=10)
        self.figures.append((label, caption, self._bookmark(cap)))
        return label

    def table(self, key, header, rows, caption, widths, font=10, align=None, bold_rows=(), shade_rows=(), note=None):
        self.tab_in_ch += 1
        label = f"{self.ch}.{self.tab_in_ch}"
        exp = self.tab_labels.get(key)
        assert exp in (None, "?.?", label), f"table {key}: expected {exp} but placed at {label}"
        self.tab_labels[key] = label
        self.used_tabs.add(key)
        cap = self._para(WD_ALIGN_PARAGRAPH.CENTER, space_after=3, space_before=8, line=1.0, keep_next=True)
        r = cap.add_run(f"Table {label} "); set_run_font(r, 10, bold=True)
        self.runs(cap, caption, size=10)
        self.tables.append((label, caption, self._bookmark(cap)))
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = self.doc.styles["Table Grid"]
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        t.autofit = False
        align = align or ["l"] + ["c"] * (len(header) - 1)
        amap = {"l": WD_ALIGN_PARAGRAPH.LEFT, "c": WD_ALIGN_PARAGRAPH.CENTER}

        TX.table_counter += 1
        tid = TX.table_counter

        def fill(cell, txt, al, bold=False):
            cell.text = ""
            para = cell.paragraphs[0]
            para.alignment = amap[al]
            para.paragraph_format.space_after = Pt(1)
            para.paragraph_format.space_before = Pt(1)
            para.paragraph_format.line_spacing = 1.0
            self.runs(para, str(txt), size=font, bold=True if bold else None)
            TX.ctx = None

        for i, htxt in enumerate(header):
            c = t.rows[0].cells[i]
            TX.ctx = (tid, 0, i)
            fill(c, htxt, "c" if align[i] == "c" else "l", bold=True)
            shade_cell(c, "D9E2F3")
        trPr = t.rows[0]._tr.get_or_add_trPr()
        el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trPr.append(el)
        for ri, row in enumerate(rows, start=1):
            trPr = t.rows[ri]._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit"); cs.set(qn("w:val"), "true"); trPr.append(cs)
            for ci, val in enumerate(row):
                c = t.rows[ri].cells[ci]
                TX.ctx = (tid, ri, ci)
                fill(c, val, align[ci], bold=(ri - 1) in bold_rows)
                if (ri - 1) in shade_rows:
                    shade_cell(c, "F2F2F2")
        for row in t.rows:
            for c, w in zip(row.cells, widths):
                c.width = Inches(w)
        n = self._para(WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=8, space_before=2, line=1.0)
        if note:
            self.runs(n, note, size=10)
        return label

    # ------------------------------------------------------------ code
    def code(self, lines, size=8.5):
        for ln in lines:
            para = self.doc.add_paragraph()
            pf = para.paragraph_format
            pf.space_after = Pt(0); pf.space_before = Pt(0); pf.line_spacing = 1.0
            pf.left_indent = Inches(0.1)
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = para.add_run(ln if ln else " ")
            set_run_font(r, size, name="Courier New")
            pPr = para._p.get_or_add_pPr()
            shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F2F2F2")
            pPr.insert_element_before(shd, "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap", "w:overflowPunct",
                                      "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi", "w:adjustRightInd",
                                      "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
                                      "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment", "w:textboxTightWrap",
                                      "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr", "w:pPrChange")
        self.doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # ------------------------------------------------------------ references
    def reference_list(self):
        for i, k in enumerate(self.cite_order, start=1):
            para = self._para(WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=4, indent=0.4, first=-0.4, line=1.15)
            mark = para.add_run(f"[{i}]\t"); set_run_font(mark, BODY_PT)
            self.runs(para, self.refs[k], size=BODY_PT)
            para.paragraph_format.tab_stops.add_tab_stop(Inches(0.4))
