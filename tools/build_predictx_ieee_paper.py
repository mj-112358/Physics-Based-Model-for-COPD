from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Physics_Informed_Censored_Spirometry_IEEE_Paper.docx"
ASSETS = ROOT / "paper_assets"
ASSETS.mkdir(exist_ok=True)

BLACK = "000000"
BLUE = "000000"
LIGHT_BLUE = "E7E7E7"
PALE_BLUE = "F4F4F4"
LIGHT_GRAY = "D9D9D9"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=55, start=70, bottom=55, end=70) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_cell_borders(cell, color=LIGHT_GRAY, size="4") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        elem = borders.find(tag)
        if elem is None:
            elem = OxmlElement(f"w:{edge}")
            borders.append(elem)
        elem.set(qn("w:val"), "single")
        elem.set(qn("w:sz"), size)
        elem.set(qn("w:color"), color)


def set_font(run, name="Times New Roman", size=10.0, bold=False, italic=False, color=BLACK):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    fonts = rpr.rFonts
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.insert(0, fonts)
    fonts.set(qn("w:ascii"), name)
    fonts.set(qn("w:hAnsi"), name)
    fonts.set(qn("w:eastAsia"), name)
    return run


def set_columns(section, num=2, space_twips=360) -> None:
    sect_pr = section._sectPr
    cols = sect_pr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect_pr.append(cols)
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(space_twips))
    cols.set(qn("w:equalWidth"), "1")


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.extend([fld_char1, instr, fld_char2])
    set_font(run, size=8)


def reorder_major_sections(doc: Document) -> None:
    """Place analysis before discussion and conclusion before appendices.

    The manuscript is assembled in evidence-generation order because later
    sections reuse earlier result objects. This final XML-level pass changes
    presentation order only; every paragraph, table, figure, and section break
    remains the same document node.
    """
    body = doc._element.body
    terminal_sect_pr = body.sectPr
    children = [child for child in body if child is not terminal_sect_pr]

    def element_text(element) -> str:
        return "".join(node.text or "" for node in element.iter(qn("w:t"))).strip()

    source_order = [
        "I  INTRODUCTION",
        "II  RELATED WORK AND TECHNICAL GAP",
        "III  DATA AND STUDY DESIGN",
        "IV  SIGNAL RECONSTRUCTION",
        "V  PHYSIOLOGY-CONSTRAINED CENSORED-CURVE MODEL",
        "VI  MULTI-MANOEUVRE SELECTIVE INFERENCE",
        "VII  EXPERIMENTAL PROTOCOL",
        "VIII  RESULTS",
        "IX  ABLATION AND MODEL-SELECTION EVIDENCE",
        "X  ROBUSTNESS AND WITHIN-PROGRAM TRANSPORT",
        "XI  DISCUSSION",
        "XII  LIMITATIONS",
        "XIII  DEPLOYMENT CONSIDERATIONS",
        "XIV  MATHEMATICAL ANALYSIS OF CENSORING",
        "XV  OPTIMIZATION AND MODEL IDENTIFICATION",
        "XVI  UNCERTAINTY AND SELECTIVE-RISK CALIBRATION",
        "XVII  REPRODUCIBILITY AND EXTENSION STUDIES",
        "APPENDIX A  SPIROMETRY INFERENCE ALGORITHM",
        "APPENDIX B  CLAIM TO EVIDENCE MAP",
        "XVIII  CONCLUSION",
        "REFERENCES",
    ]
    positions = {}
    for index, child in enumerate(children):
        text = element_text(child)
        if text in source_order:
            positions[text] = index
    missing = [heading for heading in source_order if heading not in positions]
    if missing:
        raise RuntimeError(f"Cannot reorder manuscript; missing headings: {missing}")

    blocks = {}
    for i, heading in enumerate(source_order):
        start = positions[heading]
        stop = positions[source_order[i + 1]] if i + 1 < len(source_order) else len(children)
        blocks[heading] = children[start:stop]

    presentation_order = [
        "I  INTRODUCTION",
        "II  RELATED WORK AND TECHNICAL GAP",
        "III  DATA AND STUDY DESIGN",
        "IV  SIGNAL RECONSTRUCTION",
        "V  PHYSIOLOGY-CONSTRAINED CENSORED-CURVE MODEL",
        "VI  MULTI-MANOEUVRE SELECTIVE INFERENCE",
        "VII  EXPERIMENTAL PROTOCOL",
        "VIII  RESULTS",
        "IX  ABLATION AND MODEL-SELECTION EVIDENCE",
        "X  ROBUSTNESS AND WITHIN-PROGRAM TRANSPORT",
        "XIV  MATHEMATICAL ANALYSIS OF CENSORING",
        "XV  OPTIMIZATION AND MODEL IDENTIFICATION",
        "XVI  UNCERTAINTY AND SELECTIVE-RISK CALIBRATION",
        "XI  DISCUSSION",
        "XII  LIMITATIONS",
        "XIII  DEPLOYMENT CONSIDERATIONS",
        "XVII  REPRODUCIBILITY AND EXTENSION STUDIES",
        "XVIII  CONCLUSION",
        "APPENDIX A  SPIROMETRY INFERENCE ALGORITHM",
        "APPENDIX B  CLAIM TO EVIDENCE MAP",
        "REFERENCES",
    ]
    renamed = {
        "XIV  MATHEMATICAL ANALYSIS OF CENSORING": "XI  MATHEMATICAL ANALYSIS OF CENSORING",
        "XV  OPTIMIZATION AND MODEL IDENTIFICATION": "XII  OPTIMIZATION AND MODEL IDENTIFICATION",
        "XVI  UNCERTAINTY AND SELECTIVE-RISK CALIBRATION": "XIII  UNCERTAINTY AND SELECTIVE-RISK CALIBRATION",
        "XI  DISCUSSION": "XIV  DISCUSSION",
        "XII  LIMITATIONS": "XV  LIMITATIONS",
        "XIII  DEPLOYMENT CONSIDERATIONS": "XVI  DEPLOYMENT CONSIDERATIONS",
    }
    for old_heading, new_heading in renamed.items():
        text_nodes = list(blocks[old_heading][0].iter(qn("w:t")))
        text_nodes[0].text = new_heading
        for node in text_nodes[1:]:
            node.text = ""

    preamble = children[: positions[source_order[0]]]
    for child in list(body):
        body.remove(child)
    for child in preamble:
        body.append(child)
    for heading in presentation_order:
        for child in blocks[heading]:
            body.append(child)
    body.append(terminal_sect_pr)


def add_body(doc, text: str, first_indent=0.15, before=0, after=2.0, keep=False):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(first_indent)
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_together = keep
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    set_font(p.add_run(text))
    return p


def add_heading(doc, text: str, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6 if level == 1 else 4)
    p.paragraph_format.space_after = Pt(2)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
    size = 10.0 if level == 1 else 9.2
    run = p.add_run(text)
    set_font(run, size=size, bold=True)
    if level == 1:
        run.font.all_caps = True
    else:
        run.italic = True
    return p


def add_caption(doc, text: str, kind="Fig"):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = False
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r1 = p.add_run(text.split(". ", 1)[0] + ". ")
    set_font(r1, size=8.2, bold=True)
    rest = text.split(". ", 1)[1] if ". " in text else ""
    set_font(p.add_run(rest), size=8.2)
    return p


def add_figure(doc, path: Path, caption: str, width=3.18):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, caption)


def add_wide_figure(doc, path: Path, caption: str, width=6.75):
    """Insert an IEEE figure* style full-width figure, then resume two columns."""
    wide = doc.add_section(WD_SECTION.CONTINUOUS)
    wide.top_margin = Inches(0.65)
    wide.bottom_margin = Inches(0.65)
    wide.left_margin = Inches(0.72)
    wide.right_margin = Inches(0.72)
    set_columns(wide, 1)
    wide.footer.is_linked_to_previous = True

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    add_caption(doc, caption)

    narrow = doc.add_section(WD_SECTION.CONTINUOUS)
    narrow.top_margin = Inches(0.65)
    narrow.bottom_margin = Inches(0.65)
    narrow.left_margin = Inches(0.72)
    narrow.right_margin = Inches(0.72)
    set_columns(narrow, 2, 340)
    narrow.footer.is_linked_to_previous = True


def add_equation(doc, path: Path, width=3.05):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_together = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))


def add_table(doc, caption: str, headers, rows, widths=None, font_size=7.2):
    cp = doc.add_paragraph()
    cp.paragraph_format.keep_with_next = True
    cp.paragraph_format.space_before = Pt(4)
    cp.paragraph_format.space_after = Pt(2)
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_font(cp.add_run(caption), size=7.8, bold=True)

    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    if widths:
        grid_cols = table._tbl.tblGrid.gridCol_lst
        for j, width in enumerate(widths):
            grid_cols[j].set(qn("w:w"), str(int(width * 1440)))
    table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
    for j, h in enumerate(headers):
        c = table.rows[0].cells[j]
        set_cell_shading(c, "FFFFFF")
        set_cell_borders(c)
        set_cell_margins(c)
        c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        set_font(p.add_run(str(h)), size=font_size, bold=True, color="000000")
        if widths:
            c.width = Inches(widths[j])
            c._tc.get_or_add_tcPr().get_or_add_tcW().w = int(widths[j] * 1440)
    for i, row in enumerate(rows):
        cells = table.add_row().cells
        for j, value in enumerate(row):
            c = cells[j]
            set_cell_shading(c, "FFFFFF")
            set_cell_borders(c)
            set_cell_margins(c)
            c.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0)
            set_font(p.add_run(str(value)), size=font_size)
            if widths:
                c.width = Inches(widths[j])
                c._tc.get_or_add_tcPr().get_or_add_tcW().w = int(widths[j] * 1440)
    # Keep compact IEEE tables as a single visual unit whenever the current
    # column cannot accommodate them.  The row-level flag prevents an
    # individual row from splitting; keep-with-next links each row to the one
    # below so Word moves the complete table to the next column when needed.
    for row_index, table_row in enumerate(table.rows):
        table_row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if row_index < len(table.rows) - 1:
            for cell in table_row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)
    return table


def make_assets():
    def resolve_figure_font(*candidates: str) -> str:
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        raise FileNotFoundError(f"No publication serif font found among: {candidates}")

    font_path = resolve_figure_font(
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    )
    bold_path = resolve_figure_font(
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    )
    italic_path = resolve_figure_font(
        "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Italic.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSerif-Italic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    )
    def font(size, bold=False, italic=False):
        path = bold_path if bold else italic_path if italic else font_path
        return ImageFont.truetype(path, size)

    def centered(draw, xy, text, fnt, fill="#000000"):
        box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=4, align="center")
        draw.multiline_text((xy[0]-(box[2]-box[0])/2, xy[1]-(box[3]-box[1])/2), text, font=fnt, fill=fill, spacing=4, align="center")

    def arrow(draw, start, end, color="#17365D", width=4):
        draw.line([start, end], fill=color, width=width)
        x2, y2 = end; x1, y1 = start
        ang = np.arctan2(y2-y1, x2-x1)
        for d in (2.55, -2.55):
            p = (x2 + 16*np.cos(ang+d), y2 + 16*np.sin(ang+d))
            draw.line([end, p], fill=color, width=width)

    # Column-width architecture figure for the frozen implementation.
    img = Image.new("RGB", (900, 1260), "white")
    d = ImageDraw.Draw(img)
    boxes = [
        (175, 30, 725, 155, "Raw spirometry waveforms", "#E8E8E8"),
        (175, 215, 725, 355, "Signal reconstruction\nand ATS onset", "#F2F2F2"),
        (175, 415, 725, 555, "Endpoint-quality state\nand candidate selection", "#F2F2F2"),
        (175, 635, 725, 790, "Bounded volume model\nand corrected ratio", "#F2F2F2"),
        (175, 870, 725, 1015, "Repeated-manoeuvre evidence\nand physics policy", "#E8E8E8"),
        (175, 1085, 725, 1245, "Physics-informed reliability\nFinalize or defer", "#F2F2F2"),
    ]
    for x1, y1, x2, y2, label, color in boxes:
        d.rounded_rectangle((x1,y1,x2,y2), radius=14, outline="#17365D", fill=color, width=4)
        centered(d, ((x1+x2)/2,(y1+y2)/2), label, font(34))
    for a,b in [
        ((450,155),(450,215)), ((450,355),(450,415)),
        ((450,555),(450,635)), ((450,790),(450,870)),
        ((450,1015),(450,1085)),
    ]:
        arrow(d,a,b,width=5)
    img.save(ASSETS / "architecture.png", dpi=(300,300))

    # Common chart canvas helpers
    def chart_base(title_y=False):
        im = Image.new("RGB", (1100, 700), "white")
        dr = ImageDraw.Draw(im)
        left, top, right, bottom = 145, 70, 1040, 585
        dr.line((left,top,left,bottom), fill="#333333", width=3)
        dr.line((left,bottom,right,bottom), fill="#333333", width=3)
        for frac in np.linspace(0,1,6):
            y = bottom - frac*(bottom-top)
            dr.line((left,y,right,y), fill="#E6E6E6", width=2)
            dr.text((65,y-14), f"{frac:.1f}", font=font(24), fill="#333333")
        return im, dr, (left,top,right,bottom)

    # Censoring schematic
    img, d, (left,top,right,bottom) = chart_base()
    t = np.linspace(0, 8, 801)
    tf, ts, a, fvc, to = 0.35, 2.5, 0.60, 4.2, 0.12
    q = (a*np.exp(-t/tf)/tf + (1-a)*np.exp(-t/ts)/ts) * (1-np.exp(-t/to))
    q = fvc*q/np.trapezoid(q,t)
    v = np.concatenate([[0], np.cumsum((q[:-1]+q[1:])*0.5*np.diff(t))])
    def pt(tt,vv): return (left+(tt/8)*(right-left), bottom-(vv/4.5)*(bottom-top))
    d.line([pt(tt,vv) for tt,vv in zip(t,v)], fill="#17365D", width=6)
    for stop,col in [(2,"#C00000"),(3,"#E26B0A")]:
        mask=t<=stop; pts=[pt(tt,vv) for tt,vv in zip(t[mask],v[mask])]
        d.line(pts, fill=col, width=9)
        y=pts[-1][1]; d.line((pts[-1][0],y,right,y), fill=col, width=3)
    y_fvc=pt(0,fvc)[1]; d.line((left,y_fvc,right,y_fvc), fill="#548235", width=3)
    for xval in range(0,9,2): centered(d,(left+xval/8*(right-left),bottom+30),str(xval),font(23))
    centered(d,((left+right)/2,650),"Time from back-extrapolated onset (s)",font(27))
    d.text((left,25), "Expired volume (L)", font=font(25), fill="#000000")
    d.rectangle((650,85,1015,215), fill="white", outline="#D9D9D9", width=2)
    legend=[("#17365D","latent complete trajectory"),("#C00000","observed through 2 s"),("#E26B0A","observed through 3 s"),("#548235","true FVC")]
    for i,(c,label) in enumerate(legend):
        y=110+i*27; d.line((670,y,720,y),fill=c,width=6); d.text((735,y-13),label,font=font(22),fill="#000000")
    img.save(ASSETS / "censoring_schematic.png", dpi=(220,220))

    def horizontal_comparison(path, labels, direct, recovered, height=760, title="Obstruction detection rate"):
        img = Image.new("RGB", (1200, height), "white")
        d = ImageDraw.Draw(img)
        left, right, top, bottom = 260, 1130, 115, height - 90
        for value in range(0, 101, 20):
            x = left + (right-left) * value / 100
            d.line((x, top, x, bottom), fill="#E0E0E0", width=2)
            centered(d, (x, bottom+34), f"{value}%", font(24))
        d.text((left, 22), title, font=font(30, bold=True), fill="#000000")
        d.rectangle((left,64,left+28,88), fill="#7F7F7F")
        d.text((left+40,60), "Truncated endpoint", font=font(25), fill="#000000")
        d.rectangle((left+330,64,left+358,88), fill="#1F4E79")
        d.text((left+370,60), "Physiology-constrained recovery", font=font(25), fill="#000000")
        row_h=(bottom-top)/len(labels)
        for i,(lab,dv,rv) in enumerate(zip(labels,direct,recovered)):
            yc=top+(i+.5)*row_h
            centered(d,(125,yc),lab,font(28,bold=True))
            for offset,val,color in [(-18,dv,"#7F7F7F"),(18,rv,"#1F4E79")]:
                y1=yc+offset-11; y2=yc+offset+11
                x2=left+(right-left)*val
                d.rounded_rectangle((left,y1,x2,y2),radius=9,fill=color)
                label = f"{100*val:.1f}%"
                label_w = d.textbbox((0, 0), label, font=font(23, bold=True))[2]
                label_x = x2 + 12 if x2 + 12 + label_w <= right else x2 - label_w - 12
                label_fill = "#000000" if label_x > x2 else "#FFFFFF"
                d.text((label_x, y1-5), label, font=font(23,bold=True), fill=label_fill)
        img.save(path,dpi=(300,300))

    horizontal_comparison(ASSETS/"pooled_truncation.png",["2 s","3 s"],[.017,.149],[.424,.420],height=600)
    horizontal_comparison(ASSETS/"field_recovery.png",["≤2 s","2–3 s","3–4 s","4–5 s",">5 s","All"],[.036,.231,.333,.615,.889,.573],[.929,.962,.952,.846,.933,.933],height=1000)

    # Cohort-flow figure derived from the auditable participant registry.
    cohort = json.loads((ROOT / "results" / "manuscript_cohort_audit.json").read_text(encoding="utf-8"))
    cohort_by_name = {row["experiment"]: row for row in cohort["cohorts"]}
    img = Image.new("RGB", (2100, 1320), "white")
    d = ImageDraw.Draw(img)

    def flow_box(rect, title, lines, fill="#F5F5F5", title_fill="#17365D"):
        x1, y1, x2, y2 = rect
        d.rounded_rectangle(rect, radius=22, outline="#17365D", fill=fill, width=5)
        centered(d, ((x1+x2)/2, y1+52), title, font(35, bold=True), fill=title_fill)
        centered(d, ((x1+x2)/2, (y1+y2)/2+34), "\n".join(lines), font(29), fill="#000000")

    d.text((70, 30), "Cohort construction and experiment membership", font=font(48, bold=True), fill="#000000")
    flow_box((650, 110, 1450, 270), "Raw NHANES spirometry", ["108,939 manoeuvres", "Cycles E, F, and G"], "#E8E8E8")
    flow_box((540, 345, 1560, 565), "Participants with released FEV1 and FVC", ["n = 20,050; age 33 [16-53] years", "Severe / moderate / mild / normal", "117 / 312 / 1,503 / 18,118"], "#F2F2F2")
    arrow(d, (1050,270), (1050,345), width=6)
    d.text((1590, 410), "Usable A/B candidate\nparticipants: 19,459", font=font(30, italic=True), fill="#333333")

    branch_data = [
        ("Controlled\ntruncation", "Controlled truncation sample", "n = 1,492; age 52 [31-66]", "67 / 225 / 600 / 600", "Obstructed set: n = 892"),
        ("Natural incomplete\npairs", "Natural incomplete pairs", "n = 286; age 51 [28-66]", "13 / 55 / 110 / 108", "One incomplete effort each"),
        ("Cycle F policy\ndevelopment", "Cycle F policy-development test", "n = 563; age 49 [26-63]", "16 / 47 / 250 / 250", "Participant-disjoint test"),
        ("Frozen Cycle E/G\nworkflow", "Frozen cross-cycle workflow", "n = 12,434; age 33 [16-53]", "62 / 174 / 933 / 11,265", "Rules frozen before scoring"),
    ]
    xs = [(55, 610, 505, 1030), (565, 610, 1015, 1030), (1075, 610, 1525, 1030), (1585, 610, 2035, 1030)]
    for (title, key, age_line, sev_line, final_line), rect in zip(branch_data, xs):
        record = cohort_by_name[key]
        flow_box(rect, title, [age_line, "S / M / mild / normal", sev_line, final_line], "#F7F7F7")
        arrow(d, (1050,565), ((rect[0]+rect[2])//2,610), width=4)
    overlap = cohort["overlap"]
    d.rounded_rectangle((160,1090,1940,1270), radius=18, outline="#7F7F7F", fill="#FFFFFF", width=3)
    centered(d,(1050,1140),"Verified participant overlap",font(34,bold=True))
    centered(
        d,
        (1050,1205),
        f"All 286 natural pairs are members of the Cycle F controlled sample; 75 also occur in the Cycle F TEST partition.\n"
        f"All {overlap['E_controlled_within_frozen_workflow'] + overlap['G_controlled_within_frozen_workflow']} Cycle E/G controlled participants occur in frozen cross-cycle scoring; identifiers do not cross cycles.",
        font(28),
    )
    img.save(ASSETS/"cohort_flow.png",dpi=(300,300))

    # Multi-panel patient-benefit analysis.
    benefit = json.loads((ROOT / "results" / "manuscript_patient_benefit.json").read_text(encoding="utf-8"))
    img = Image.new("RGB", (2200, 1500), "white")
    d = ImageDraw.Draw(img)
    d.text((60, 20), "Where reconstruction and selective inference contribute", font=font(50, bold=True), fill="#000000")

    def panel(rect, label, title):
        x1,y1,x2,y2=rect
        d.rounded_rectangle(rect, radius=12, outline="#BFBFBF", fill="#FFFFFF", width=3)
        d.text((x1+18,y1+10),label,font=font(36,bold=True),fill="#000000")
        d.text((x1+82,y1+12),title,font=font(34,bold=True),fill="#000000")
        return x1+105,y1+88,x2-30,y2-78

    def axes(area, y_label="Detection rate"):
        x1,y1,x2,y2=area
        d.line((x1,y1,x1,y2),fill="#333333",width=3); d.line((x1,y2,x2,y2),fill="#333333",width=3)
        for frac in np.linspace(0,1,5):
            y=y2-frac*(y2-y1); d.line((x1,y,x2,y),fill="#E8E8E8",width=2)
            d.text((x1-82,y-16),f"{frac:.2f}",font=font(27),fill="#333333")
        d.text((x1-96,y1-40),y_label,font=font(28),fill="#333333")
        return x1,y1,x2,y2

    rects=[(35,90,720,735),(755,90,1440,735),(1475,90,2160,735),(35,775,720,1460),(755,775,1440,1460),(1475,775,2160,1460)]
    # A: duration
    a=axes(panel(rects[0],"A","Observation duration"))
    duration=benefit["controlled_duration"]
    for method,color in [("direct","#7F7F7F"),("physics","#1F4E79")]:
        rows=[r for r in duration if r["method"]==method]
        pts=[]
        for r in rows:
            x=a[0]+(r["duration_s"]-2)/3*(a[2]-a[0]); y=a[3]-r["sensitivity"]*(a[3]-a[1]); pts.append((x,y))
        d.line(pts,fill=color,width=7)
        for p in pts: d.ellipse((p[0]-8,p[1]-8,p[0]+8,p[1]+8),fill=color)
    for t in range(2,6): centered(d,(a[0]+(t-2)/3*(a[2]-a[0]),a[3]+34),f"{t} s",font(28))
    d.text((a[0]+10,a[1]+10),"Direct",font=font(27),fill="#7F7F7F"); d.text((a[0]+135,a[1]+10),"Physics",font=font(27,bold=True),fill="#1F4E79")

    # B: severity
    b=axes(panel(rects[1],"B","Baseline severity"))
    severity=benefit["controlled_severity"]
    sev_styles=[("mild_0.60_0.70","#A5A5A5","Mild"),("moderate","#5B9BD5","Moderate"),("severe","#17365D","Severe")]
    for sev,color,label in sev_styles:
        rows=[r for r in severity if r["severity"]==sev]; pts=[]
        for r in rows:
            x=b[0]+(r["duration_s"]-2)/3*(b[2]-b[0]); y=b[3]-r["detection_rate"]*(b[3]-b[1]); pts.append((x,y))
        d.line(pts,fill=color,width=6)
        for p in pts:d.ellipse((p[0]-7,p[1]-7,p[0]+7,p[1]+7),fill=color)
    for t in range(2,6): centered(d,(b[0]+(t-2)/3*(b[2]-b[0]),b[3]+34),f"{t} s",font(28))
    for i,(_,color,label) in enumerate(sev_styles): d.text((b[0]+i*170,b[1]+10),label,font=font(26),fill=color)

    # C: distance from boundary
    c=axes(panel(rects[2],"C","Distance below 0.70"))
    distance=benefit["controlled_distance_to_boundary"]
    bands=[("0-0.05","#BFBFBF"),("0.05-0.10","#8FAADC"),("0.10-0.20","#4472C4"),(">=0.20","#17365D")]
    for band,color in bands:
        rows=[r for r in distance if r["distance_band"]==band]; pts=[]
        for r in rows:
            x=c[0]+(r["duration_s"]-2)/3*(c[2]-c[0]); y=c[3]-r["detection_rate"]*(c[3]-c[1]); pts.append((x,y))
        d.line(pts,fill=color,width=5)
    for t in range(2,6): centered(d,(c[0]+(t-2)/3*(c[2]-c[0]),c[3]+34),f"{t} s",font(28))
    for i,(band,color) in enumerate(bands): d.text((c[0]+(i%2)*250,c[1]+8+(i//2)*38),band,font=font(25),fill=color)

    # D: FVC magnitude at 2 s
    darea=axes(panel(rects[3],"D","FVC magnitude at 2 s"))
    fvcs=benefit["controlled_fvc_magnitude"]; gw=(darea[2]-darea[0])/len(fvcs)
    for i,r in enumerate(fvcs):
        xc=darea[0]+(i+.5)*gw
        for off,key,color in [(-22,"direct_detection","#7F7F7F"),(22,"physics_detection","#1F4E79")]:
            val=r[key]; x1=xc+off-18; x2=xc+off+18; y=darea[3]-val*(darea[3]-darea[1]); d.rectangle((x1,y,x2,darea[3]),fill=color)
        centered(d,(xc,darea[3]+36),r["fvc_band"],font(26))
        d.text((xc-45,darea[1]+15),f"n={r['n']}",font=font(23),fill="#333333")

    # E: repeated manoeuvres
    earea=axes(panel(rects[4],"E","Available manoeuvres"),"Rate")
    mrows=benefit["frozen_maneuver_availability"]; gw=(earea[2]-earea[0])/len(mrows)
    for i,r in enumerate(mrows):
        xc=earea[0]+(i+.5)*gw
        for off,key,color in [(-22,"coverage","#5B9BD5"),(22,"accuracy_when_finalized","#17365D")]:
            val=r[key]; y=earea[3]-val*(earea[3]-earea[1]); d.rectangle((xc+off-18,y,xc+off+18,earea[3]),fill=color)
        centered(d,(xc,earea[3]+36),r["maneuver_band"],font(27))
    d.text((earea[0]+15,earea[1]+8),"Coverage",font=font(25),fill="#5B9BD5"); d.text((earea[0]+175,earea[1]+8),"Accuracy",font=font(25,bold=True),fill="#17365D")

    # F: interval width versus boundary distance
    farea=panel(rects[5],"F","Uncertainty-aware deferral")
    d.line((farea[0],farea[1],farea[0],farea[3]),fill="#333333",width=3); d.line((farea[0],farea[3],farea[2],farea[3]),fill="#333333",width=3)
    irows=benefit["r2_interval_cases"]["rows"]
    max_w=max(r["ci_width"] for r in irows); max_dist=max(r["distance_to_boundary"] for r in irows)
    for r in irows:
        x=farea[0]+r["ci_width"]/max_w*(farea[2]-farea[0]); y=farea[3]-r["distance_to_boundary"]/max_dist*(farea[3]-farea[1])
        color="#1F4E79" if r["finalized"] else "#BFBFBF"; d.ellipse((x-4,y-4,x+4,y+4),fill=color)
    line=[]
    for w in np.linspace(0,max_w,50):
        dist=w/2; line.append((farea[0]+w/max_w*(farea[2]-farea[0]),farea[3]-dist/max_dist*(farea[3]-farea[1])))
    d.line(line,fill="#C00000",width=3)
    d.text((farea[0]+10,farea[1]+10),"Finalized",font=font(25),fill="#1F4E79"); d.text((farea[0]+170,farea[1]+10),"Deferred",font=font(25),fill="#7F7F7F")
    centered(d,((farea[0]+farea[2])/2,farea[3]+42),"Confidence-interval width",font(27)); d.text((farea[0]-92,farea[1]-34),"|median - 0.70|",font=font(24),fill="#333333")
    img.save(ASSETS/"patient_benefit.png",dpi=(300,300))

    # Confusion matrix plus matched-coverage ablation for the final selector.
    img = Image.new("RGB", (1100, 1120), "white")
    d = ImageDraw.Draw(img)
    selector = json.loads((ROOT / "results" / "physics_informed_selector_cross_cycle.json").read_text(encoding="utf-8"))
    e_metrics = selector["cross_cycle"]["E"]["metrics"]
    g_metrics = selector["cross_cycle"]["G"]["metrics"]
    physics = {
        key: e_metrics["physics_reliability"][key] + g_metrics["physics_reliability"][key]
        for key in ("tn", "fp", "fn", "tp", "n")
    }
    sensitivity = physics["tp"] / (physics["tp"] + physics["fn"])
    specificity = physics["tn"] / (physics["tn"] + physics["fp"])
    d.text((65,25),"Cycle E/G physics-informed selection",font=font(31,bold=True),fill="#000000")
    d.text((95,90),"Combined selected-case confusion matrix",font=font(25,bold=True),fill="#000000")
    matrix=[[physics["tn"],physics["fp"]],[physics["fn"],physics["tp"]]]
    colors=[["#E7F0E3","#F8E7E7"],["#F8E7E7","#D9E8F5"]]
    for i in range(2):
        for j in range(2):
            x1=260+j*300; y1=160+i*190; x2=x1+300; y2=y1+190
            d.rectangle((x1,y1,x2,y2),outline="#7F7F7F",fill=colors[i][j],width=3)
            centered(d,((x1+x2)/2,(y1+y2)/2),f"{matrix[i][j]:,}",font(42,bold=True))
    centered(d,(410,135),"Predicted normal",font(22)); centered(d,(710,135),"Predicted obstructed",font(22))
    d.text((45,235),"True normal",font=font(22),fill="#000000"); d.text((20,425),"True obstructed",font=font(22),fill="#000000")
    centered(d,(550,580),f"Coverage 75.0%   Sensitivity {100*sensitivity:.1f}%   Specificity {100*specificity:.1f}%",font(24,bold=True))
    left,top,right,bottom=145,690,1010,1025
    d.line((left,top,left,bottom),fill="#333333",width=3); d.line((left,bottom,right,bottom),fill="#333333",width=3)
    for value in np.linspace(.88,1.00,7):
        y=bottom-(value-.88)/.12*(bottom-top); d.line((left,y,right,y),fill="#E8E8E8",width=2); d.text((50,y-13),f"{value:.2f}",font=font(20),fill="#333333")
    d.text((left,630),"Cohen's kappa at matched 75% coverage",font=font(25,bold=True),fill="#000000")
    groups = [("Cycle E", e_metrics), ("Cycle G", g_metrics)]
    colors = ["#1F4E79", "#7F7F7F", "#BFBFBF"]
    labels = ["Physics-informed", "Nonphysics selector", "Ridge margin"]
    keys = ["physics_reliability", "ridge_reliability", "ridge_margin"]
    for group_index, (cycle, metrics) in enumerate(groups):
        center_x = left + 255 + group_index * 410
        centered(d,(center_x,bottom+34),cycle,font(23,bold=True))
        for bar_index, (key, color) in enumerate(zip(keys, colors)):
            value = metrics[key]["kappa"]
            x1 = center_x - 120 + bar_index * 80
            y = bottom - (value-.88)/.12*(bottom-top)
            d.rectangle((x1,y,x1+52,bottom),fill=color)
            centered(d,(x1+26,y-22),f"{value:.3f}",font(18,bold=True))
    for index, (label, color) in enumerate(zip(labels, colors)):
        x = left + index * 275
        d.rectangle((x,655,x+24,679),fill=color)
        d.text((x+34,651),label,font=font(19),fill="#000000")
    img.save(ASSETS/"final_policy.png",dpi=(300,300))


def configure_doc(doc: Document):
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11)
    sec.top_margin = Inches(0.65)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.72)
    sec.right_margin = Inches(0.72)
    sec.header_distance = Inches(0.25)
    sec.footer_distance = Inches(0.25)
    set_columns(sec, 1)
    sec.footer.is_linked_to_previous = False
    add_page_number(sec.footer.paragraphs[0])

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.0)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.0


def add_title_block(doc: Document):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run("Physics-Informed Selective Inference for Incomplete Spirometry and Airflow-Obstruction Screening"), size=24, bold=False)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    set_font(p.add_run("Mrityunjay Gupta"), size=11)


def add_abstract_block(doc: Document):

    abstract = (
        "Premature expiration during spirometry underestimates forced vital capacity (FVC), raises the forced expiratory volume in one second to FVC ratio, and can conceal airflow obstruction associated with chronic obstructive pulmonary disease. This study extends PredictX with a state-aware method that retains direct endpoints for complete manoeuvres and reconstructs missing FVC for incomplete manoeuvres using a bounded two-compartment volume model. Raw National Health and Nutrition Examination Survey waveforms were decoded at 100 Hz, aligned by back-extrapolated onset, and evaluated through controlled truncation, naturally incomplete recordings, and participant-disjoint cross-cycle scoring. A physics-informed reliability selector was trained on Cycle F using nested participant-separated cross-fitting. It combines ridge confidence with physics-ridge agreement, recovered-FVC discrepancy, fit quality, duration, and terminal-volume change, then classifies the 75% of cases with highest predicted reliability. At two seconds, physiological recovery increased obstruction detection by 40.7 percentage points across 892 obstructed curves. In Cycles E and G, the selector achieved sensitivity of 96.0% and 96.1%, specificity of 99.7% and 99.8%, and Cohen's kappa of 0.952 and 0.959. Kappa exceeded an identical nonphysics selector by 0.019 and 0.030 and ridge-margin selection by 0.045 and 0.042, with all paired bootstrap intervals excluding zero. A physics-plus-ridge endpoint model also reduced FVC error in both cycles, establishing measurable physics-derived value beyond ridge regression alone."
    )
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.15)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run("Abstract- ")
    set_font(r, size=9, bold=True, italic=True)
    set_font(p.add_run(abstract), size=9, italic=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.15)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    set_font(p.add_run("Key\u00a0Terms- "), size=9, bold=True, italic=True)
    set_font(p.add_run("censored time series, chronic obstructive pulmonary disease screening, inverse problems, physiological modeling, selective prediction, spirometry, uncertainty quantification"), size=9, italic=True)


def build():
    make_assets()
    baseline_result = json.loads((ROOT / "results" / "manuscript_baseline_comparison.json").read_text(encoding="utf-8"))
    baseline_metrics = baseline_result["metrics"]
    selector_result = json.loads((ROOT / "results" / "physics_informed_selector_cross_cycle.json").read_text(encoding="utf-8"))
    selector_cycles = selector_result["cross_cycle"]
    doc = Document()
    configure_doc(doc)
    add_title_block(doc)

    sec = doc.add_section(WD_SECTION.CONTINUOUS)
    sec.top_margin = Inches(0.65)
    sec.bottom_margin = Inches(0.65)
    sec.left_margin = Inches(0.72)
    sec.right_margin = Inches(0.72)
    set_columns(sec, 2, 340)
    sec.footer.is_linked_to_previous = True

    add_abstract_block(doc)
    add_heading(doc, "I  INTRODUCTION")
    add_body(doc, "This study extends PredictX by addressing a specific measurement challenge in spirometry-based airflow-obstruction screening: the expiratory recording ends before forced vital capacity (FVC) has been measured [17].")
    add_body(doc, "Spirometry identifies airflow obstruction from the ratio of forced expiratory volume in one second (FEV1) to FVC. A patient who stops exhaling early can still produce a usable FEV1, while the recorded FVC remains too small. The resulting FEV1/FVC ratio is therefore biased upward and may cross the 0.70 screening threshold from obstructed to apparently normal. This creates a predictable source of missed chronic obstructive pulmonary disease (COPD) screening cases.")
    add_body(doc, "The proposed method reconstructs the raw expiratory waveform, measures FEV1 from a corrected time origin, determines whether FVC was fully observed, and estimates the missing endpoint when expiration was incomplete. The estimator represents lung emptying with fast and slow physiological components and fits only the portion of the volume-time curve available before termination. Repeated manoeuvres, confidence intervals, and consistency rules determine whether the recovered evidence supports a screening classification or requires repeat testing.")
    add_body(doc, "The study separates model development from cross-cycle scoring. Artificial truncation and naturally incomplete Cycle F recordings informed the estimator and action policy. The final state-aware endpoint model and reliability selector were trained on Cycle F TRAIN with participant separation. Participant-level scoring in Cycles E and G compared the complete workflow with quality-control-first handling, FEV1/FEV6, direct-ratio selection, ridge regression, a same-capacity nonphysics selector, and a physics-plus-ridge endpoint model.")
    add_body(doc, "The central contribution is a learned reliability coordinate derived from the relation between two independently structured endpoint estimates. A constrained physiological inverse model supplies recovered FVC, fit quality, and disagreement with a participant-separated ridge estimate. Nested cross-fitting estimates how these quantities predict decision correctness, and a fixed coverage rule converts the resulting score into classification or deferral. This design makes the incremental value of the physics representation directly testable against an identical selector without physics features.")
    add_figure(doc, ASSETS / "architecture.png", "Fig. 1. State-aware system. Complete manoeuvres retain direct endpoints; incomplete manoeuvres use recovered FVC. Physics-informed reliability supports classification or deferral.", width=2.85)

    add_heading(doc, "II  RELATED WORK AND TECHNICAL GAP")
    add_heading(doc, "A  Clinical handling of incomplete expiration", level=2)
    add_body(doc, "The American Thoracic Society and European Respiratory Society (ATS/ERS) technical standard defines back-extrapolated time zero, end-of-forced-expiration acceptability, repeatability grading, and session-level reporting from the largest acceptable FEV1 and FVC [2]. A manoeuvre without an acceptable FVC endpoint normally triggers another effort or further assessment. This quality-control-first pathway is the principal clinical baseline in this study.")
    add_body(doc, "Forced expiratory volume in six seconds (FEV6) provides a validated alternative when six seconds of expiration are available. Population studies report high agreement between FEV1/FEV6 and FEV1/FVC, with a 0.73 fixed threshold providing a strong screening operating point [18], [19]. FEV1/FEV6 addresses shorter acceptable tests; recordings ending between two and five seconds define the earlier censoring regime studied here.")
    add_heading(doc, "B  Mathematical recovery of forced vital capacity", level=2)
    add_body(doc, "Earlier work fitted mono-, bi-, and tri-exponential functions to unfinished volume-time curves. Bi-exponential extrapolation captured heterogeneous lung emptying and produced nearly unbiased FVC estimates in small clinical cohorts [5], [23]. These studies established the feasibility of estimating terminal volume from incomplete expiration. The present analysis treats the observed duration as an explicit censoring time and evaluates the resulting endpoint estimate at the clinical decision boundary across three acquisition cycles.")
    add_heading(doc, "C  Learned inference from spirometry", level=2)
    add_body(doc, "Neural and statistical models have been applied to spirometry quality assessment, disease classification, structural phenotyping, and outcome prediction from suboptimal efforts [20]-[22]. We trained ridge regression on Cycle F TRAIN participants using observed FEV1, terminal observed volume, duration, age, height, and sex. A state-aware version used a ridge model trained only on incomplete Cycle F manoeuvres for incomplete endpoints and direct measurement for complete endpoints. A physics-plus-ridge endpoint model added recovered FVC to the same incomplete-state feature set. Regularization, imputation, scaling, and model coefficients were estimated with participant separation.")
    add_heading(doc, "D  Censoring-aware selective formulation", level=2)
    add_body(doc, "The technical objective is to infer latent FVC from a two-to-five-second prefix while preserving session quality control and quantifying when the available evidence supports action. The state-conditional estimator includes censoring time in the inverse objective and enforces monotonicity and total-volume conservation. The participant-level selector models the probability that the state-aware ridge decision is correct from ridge margin, measurement state, observation duration, terminal-volume change, physics margin, model disagreement, recovered-FVC discrepancy, and volume-fit quality. Its incremental value is evaluated at identical coverage against ridge-margin ranking and an ExtraTrees selector with the same capacity but no physics features.")

    add_heading(doc, "III  DATA AND STUDY DESIGN")
    add_heading(doc, "A  NHANES cycles and unit of analysis", level=2)
    add_body(doc, "The analysis used raw and summary spirometry from NHANES 2007-2008 (cycle E), 2009-2010 (cycle F), and 2011-2012 (cycle G). Raw rows represent individual attempted manoeuvres; participant-level analyses were keyed by the NHANES sequence identifier. One parser definition was applied to all cycles. The repository records 34,857, 36,421, and 37,661 raw rows in cycles E, F, and G, respectively, and 6,319, 7,079, and 6,652 participants with summary truth. The selected usable A/B-quality counts were 6,107, 6,905, and 6,447. This design measures replication across acquisition periods under a fixed computational definition.")
    add_table(doc, "TABLE I  NHANES COHORT ACCOUNTING", ["Cycle", "Years", "Raw rows", "Truth participants", "Usable A/B"], [
        ("E", "2007-08", "34,857", "6,319", "6,107"),
        ("F", "2009-10", "36,421", "7,079", "6,905"),
        ("G", "2011-12", "37,661", "6,652", "6,447"),
        ("Total", "2007-12", "108,939", "20,050", "19,459"),
    ], widths=[.35,.55,.66,.82,.68], font_size=6.9)
    add_heading(doc, "B  Reference endpoints and clinical labels", level=2)
    add_body(doc, "Parser validation used the released NHANES best-test summary as the reference and compared it with the maximum acceptable pre-bronchodilator FEV1 and FVC reconstructed from raw curves. Artificial-censoring experiments used each participant's selected acceptable A-quality complete-plateau curve as within-curve truth. Completeness required forced expiratory time of at least 6 s and no more than 0.025 L of additional volume in the terminal second. Natural-pair evaluation matched a separate pre-bronchodilator manoeuvre carrying a quality or plateau flag to the same participant's clean target, then selected the shortest usable candidate to preserve one independent pair per participant.")
    add_body(doc, "The principal binary endpoint was fixed-ratio airflow obstruction, FEV1/FVC < 0.70, matching the repository definition and the GOLD diagnostic ratio threshold. Global Lung Function Initiative 2012 equations supplied secondary age-, height-, sex-, and population-referenced checks [9]. Experimental cohorts were constructed for paired error analysis under censoring through within-participant and within-curve comparisons. Population-prevalence estimates require survey-weighted epidemiologic sampling.")
    add_body(doc, "Acquisition acceptability and reconstruction eligibility are recorded as separate variables. NHANES SPAACC, SPAQEFF, and SPAPLAT fields retain the released technical assessment of each manoeuvre. Reconstruction eligibility is assigned from the decoded duration, terminal-second volume, parse status, model-fit status, and repeated-effort evidence. A reconstructed endpoint therefore contributes a new evidence source while the original acceptability label remains unchanged.")
    add_heading(doc, "C  Evaluation hierarchy and label use", level=2)
    add_body(doc, "Artificial truncation hides the complete endpoint from the estimator while retaining it as ground truth. Natural-pair experiments compare separate manoeuvres from the same Cycle F examination. Cycle F supplied policy-development evidence and all training data for the endpoint and reliability models. Cycles E and G contain disjoint participants and supply cross-cycle transport evidence. Their labels do not enter model fitting, feature imputation, regularization, reliability training, or coverage selection.")
    add_wide_figure(doc, ASSETS / "cohort_flow.png", "Fig. 2. Cohort construction, demographic composition, severity distribution, and verified experiment overlap. Severity entries are severe / moderate / mild obstruction / normal. Ages are median [interquartile range]. Controlled truncation used 1,492 complete-plateau curves, of which 892 were obstructed; natural pairs, policy development, and frozen cross-cycle scoring are distinct validation units.", width=6.75)

    add_heading(doc, "IV  SIGNAL RECONSTRUCTION")
    add_heading(doc, "A  Raw waveform decoding", level=2)
    add_body(doc, "Each SPXRAW record is a comma-delimited sequence of signed integer volume increments sampled at 0.01-s intervals. The reconstruction indexes each increment by sample and applies the record-specific SPAFACT conversion factor. Flow and cumulative volume follow from")
    add_equation(doc, ASSETS / "raw_decode_equations.png", width=2.05)
    add_body(doc, "Conversion precedes endpoint derivation, and all subsequent fitting uses liters and seconds. Direct decoding of the released increments preserves the acquisition semantics and avoids noise amplification from numerical differentiation.")
    add_heading(doc, "B  Forced-expiration segmentation and time zero", level=2)
    add_body(doc, "A five-sample moving average stabilizes the flow series for segmentation. Peak expiratory flow (PEF) is identified as the maximum of the smoothed signal. Baseline location and scale are estimated robustly from pre-PEF samples using the median and median absolute deviation. The onset threshold is the maximum of 0.05 L/s, baseline plus five robust standard deviations, and 2% of PEF; onset requires four of five consecutive samples above threshold. The expiratory endpoint is the sample preceding sustained negative flow that marks the following inspiration. Median baseline flow is removed before volume integration.")
    add_body(doc, "Back-extrapolated time zero is obtained from the line connecting detected onset to peak expiratory flow. The equations below define peak-flow time and volume, onset volume, peak flow, the corrected first-second volume, and the direct endpoint volume.")
    add_equation(doc, ASSETS / "time_zero_equations.png", width=2.55)
    add_body(doc, "Synthetic verification over 200 generated recordings recovered onset within two samples and FEV1 within 2% in every case.")
    add_heading(doc, "C  Parser fidelity across cycles", level=2)
    add_body(doc, "The reconstructed best-test endpoints agreed closely with the NHANES summaries. Median absolute FEV1 error was 3.143%, 2.908%, and 2.948% in cycles E, F, and G; median absolute FVC error was 3.716%, 3.527%, and 3.611%. Comparable magnitudes across the three cycles support parser transportability.")
    add_table(doc, "TABLE II  RAW PARSER CONCORDANCE WITH NHANES BEST TEST", ["Cycle", "n", "FEV1 MAE", "FVC MAE", "Parse errors"], [
        ("E", "6,319", "3.143%", "3.716%", "26"),
        ("F", "7,079", "2.908%", "3.527%", "45"),
        ("G", "6,652", "2.948%", "3.611%", "60"),
    ], widths=[.38,.64,.67,.67,.70], font_size=6.9)

    add_heading(doc, "V  PHYSIOLOGY-CONSTRAINED CENSORED-CURVE MODEL")
    add_heading(doc, "A  Forward model", level=2)
    add_body(doc, "Latent expiratory flow is represented by fast and slow emptying compartments multiplied by a finite-rise term. The parameters τf and τs are the fast and slow time constants, α is the fast-compartment fraction, FVC is total expired volume, and τo controls onset rise. Numerical normalization sets the integrated flow equal to FVC. Interval-aware trapezoidal integration then produces cumulative volume on the inference grid and preserves total-volume conservation under nonuniform sampling.")
    add_figure(doc, ASSETS / "model_equations.png", "Fig. 3. Volume-conserving forward model and censored inverse estimator. The objective uses observations through censoring time τ and returns recovered FVC and the corresponding FEV1/FVC ratio.", width=3.08)
    add_heading(doc, "B  Constrained inverse estimator", level=2)
    add_body(doc, "For a curve observed through censoring time τ, the estimator minimizes noise-standardized volume residuals over the observed prefix. The parameter vector and feasible region are")
    add_equation(doc, ASSETS / "parameter_bounds.png", width=2.85)
    add_body(doc, "Optimization uses bounded trust-region reflective least squares with a three-point numerical Jacobian and a 600-evaluation budget. The residual scale is estimated from Savitzky-Golay-smoothed data with a finite dispersion-based fallback. Recovered FVC is the modeled terminal volume; FEV1 remains anchored to the observed first second, and the recovered ratio is observed FEV1 divided by recovered FVC.")
    add_body(doc, "The five parameters define the family of continuations permitted after the observed prefix. The optimizer returns the terminal FVC implied by the best-fitting continuation. Fisher-information analysis characterizes uncertainty in the fitted parameters, while endpoint error measures the quantity used in the screening ratio.")
    add_figure(doc, ASSETS / "censoring_schematic.png", "Fig. 4. Schematic of endpoint censoring. Direct interpretation freezes FVC at the observed endpoint. The constrained model estimates continuation toward the latent plateau.", width=3.12)
    add_heading(doc, "C  Estimator routing by measurement state", level=2)
    add_body(doc, "The evaluated implementation routes endpoints by measurement state. Complete manoeuvres retain directly measured FEV1, FVC, and FEV1/FVC. Incomplete manoeuvres retain measured FEV1 and use the physiology-constrained FVC estimate; the Cycle F ratio correction is applied only to this recovered branch. The workflow then integrates endpoint state, repeated manoeuvres, and the learned reliability score into a participant-level action.")

    add_heading(doc, "VI  MULTI-MANOEUVRE SELECTIVE INFERENCE")
    add_heading(doc, "A  Session-level best-test evidence", level=2)
    add_body(doc, "For each participant, session-level FEV1 is the largest acceptable FEV1 and session-level FVC is the largest acceptable FVC, consistent with ATS/ERS reporting [2]. The two maxima may arise from different efforts.")
    add_equation(doc, ASSETS / "session_equations.png", width=2.75)
    add_body(doc, "Repeatability grades A–F are computed from the differences between the two largest FEV1 values and the two largest FVC values. The session ratio is formed after endpoint correction on a common calibrated scale.")
    add_heading(doc, "B  Reliability states and calibrated deferral", level=2)
    add_body(doc, "The reliability layer represents four evidence states. R1 denotes truncated or incomplete expiration. R2 denotes a ratio interval that intersects the 0.70 boundary. R3 denotes a cross-manoeuvre consistency review. R4 denotes a signal-artifact review. R1 is re-evaluated through structural endpoint recovery or an alternate acceptable manoeuvre. R2 requires the complete bootstrap interval to clear the decision boundary. R3 uses ATS best-test evidence and a cross-manoeuvre consistency rule. R4 is finalized when an independent acceptable signal supplies sufficient evidence.")
    add_heading(doc, "C  Selective decision objective", level=2)
    add_body(doc, "The obstruction classifier and binary selection function jointly determine finalized predictions. Coverage is the probability of selection, and selective risk is expected loss conditional on selection.")
    add_equation(doc, ASSETS / "selective_equations.png", width=2.85)
    add_body(doc, "The final selector estimates decision reliability rather than disease probability. Its base decision is the state-aware ridge ratio at the 0.70 boundary. An ExtraTrees classifier with 500 trees, minimum leaf size five, balanced class weights, and 70% feature sampling predicts whether that decision is correct. The operating point maximizes nested out-of-fold Cycle F kappa subject to at least 75% coverage, which selected a 75% deployment coverage.")
    add_body(doc, "Five participant-hash outer folds separate reliability evaluation. Within each outer development set, inner participant folds generate out-of-fold ridge ratios and physics corrections for reliability training; the outer fold supplies the reliability prediction. The final selector is then fitted to cross-fitted Cycle F TRAIN features. Outcome labels from Cycles E and G enter only after selection to compute coverage, sensitivity, specificity, Cohen's kappa, and severity-resolved error.")

    add_heading(doc, "VII  EXPERIMENTAL PROTOCOL")
    add_heading(doc, "A  Artificial truncation", level=2)
    add_body(doc, "Complete-plateau curves were censored at 2, 3, 4, and 5 s. The estimator received the observed prefix, and the complete curve supplied hidden truth. Up to 200 participants per severity stratum were selected within each cycle. Direct calculation from the truncated endpoint quantified the censoring-induced ratio shift, while the quality-control-first comparator represented standard repeat-measurement handling. Smoothing spline, linear-tail, mono-exponential, two-compartment, and hybrid structural estimators measured alternative recovery behavior.")
    add_heading(doc, "B  Naturally incomplete recordings", level=2)
    add_body(doc, "The field-realistic experiment paired each clean participant target with a separate pre-bronchodilator manoeuvre identified by the released quality or plateau fields. The shortest usable candidate per participant was selected, and its duration determined the censoring endpoint. The cohort contained 286 participants: 63 at ≤2 s, 42 at 2–3 s, 42 at 3–4 s, 20 at 4–5 s, and 119 above 5 s. These pairs represent naturally occurring incomplete-acquisition states.")
    add_heading(doc, "C  Severe-obstruction power analysis", level=2)
    add_body(doc, "Severe obstruction was examined as a prespecified high-priority stratum. Pooling cycles E, F, and G yielded 46 participants with paired naturally incomplete severe recordings and 67 severe complete-plateau curves for artificial truncation. Exact two-sided Clopper-Pearson 95% intervals quantified event-rate precision.")
    add_heading(doc, "D  Robustness and subgroup replication", level=2)
    add_body(doc, "Device stress tests perturbed in-memory curve copies using additive flow noise, lower sampling rate, gain error, and low-frequency baseline drift. The realistic combined profile used 2% of PEF additive noise, 30-Hz sampling, +2% gain, and net drift equal to 0.5% of FVC. The locked parser, optimizer, parameter bounds, and classifier were applied to every perturbation. Within-program subgroup analyses stratified pooled obstructed curves by smoking status, self-reported emphysema or chronic bronchitis, and age band. Replication required positive paired margins with bootstrap confidence intervals above zero.")
    add_heading(doc, "E  Clinical and computational baselines", level=2)
    add_body(doc, "The quality-control-first baseline deferred every ATS-truncated manoeuvre and classified the remaining cases from the directly measured ratio. The FEV1/FEV6 baseline used the released six-second volume and the published 0.73 threshold [18], [19]. A direct-ratio confidence baseline ranked cases by absolute distance from 0.70 and retained the same number of cases as the final policy.")
    add_body(doc, "The archived ridge baseline predicted released summary FVC from observed FEV1, terminal observed volume, duration, age, height, and sex. Five deterministic participant folds within Cycle F TRAIN selected regularization; imputation and scaling were estimated within fitting folds. Cycle F TRAIN and TEST contained 562 and 563 unique participants with zero overlap. A second ridge model was trained only on incomplete Cycle F TRAIN manoeuvres and routed by endpoint state. The same-capacity reliability ablation used the final ExtraTrees configuration with ridge margin, duration, terminal-volume change, and truncation state while excluding all physics features.")
    add_heading(doc, "F  Statistical analysis", level=2)
    add_body(doc, "False-negative rate, specificity, and agreement beyond chance were defined as")
    add_equation(doc, ASSETS / "metrics_equations.png", width=2.65)
    add_body(doc, "All Cycle E/G intervals used 2,000 participant bootstrap replicates. Paired replicates estimated differences in coverage, sensitivity, specificity, Cohen's kappa, and FVC mean absolute error. Reliability methods retained the same number of participants at the Cycle-F-selected 75% coverage. Matched-coverage comparisons therefore isolate selection quality from abstention rate. Binomial intervals were reported for proportions, and zero-event severe analyses used exact intervals.")

    add_heading(doc, "VIII  RESULTS")
    add_heading(doc, "A  Three-cycle artificial-censoring replication", level=2)
    add_body(doc, "Across 892 obstructed complete curves, physiology-constrained recovery increased obstruction detection after 2-s truncation from 1.7% to 42.4%, a paired gain of 40.7 percentage points with a bootstrap 95% confidence interval of 37.3-43.8 points. At 3 s, detection increased from 14.9% to 42.0%, a 27.1-point gain. Every cycle-specific paired interval excluded zero.")
    add_body(doc, "This experiment isolates endpoint-recovery behavior under known censoring. The recovery gain replicated independently in Cycles E, F, and G and was largest when the observed endpoint contained the least plateau information. All severe cases were detected across the pooled artificial-censoring experiment.")
    add_figure(doc, ASSETS / "pooled_truncation.png", "Fig. 5. Obstruction detection under pooled three-cycle artificial truncation among 892 obstructed complete curves. Physiology-constrained recovery restores information from the unobserved terminal volume.", width=3.12)
    add_table(doc, "TABLE III  THREE-CYCLE OBSTRUCTION-DETECTION REPLICATION", ["Cycle", "T", "Truncated", "Recovery", "Gain"], [
        ("E", "2 s", ".010", ".389", ".379"), ("E", "3 s", ".113", ".385", ".272"),
        ("F", "2 s", ".013", ".407", ".394"), ("F", "3 s", ".152", ".370", ".219"),
        ("G", "2 s", ".027", ".476", ".449"), ("G", "3 s", ".184", ".507", ".323"),
        ("Pooled", "2 s", ".017", ".424", ".407"), ("Pooled", "3 s", ".149", ".420", ".271"),
    ], widths=[.52,.42,.58,.66,.68], font_size=6.7)
    add_heading(doc, "B  Naturally incomplete field recordings", level=2)
    add_body(doc, "The 286 natural pairs contained 178 obstructed and 108 normal participants. Physiology-constrained reconstruction identified 166 of 178 obstructed participants, corresponding to 93.3% sensitivity. The detection gain was consistent across observed-duration strata and reached 92.9% among recordings ending within two seconds.")
    add_body(doc, "The natural-pair experiment demonstrates recovery on separate field-recorded manoeuvres from the same examination. Boundary clearance and repeated-manoeuvre evidence subsequently convert the recovered endpoint into a selective participant-level action.")
    add_figure(doc, ASSETS / "field_recovery.png", "Fig. 6. Obstruction detection in naturally incomplete same-participant recordings by observed duration. Recovery provides the largest detection gains under the strongest endpoint censoring.", width=3.18)
    add_body(doc, "The representative reconstructions show the estimator operating on real decoded waveforms across severe, moderate, and mild obstruction. The fitted continuation begins from the observed prefix and converges toward the complete reference endpoint while preserving a monotone volume trajectory and the correct obstruction classification.")
    add_heading(doc, "C  Severity-resolved performance", level=2)
    add_body(doc, "Physiology-constrained reconstruction correctly identified all 46 participants with naturally incomplete severe recordings. The pooled artificial cohort contained 67 severe complete curves, and every severe case was identified after truncation at 2, 3, 4, and 5 s. This severity-resolved result replicated across natural and controlled censoring experiments.")
    add_table(doc, "TABLE IV  POOLED SEVERE-OBSTRUCTION IDENTIFICATION", ["Experiment", "n", "Truncated detected", "Recovery detected"], [
        ("Natural incomplete", "46", "39 / 46", "46 / 46"),
        ("Artificial 2 s", "67", "-", "67 / 67"),
        ("Artificial 3 s", "67", "-", "67 / 67"),
        ("Artificial 4 s", "67", "-", "67 / 67"),
        ("Artificial 5 s", "67", "-", "67 / 67"),
    ], widths=[1.12,.42,.92,.92], font_size=6.8)
    add_wide_figure(doc, ASSETS / "real_reconstruction_examples.png", "Fig. 7. Representative Cycle F NHANES reconstructions. Within each prespecified severity-duration stratum, the displayed curve is closest to the stratum median absolute FVC error among correctly classified, non-bound-active fits. Each panel shows the complete decoded volume-time curve, the observed prefix, and the fitted volume-conserving continuation.", width=6.75)
    add_wide_figure(doc, ASSETS / "patient_benefit.png", "Fig. 8. Patient-level benefit map. A-C: controlled-truncation detection across observation duration, baseline severity, and distance below the 0.70 boundary. D: two-second detection by complete-curve FVC magnitude. E: frozen Cycle E/G coverage and conditional accuracy by available manoeuvres. F: participant-specific interval width versus distance from 0.70; every interval crossing the boundary was deferred.", width=6.75)
    add_body(doc, "The benefit map localizes the recovery effect. Severe obstruction was detected at every controlled duration, moderate-case detection rose from 72.5% at 2 s to 97.5% at 5 s, and detection increased as the complete-curve ratio moved farther below 0.70. In frozen Cycle E/G scoring, three or more fitted manoeuvres supported 86.5-88.1% coverage with 97.1-97.8% conditional accuracy. All 94 participant intervals that crossed the 0.70 boundary were deferred.")
    add_heading(doc, "D  Final selective workflow", level=2)
    add_body(doc, "The Cycle F policy-development partition contained 563 cases. The final ATS best-test plus residual-incomplete policy classified 464 cases (82.4% coverage), with 233 true positives and 217 true negatives. Conditional sensitivity was 96.3%, specificity was 97.7%, and Cohen's kappa was 0.940. All seven severe cases in the finalized subset were classified correctly.")
    add_body(doc, "The proposed policy combines physiological endpoint recovery with prespecified session-level evidence, producing an explicit action trace for every finalized or deferred case. Ninety-nine cases were routed to repeat testing, referral, or indeterminate handling through the boundary, signal-quality, and repeated-manoeuvre rules.")

    add_heading(doc, "E  Cycle E and G cross-cycle evaluation", level=2)
    add_body(doc, "Cycle E contributed 6,026 eligible participants and Cycle G contributed 6,408. State-aware endpoint routing finalized 5,405 and 5,712 cases, corresponding to 89.7% and 89.1% coverage. Sensitivity was 94.9% and 97.3%, specificity was 97.3% and 97.4%, and kappa was 0.839 and 0.836. These measurements establish the transport behavior of the corrected direct-versus-recovered endpoint rule.")
    add_body(doc, "The physics-informed reliability selector retained 4,520 Cycle E and 4,806 Cycle G participants at the Cycle-F-selected 75% coverage. Sensitivity was 96.0% and 96.1%, specificity was 99.7% and 99.8%, and kappa was 0.952 and 0.959. Relative to the same-capacity nonphysics selector, kappa increased by 0.019 (95% CI 0.002 to 0.036) and 0.030 (0.013 to 0.050). Relative to ridge-margin selection, the gains were 0.045 (0.024 to 0.066) and 0.042 (0.020 to 0.066). All 25 Cycle E and 36 Cycle G severe cases selected by the reliability rule were classified correctly.")
    add_figure(doc, ASSETS / "final_policy.png", "Fig. 9. Cycle E/G physics-informed reliability results. The confusion matrix combines 9,326 selected participants. The lower panel compares Cohen's kappa with the identical nonphysics selector and ridge-margin selection at matched 75% coverage.", width=3.10)
    add_table(doc, "TABLE V  CROSS-CYCLE SELECTIVE PERFORMANCE", ["Cycle", "Method", "Coverage", "Sensitivity", "Specificity", "Kappa"], [
        (
            cycle,
            label,
            f"{selector_cycles[cycle]['metrics'][key]['coverage']:.3f}",
            f"{selector_cycles[cycle]['metrics'][key]['sensitivity']:.3f}",
            f"{selector_cycles[cycle]['metrics'][key]['specificity']:.3f}",
            f"{selector_cycles[cycle]['metrics'][key]['kappa']:.3f}",
        )
        for cycle in ("E", "G")
        for label, key in (
            ("Physics selector", "physics_reliability"),
            ("Nonphysics selector", "ridge_reliability"),
            ("Ridge margin", "ridge_margin"),
        )
    ], widths=[.38,.95,.58,.68,.68,.52], font_size=6.2)
    add_body(doc, "The physics-plus-ridge endpoint model reduced state-aware FVC mean absolute error from 184.0 to 182.8 mL in Cycle E and from 181.7 to 180.8 mL in Cycle G. Paired reductions were 1.22 mL (95% CI 0.49 to 1.92) and 0.96 mL (0.25 to 1.72), establishing incremental endpoint information beyond the equal-input ridge estimate.")

    add_heading(doc, "IX  ABLATION AND MODEL-SELECTION EVIDENCE")
    add_heading(doc, "A  Complete versus censored estimands", level=2)
    add_body(doc, "The state-specific ablation established the routing principle used by the research design. Direct measurement reached kappa 0.918 on complete Cycle F manoeuvres. For censored manoeuvres, physiology-constrained recovery increased obstruction detection after 2-s truncation from 1.3% to 40.7%. These complementary results support direct endpoints for observed plateaus and structural recovery for latent endpoints.")
    add_heading(doc, "B  Comparator behavior", level=2)
    add_body(doc, "The two-compartment estimator supplied the strongest high-specificity detection gain among the bounded structural candidates. At 2 and 3 s, it achieved 41.1% and 40.4% obstruction detection with 96.5% and 97.0% specificity, and every optimization returned a finite endpoint. Its fast and slow emptying components therefore provided the selected structural family for participant-level inference.")
    add_table(doc, "TABLE VI  SELECTED STRUCTURAL ESTIMATOR PERFORMANCE", ["Estimator", "T", "Detection", "Specificity", "Finite fits"], [
        ("Two-compartment", "2 s", ".411", ".965", "497 / 497"),
        ("Two-compartment", "3 s", ".404", ".970", "497 / 497"),
    ], widths=[1.20,.46,.70,.70,.76], font_size=6.8)
    add_heading(doc, "C  FVC accuracy across observed duration", level=2)
    add_body(doc, "Cycle-F clean-cut curves supplied direct endpoint-accuracy measurements across four observed durations. Median absolute FVC accuracy improved continuously as plateau information accumulated, from 330 mL at 2 s to 206 mL at 5 s. The 95th-percentile absolute difference improved from 836 mL to 621 mL over the same interval. All six bound-active severe curves at 2 s remained correctly identified as obstructed.")
    add_table(doc, "TABLE VII  FVC ACCURACY BY OBSERVED DURATION", ["T", "Median abs.", "Improvement from 2 s", "95th abs."], [
        ("2 s", "330 mL", "0 mL", "836 mL"),
        ("3 s", "292 mL", "38 mL", "778 mL"),
        ("4 s", "249 mL", "81 mL", "700 mL"),
        ("5 s", "206 mL", "124 mL", "621 mL"),
    ], widths=[.48,.84,1.18,.82], font_size=6.7)
    add_heading(doc, "D  Endpoint-specific validation", level=2)
    add_body(doc, "The model was optimized and validated for cumulative-volume endpoint inference under censoring. Its volume-conservation and monotonicity invariants directly govern the terminal quantity used in the screening ratio, while selective rules translate endpoint evidence into participant-level action.")
    add_heading(doc, "E  Incremental physics ablation", level=2)
    add_body(doc, "The endpoint and reliability ablations test two distinct contributions. Adding recovered FVC to the state-aware ridge endpoint model reduced FVC error in both cycles. Adding physics margin, model disagreement, FVC discrepancy, and fit quality to an otherwise identical reliability model increased kappa in both cycles, with paired confidence intervals above zero. The cross-cycle gain therefore follows from measured information supplied by the constrained trajectory model rather than additional classifier capacity.")

    add_heading(doc, "X  ROBUSTNESS AND WITHIN-PROGRAM TRANSPORT")
    add_heading(doc, "A  Portable-device degradation", level=2)
    add_body(doc, "The realistic combined degradation profile preserved the principal recovery margins. At 2 s, obstruction detection increased from 1.8% to 41.7%, a gain of 39.9 points with confidence interval 36.9-43.0. At 3 s, detection increased from 15.1% to 42.2%, a gain of 27.0 points with confidence interval 24.2-30.2. Recovery identified all 46 severe natural-pair participants under the combined profile. Independent sweeps through 10% PEF noise, 10-Hz sampling, +/-10% gain, and 2% FVC drift also preserved positive paired margins at 2 and 3 s.")
    add_body(doc, "A separate break-point analysis mapped numerical sensitivity across progressively larger perturbations. The realistic combined profile in Table VIII represents the deployment-oriented operating range used for the primary robustness claim.")
    add_table(doc, "TABLE VIII  COMBINED DEVICE-BUDGET PROFILE", ["T", "Truncated detect.", "Recovery detect.", "Gain", "95% CI"], [
        ("2 s", ".018", ".417", ".399", ".369-.430"),
        ("3 s", ".151", ".422", ".270", ".242-.302"),
    ], widths=[.45,.69,.78,.68,.85], font_size=6.9)
    add_heading(doc, "B  Clinical subgroup replication", level=2)
    add_body(doc, "Recovery advantages remained positive across current, former, never, and unknown smoking strata; across age bands below 40, 40-59, and 60-79 years; and across present, absent, or missing self-reported emphysema or chronic bronchitis status. At 2 s, direct-minus-recovery differences ranged from 0.339 to 0.741 across smoking strata and from 0.349 to 0.475 across age bands, with all reported intervals excluding zero. In the self-reported obstructive-disease stratum, the 2-s margin was 0.508 and the 3-s margin 0.325. Under the combined device profile, those margins were 0.524 and 0.286. Consistency across these strata supports transport of the censoring-recovery mechanism beyond a single demographic subgroup.")

    add_heading(doc, "XI  DISCUSSION")
    add_heading(doc, "A  Scientific contribution", level=2)
    add_body(doc, "The study formulates premature expiration as right censoring of a physiological trajectory whose latent terminal volume directly controls the obstruction ratio. Censoring time enters the inverse objective, numerical normalization enforces total-volume conservation, and the recovered endpoint is evaluated through its induced decision at the 0.70 boundary. The incomplete trace is therefore represented by a latent endpoint and an observed censoring time.")
    add_body(doc, "The participant-level contribution is a selective action function over measurement state and repeated efforts. It records whether FVC was observed or recovered, the manoeuvres supporting session maxima, uncertainty relative to the decision boundary, and the reason for classification, repeat testing, or referral. The same risk-coverage surface supports direct comparison with mechanistic extrapolators, clinical quality control, FEV1/FEV6, and learned estimators.")
    add_heading(doc, "B  Meaning of the performance gains", level=2)
    add_body(doc, "Artificial truncation established a reproducible recovery effect across three survey cycles. Recovery increased obstruction detection by 40.7 percentage points at 2 s and 27.1 points at 3 s, with every cycle-specific paired interval excluding zero. Severe cases were identified throughout the pooled artificial experiment.")
    add_body(doc, "Natural pairs extended the recovery effect to separate field-recorded manoeuvres, reaching 93.3% sensitivity. The final selector combined boundary clearance and consistent session evidence to achieve 96.3% sensitivity, 97.7% specificity, and kappa of 0.940 on the 82.4% classified Cycle F subset.")
    add_body(doc, "Cross-cycle evaluation established the contribution of physics-informed reliability modeling. At equal 75% coverage, the final selector produced kappa of 0.952 and 0.959. Its improvement over both ridge-margin ranking and an identical nonphysics reliability model was statistically positive in Cycles E and G. The physics representation therefore contributes predictive information beyond ridge confidence, duration, terminal-volume change, and measurement state.")
    add_heading(doc, "C  Interdisciplinary significance", level=2)
    add_body(doc, "The method integrates respiratory physiology, biomedical signal processing, numerical optimization, statistical learning, and pulmonary-function standards. Physiology defines the latent trajectory and volume constraints. Signal processing reconstructs the encoded waveform and locates onset. Numerical optimization solves the censored inverse problem. Statistical learning supplies selective-risk analysis, bootstrap inference, calibration, and subgroup validation. Clinical standards determine session-level evidence aggregation and the cost ordering of clinical errors. Together, these constraints produce an auditable path from sample acquisition to clinical action.")

    add_heading(doc, "XII  LIMITATIONS")
    add_body(doc, "The reported endpoint is pre-bronchodilator airflow-obstruction screening. COPD diagnosis requires post-bronchodilator spirometry together with symptoms, exposure history, and clinical assessment [1], [10]. The present results do not establish diagnostic performance, prospective workflow benefit, or cross-device clinical validity.")
    add_body(doc, "Cycle F informed the action policy, ridge fits, reliability model, and operating-point selection; its 96.3% sensitivity, 97.7% specificity, and kappa of 0.940 are development estimates. Cycles E and G are participant-disjoint and their labels were excluded from fitting, but they remain within NHANES, share acquisition semantics and released endpoint construction, and had supported earlier workflow audits. They provide locked post-audit cross-cycle re-evaluation rather than independent external clinical validation.")
    add_body(doc, "The archived ridge experiment has no participant overlap and uses no TEST labels for coefficients or regularization. Its full-manoeuvre terminal-volume feature correlates 0.993 with released summary FVC and is a near-target proxy that would not be available at the intended censored decision time. The state-aware incomplete-manoeuvre ridge is therefore the relevant equal-information comparator.")
    add_body(doc, "Component-level evaluation distinguishes the final system from exclusive use of the inverse estimate. The standalone physics endpoint had higher error than the incomplete-state ridge endpoint, while the Cycle-F-trained physics-plus-ridge endpoint model achieved statistically lower error than state-aware ridge in both cycles. The final architecture therefore assigns the physics model an incremental endpoint and reliability role supported by separate ablations.")
    add_body(doc, "Raw reconstruction in the natural-pair cohort increased sensitivity to 93.3% while specificity decreased to 65.7%; below two seconds, specificity was 31.4%. The workflow addresses this operating-point shift through selective deferral. Deferred Cycle E/G groups contained 135 and 120 obstructed participants, including 18 and 20 severe cases, so deferral requires clinically defined repeat-testing and referral pathways.")
    add_body(doc, "The physics-informed selector exceeded direct-ratio margin ranking numerically by 0.008 kappa in Cycle E and 0.005 in Cycle G; the paired intervals included zero. The present superiority claim is therefore specific to ridge-based selection. Broader superiority requires an independently collected cohort with device, operator, bronchodilator, and clinical-reference variation.")
    add_body(doc, "Parameter-bound sensitivity was concentrated under the simultaneous expansion of all five feasible ranges. Relative to the nominal configuration, this all-parameter expansion changed 15.1%, 8.2%, 6.0%, and 5.2% of controlled classifications at 2, 3, 4, and 5 s, respectively. At 2 s, its 95th-percentile absolute FVC shift was 816 mL. Expansion of the FVC bound alone changed 4.8%, 4.2%, 3.2%, and 2.4% of classifications. These results make prespecified bounds and observation duration part of the estimator definition.")
    add_body(doc, "The eight-start optimization audit found materially lower objective values than the nominal start in 5 of 200 curves. The severity-balanced profile analysis produced a median log10 information-matrix condition number of 4.73 at 2 s, improving to 3.67 at 5 s. Early prefixes therefore retain substantial parameter sloppiness even when the recovered endpoint is numerically stable, and prospective software should record active bounds, multistart concordance, and functional profile intervals.")
    add_body(doc, "The participant-specific R2 interval is an empirical repeated-manoeuvre interval rather than a calibrated population confidence interval. In the available 598-participant audit, its empirical coverage of the reference ratio was 55.0%. All 94 intervals intersecting 0.70 were deferred, demonstrating the intended routing behavior, while the observed coverage requires recalibration before the interval can be interpreted probabilistically.")
    add_body(doc, "The zero-event severe results have limited denominators. Zero false negatives among 46 natural severe recordings corresponds to an exact two-sided 95% upper bound of 7.7%, and zero errors among seven severe finalized Cycle F cases corresponds to an upper bound of 41.0%. These intervals preclude a zero-risk interpretation.")
    add_table(doc, "TABLE X  CYCLE-F INTERNAL BASELINE COMPARISON", ["Method", "Coverage", "Sensitivity", "Specificity", "Kappa"], [
        ("QC-first direct", f"{baseline_metrics['quality_control_first']['coverage']:.3f}", f"{baseline_metrics['quality_control_first']['sensitivity']:.3f}", f"{baseline_metrics['quality_control_first']['specificity']:.3f}", f"{baseline_metrics['quality_control_first']['kappa']:.3f}"),
        ("FEV1/FEV6", f"{baseline_metrics['fev1_fev6']['coverage']:.3f}", f"{baseline_metrics['fev1_fev6']['sensitivity']:.3f}", f"{baseline_metrics['fev1_fev6']['specificity']:.3f}", f"{baseline_metrics['fev1_fev6']['kappa']:.3f}"),
        ("Direct margin", f"{baseline_metrics['direct_ratio_matched_coverage']['coverage']:.3f}", f"{baseline_metrics['direct_ratio_matched_coverage']['sensitivity']:.3f}", f"{baseline_metrics['direct_ratio_matched_coverage']['specificity']:.3f}", f"{baseline_metrics['direct_ratio_matched_coverage']['kappa']:.3f}"),
        ("Ridge all", f"{baseline_metrics['ridge_all']['coverage']:.3f}", f"{baseline_metrics['ridge_all']['sensitivity']:.3f}", f"{baseline_metrics['ridge_all']['specificity']:.3f}", f"{baseline_metrics['ridge_all']['kappa']:.3f}"),
        ("Ridge matched", f"{baseline_metrics['ridge_matched_coverage']['coverage']:.3f}", f"{baseline_metrics['ridge_matched_coverage']['sensitivity']:.3f}", f"{baseline_metrics['ridge_matched_coverage']['specificity']:.3f}", f"{baseline_metrics['ridge_matched_coverage']['kappa']:.3f}"),
        ("Proposed policy", f"{baseline_metrics['final_selective_policy']['coverage']:.3f}", f"{baseline_metrics['final_selective_policy']['sensitivity']:.3f}", f"{baseline_metrics['final_selective_policy']['specificity']:.3f}", f"{baseline_metrics['final_selective_policy']['kappa']:.3f}"),
    ], widths=[1.08,.60,.70,.70,.55], font_size=6.4)

    add_heading(doc, "XIII  DEPLOYMENT CONSIDERATIONS")
    add_body(doc, "The five-parameter inverse problem and fixed optimization budget are compatible with portable screening workflows. Required inputs comprise the reconstructed volume trajectory, acquisition factor, timestamps or sampling rate, and per-manoeuvre quality metadata. A production implementation should record observation duration, optimizer status, recovered FVC, direct and recovered ratios, uncertainty interval, manoeuvre-level evidence, repeatability grade, and final action. These records support review and calibration monitoring across devices.")
    add_body(doc, "The deployment sequence comprises measurement reconstruction, quality-state inference, endpoint routing, repeated-manoeuvre aggregation, selective decision, and clinician-facing explanation. The output should state the FVC source, manoeuvres supplying session-level maxima, uncertainty relative to the 0.70 boundary, and any deferral reason. These records support telemedicine and community screening when pulmonary-function expertise and repeat visits are limited.")
    add_body(doc, "Recovery margins persisted under a realistic low-cost-device budget comprising 30-Hz sampling, 2% PEF noise, +2% gain, and 0.5% FVC drift. Device-specific calibration, flow-standard verification, prospective monitoring, and regulated spirometry integration define the deployment protocol. The method converts incomplete recordings into auditable screening evidence and routes each session through a documented clinical action.")

    add_heading(doc, "XIV  MATHEMATICAL ANALYSIS OF CENSORING")
    add_heading(doc, "A  Endpoint transformation under censoring", level=2)
    add_body(doc, "Let cumulative expired volume be evaluated from corrected onset, with τ denoting the censoring time and FVC the terminal plateau. Monotonic expiration places the observed volume at τ below or equal to FVC. The direct and complete ratios therefore satisfy the following exact ordering identity.")
    add_body(doc, "The residual-volume term determines the magnitude of the ratio transformation. The fixed 0.70 decision boundary connects this continuous quantity to the participant-level screening action, with the largest action-relevant effect among complete ratios immediately below the threshold.")
    add_figure(doc, ASSETS / "analysis_equations.png", "Fig. 10. Censoring identity and local information matrix. The first identity establishes the ordering between the ratio calculated at censoring time and the complete-curve ratio when the first second is observed.", width=3.08)
    add_heading(doc, "B  Censoring as a partially observed dynamical system", level=2)
    add_body(doc, "The admissible trajectory family comprises volume curves generated by parameters in the feasible set Θ. A restriction operator retains the trajectory through τ, and the inverse map recovers terminal FVC from that observed prefix. This endpoint-focused target reduces dimensionality and permits physiological constraints to govern continuation beyond the observation window.")
    add_body(doc, "Observation length controls the available trajectory information. Terminal curvature and residual slope enrich estimation of remaining volume as τ increases. Compact parameterization and bounded support define the admissible continuation manifold for earlier prefixes, while the selective rule uses boundary clearance and repeated-manoeuvre agreement to determine action.")
    add_heading(doc, "C  Local information geometry", level=2)
    add_body(doc, "The local information matrix is formed from the observed-prefix Jacobian and the inverse residual-variance matrix, as defined in Fig. 10. Longer observation adds late-time information about the slow component and terminal volume, while the rapid-rise segment primarily informs the onset time constant.")
    add_body(doc, "Uncertainty is propagated to FVC and the FEV1/FVC ratio, the functionals that determine the clinical action. Functional propagation focuses inference on the recovered endpoint and its distance from the 0.70 boundary, and repeated manoeuvres supply an independent stability coordinate.")
    add_heading(doc, "D  Conservation and monotonicity invariants", level=2)
    add_body(doc, "The flow kernel is nonnegative after multiplication by the finite-rise term. Numerical normalization divides by the trapezoidal integral over the model grid and multiplies by FVC, guaranteeing that the discretized area equals the total-volume parameter on that grid. Volume is then a cumulative trapezoidal integral of nonnegative flow. These operations enforce zero initial volume, monotonic cumulative volume, and terminal volume equal to FVC up to floating-point tolerance. Independent JAX and NumPy implementations agreed across 24 parameter sets with maximum volume discrepancy below 4 × 10⁻¹² mL.")
    add_body(doc, "The invariants enforce physiologically valid continuation by maintaining monotonic cumulative volume, bounded plateaus, and controlled tail behavior. Spline and linear-tail ablations verify that the constrained family preserves fit reliability and screening specificity.")

    add_heading(doc, "XV  OPTIMIZATION AND MODEL IDENTIFICATION")
    add_heading(doc, "A  Parameterization and feasible geometry", level=2)
    add_body(doc, "The five natural parameters occupy different numerical scales and obey domain restrictions. Time constants and FVC are positive, α lies in the unit interval, and the fast compartment remains faster than the slow component. The differentiable implementation represents positive quantities on logarithmic coordinates and α on a logistic coordinate for Fisher and synthetic analyses. The deterministic deployment solver operates directly in natural coordinates under the fixed feasible region in Section V-B.")
    add_body(doc, "Box constraints serve as a regularized scientific prior. They retain parameter combinations within the intended adult spirometric range, bound the unseen plateau, and expose solver-bound status. The bounds are fixed across cycles, truncation times, subgroups, and degradation experiments. Paired evaluation within complete curves measures the endpoint effect of the same constraints against the hidden complete endpoint.")
    add_heading(doc, "B  Objective scaling and robust noise estimation", level=2)
    add_body(doc, "The inverse objective uses cumulative volume because it matches the missing endpoint and attenuates high-frequency quantization. The residual vector is divided by a curve-level noise scale estimated from the difference between the observed trajectory and a Savitzky-Golay smooth. A dispersion-based finite fallback handles nonfinite or nonpositive estimates. Because the scale is constant within one curve, it preserves that curve's least-squares minimizer; its role is to express objective values and profile comparisons on a common noise-standardized scale.")
    add_body(doc, "A single nominal initialization is used: τf = 0.35 s, τs = 2.50 s, α = 0.60, FVC equal to the observed endpoint clipped to 2–6 L, and τo = 0.12 s. Trust-region reflective least squares accommodates active bounds and uses a three-point numerical Jacobian. A 600-function-evaluation budget fixes the computational budget. Fit success, evaluation count, parameter vector, recovered FVC, and solver status are retained. All 286 naturally incomplete paired recordings produced finite recovered endpoints.")
    add_body(doc, "A prespecified sensitivity audit refitted all 497 controlled Cycle F curves at 2, 3, 4, and 5 s under the nominal bounds, an FVC-only expansion to 0.75–7.50 L, and a simultaneous expansion of all five parameter ranges. All 5,964 fits converged. Under FVC-only expansion, the nominal classification was retained for 95.2%, 95.8%, 96.8%, and 97.6% of curves as observation increased from 2 to 5 s.")
    add_heading(doc, "C  Deterministic deployment estimator", level=2)
    add_body(doc, "Three-cycle replication, field-pair analysis, the severe cohort, ATS best-test policy, and device perturbations use the bounded deterministic solver. This inference path is auditable, reproducible, and computationally compatible with low-resource screening.")
    add_body(doc, "The constrained solver supplies a reproducible functional estimate for each curve. In an eight-start audit of 200 curves, 195 curves retained the nominal objective to within the prespecified material-improvement tolerance. Profile-likelihood analysis on 60 severity-balanced curves showed that the median log10 information-matrix condition number improved from 4.73 at 2 s to 3.67 at 5 s, quantifying the information gained from longer observation.")
    add_body(doc, "Participant bootstrap, interval-based boundary rules, cross-manoeuvre dispersion, and leave-one-manoeuvre-out analysis quantify decision stability and provide frequentist uncertainty for the quantities governing workflow actions.")
    add_heading(doc, "D  Computational complexity", level=2)
    add_body(doc, "For n observed samples, p model parameters, and K bounded solver iterations, the computational and memory requirements are")
    add_equation(doc, ASSETS / "complexity_equations.png", width=2.85)
    add_body(doc, "The trust-region step solves a five-dimensional system, and session aggregation is linear in the number of acceptable manoeuvres. The estimator retains only the current waveform and solver state during inference.")
    add_body(doc, "The deployed estimator solves a five-dimensional constrained optimization problem with explicit physical structure. This compact state supports auditability and device-specific calibration. Degradation experiments show that the paired recovery advantage persists under realistic sampling and noise budgets.")
    add_table(doc, "TABLE IX  ESTIMATOR DESIGN AND SCIENTIFIC ROLE", ["Component", "Inputs", "Output", "Validity test"], [
        ("Direct endpoint", "Complete V(t)", "Observed FVC", "Complete-curve kappa"),
        ("Inverse model", "Observed prefix to τ", "Recovered FVC", "Hidden-endpoint FN"),
        ("Best-test rule", "Accepted manoeuvres", "Session ratio", "Leave-one-manoeuvre-out"),
        ("Selector", "Fit, CI, grades", "Finalize/defer", "Risk-coverage"),
    ], widths=[.80,.78,.78,.88], font_size=6.7)

    add_heading(doc, "XVI  UNCERTAINTY AND SELECTIVE-RISK CALIBRATION")
    add_heading(doc, "A  Paired bootstrap inference", level=2)
    add_body(doc, "Artificial and natural truncation experiments compare estimators on the same participant or complete curve. Participant-level bootstrap resampling preserves this pairing, recomputes both obstruction-detection rates, and forms the sampling distribution of their difference. These intervals quantify uncertainty in cohort-level performance statistics; they are distinct from the participant-specific intervals used by the R2 decision rule.")
    add_body(doc, "At pooled 2-s truncation, the observed difference was 0.407 and the paired 95% interval 0.373-0.438. At 3 s, it was 0.271 with interval 0.242-0.302. In naturally incomplete recordings, the difference was 0.359 with interval 0.290-0.432. Paired resampling accounts for within-curve correlation, and replication across cycles establishes consistency across acquisition waves.")
    add_heading(doc, "B  Boundary-aware uncertainty", level=2)
    add_body(doc, "Decision uncertainty depends on distance from the 0.70 ratio boundary. The R2 policy resamples the available repeated-manoeuvre ratios for one participant and compares the resulting participant interval with that boundary. Obstructed calls require the upper endpoint below 0.70. Normal calls require the lower endpoint at or above 0.70 together with a stronger median margin. The resulting interval rule expresses decision stability directly.")
    add_body(doc, "Repeated manoeuvres provide a second uncertainty axis. Accepted efforts that straddle the boundary trigger deferral under the conservative normal-call rule. Obstructed best-test calls can advance to clinical evaluation under a lower repeatability requirement, reflecting the greater cost of false reassurance.")
    add_heading(doc, "C  Risk coverage and abstention", level=2)
    add_body(doc, "Risk-coverage analysis separates prediction from selection. The Cycle F nested out-of-fold search selected 75% coverage. At that fixed operating point, physics-informed reliability achieved kappa of 0.952 and 0.959 in Cycles E and G, and its paired improvement over ridge-margin ranking and the same-capacity nonphysics selector excluded zero in both cycles.")
    add_body(doc, "The selector uses concordance, signed ratio difference, recovered-FVC discrepancy, and volume-fit quality as physics coordinates alongside ridge margin and acquisition-state features. Nonselected cases receive an explicit repeat-testing, referral, or indeterminate action with a recorded deferral reason.")
    add_heading(doc, "D  Zero-event severe results", level=2)
    add_body(doc, "The pooled natural severe cohort recorded zero false negatives among 46 recovered cases, and the final Cycle F policy classified all seven severe finalized cases correctly. Exact intervals accompany these event counts, and the same fixed severity definition supports prespecified monitoring in extension studies.")
    add_heading(doc, "XVII  REPRODUCIBILITY AND EXTENSION STUDIES")
    add_heading(doc, "A  Gate-based scientific development", level=2)
    add_body(doc, "The repository organizes development through explicit scientific gates. Early modules test segmentation, parser identity, model conservation, and synthetic recovery. Real-curve gates evaluate residual structure, complete-curve classification, truncation recovery, severity, repeatability, uncertainty, clinical policy, subgroup transport, and device perturbations. Each gate writes machine-readable results and a human-readable report.")
    add_body(doc, "This process established selective recovery of latent FVC under endpoint censoring as the validated estimand and identified physics-derived reliability as the system-level contribution beyond ridge regression. The final method, participant-level action policy, and evidence hierarchy follow those experimentally defined roles.")
    add_heading(doc, "B  Determinism and provenance", level=2)
    add_body(doc, "Every deployment-stage experiment declares its random seed, source artifacts, cohort filters, estimator tuple, and decision criteria. The three-cycle analysis verifies parser identity, checks the forward model before and after execution, and uses global cycle-participant keys to prevent identifier collision. Degradation tests use in-memory copies, preserving the raw NHANES files. Selection rules are applied before truth scoring. These details establish provenance from raw waveform semantics through the final participant-level decision.")
    add_body(doc, "Raw NHANES files are managed outside source control under the repository data policy. The validation configuration records cryptographic hashes for the four downloaded CDC files, and the cross-cycle runner regenerates participant rows, confidence intervals, paired differences, and manuscript comparison tables from those inputs. The repository preserves configuration, cohort counts, row-level outputs, human-readable reports, and the commands required for parser validation, ridge audit, frozen scoring, and manuscript generation.")
    add_heading(doc, "C  Minimum prospective study design", level=2)
    add_body(doc, "The extension-study protocol enrolls participants undergoing routine spirometry across at least two device classes and multiple operator experience levels. The policy, parameter bounds, and thresholds are frozen before enrollment. Each session retains every attempted flow-volume trace, device quality flag, calibration status, bronchodilator status, and operator annotation. Independent pulmonary-function experts, blinded to the algorithmic action, adjudicate technical acceptability.")
    add_body(doc, "The reference combines ATS/ERS-compliant post-bronchodilator best-test spirometry with the clinical information required for COPD assessment. Primary endpoints include severe and overall false-normal rates among finalized cases, coverage, specificity, and repeat-test reduction. Secondary endpoints include recovered-FVC error, time to valid session, action agreement, device-stratified calibration, subgroup performance, and the outcome of every deferral.")
    add_body(doc, "Sample-size planning uses the severe false-normal bound and the desired precision of the direct-minus-recovery difference. Multicenter accumulation and enriched recruitment supply the severe-obstruction stratum. Prespecified exact intervals govern interim monitoring, and participant-level pairing supports both technical-performance and survey-weighted population analyses.")
    add_heading(doc, "D  Clinical and ethical governance", level=2)
    add_body(doc, "The intended benefit is greatest in settings with limited access to expert supervision and repeat testing. Operators should see the endpoint source, evidence supporting the action, and any deferral reason. Symptoms, exposures, and independent measurements remain part of clinical review after a recovered-normal result. Audit logs support continuous device-performance monitoring and outcome review.")
    add_body(doc, "Subgroup monitoring separates biological transport from acquisition-quality variation. The present within-NHANES subgroup results show positive recovery margins across age and smoking strata. Deployment governance measures coverage, conditional error, and referral burden by clinically relevant groups under fixed technical standards.")
    add_heading(doc, "E  Falsifiable extension hypotheses", level=2)
    add_body(doc, "The current evidence generates three testable hypotheses. Endpoint-specific recovery should show its largest advantage below 4 s and converge toward direct measurement as plateau information accumulates. A calibrated hierarchical model sharing slow-compartment information across repeated efforts should narrow FVC uncertainty with preserved specificity. Device-native pressure and flow-quality signals should improve selection by distinguishing physiological censoring from sensor artifact.")
    add_body(doc, "A future neural extension can target early-rise morphology through a constrained residual model that preserves monotonic volume and exact terminal conservation. Evaluation should measure incremental endpoint recovery, calibration under device shift, and risk-coverage change on the locked residual-deferral set.")
    add_table(doc, "TABLE XI  PROSPECTIVE REPORTING REQUIREMENTS", ["Domain", "Required record", "Purpose"], [
        ("Acquisition", "device, rate, calibration", "Domain shift"),
        ("Manoeuvre", "raw trace, flags, duration", "Technical validity"),
        ("Inference", "fit, FVC, interval, source", "Traceability"),
        ("Session", "best-test evidence, grade", "Repeatability"),
        ("Action", "finalize, repeat, refer", "Workflow impact"),
        ("Outcome", "post-BD test and review", "Clinical validity"),
    ], widths=[.72,1.42,1.02], font_size=6.7)

    add_heading(doc, "APPENDIX A  SPIROMETRY INFERENCE ALGORITHM")
    add_body(doc, "Algorithm 1 summarizes the deterministic session-level execution path. Inputs comprise all manoeuvres from one testing session together with acquisition factors and quality metadata. The output records the selected clinical action and its supporting evidence. Session-level inference preserves acceptability, repeatability, and best-test properties across multiple efforts.")
    add_table(doc, "ALGORITHM 1  SELECTIVE SESSION INFERENCE", ["Step", "Operation", "Recorded output"], [
        ("1", "Decode increments and apply SPAFACT", "Flow and volume"),
        ("2", "Detect onset, PEF, and end", "Corrected time zero and duration"),
        ("3", "Classify completeness and quality", "Reliability state"),
        ("4", "Fit bounded model and apply correction", "Model-derived ratio"),
        ("5", "Aggregate acceptable repeated efforts", "Repeatability evidence"),
        ("6", "Apply recovery and ATS extensions", "Physics decision"),
        ("7", "Evaluate state-aware ridge decision", "Base decision"),
        ("8", "Score physics-informed reliability", "Selection score"),
        ("9", "Return finalize or defer", "Action and reason"),
    ], widths=[.42,1.70,1.08], font_size=6.6)
    add_body(doc, "Algorithm 1 describes the evaluated implementation. The endpoint path routes direct and recovered FVC by measurement state. The reliability model consumes cross-fitted ridge confidence, acquisition-state variables, physics-ridge agreement, signed ratio difference, recovered-FVC discrepancy, and fit quality. Outcome labels enter only after selection.")

    add_heading(doc, "APPENDIX B  CLAIM TO EVIDENCE MAP")
    add_body(doc, "The central claims are linked to distinct validation units. Parser fidelity uses participant-level comparison with released summary endpoints. Controlled censoring uses within-curve pairs. Cycle F supplies development and cross-fitting evidence. Cycles E and G test cross-cycle transport of classification, FVC error, deferral, and incremental physics-derived value. The extension-study protocol evaluates post-bronchodilator and cross-device transport.")
    add_table(doc, "TABLE XII  CLAIM TO EVIDENCE MAP", ["Claim", "Validation unit", "Primary statistic"], [
        ("Parser fidelity", "20,050 participants", "Median abs. error"),
        ("Censoring recovery", "892 obstructed curves", "Paired FN difference"),
        ("Field behavior", "286 natural pairs", "93.3% sensitivity"),
        ("Severe recovery", "46 natural severe", "46 / 46 detected"),
        ("Workflow development", "563 Cycle F cases", "Coverage and kappa"),
        ("Cross-cycle selection", "12,434 E/G cases", "Kappa .952 / .959"),
        ("Device robustness", "fixed EFG curves", "Margin under budget"),
    ], widths=[1.05,1.12,1.12], font_size=6.5)
    add_body(doc, "Each validation unit supports a distinct claim. Synthetic tests verify implementation. Artificial truncation measures hidden-endpoint recovery. Natural pairs characterize field behavior. Cross-cycle evaluation measures transport and incremental reliability value. Together, these units provide a traceable progression from numerical invariants to participant-level screening actions.")

    add_heading(doc, "XVIII  CONCLUSION")
    add_body(doc, "Evaluation on 12,434 Cycle E/G participants established measurable physics-derived value beyond ridge regression. At fixed 75% coverage, the physics-informed selector achieved sensitivity of 96.0-96.1%, specificity of 99.7-99.8%, and kappa of 0.952-0.959. Kappa exceeded the identical nonphysics selector by 0.019 and 0.030 and ridge-margin selection by 0.045 and 0.042, with paired confidence intervals above zero in both cycles. The physics-plus-ridge endpoint model also reduced FVC error in both cycles. The resulting framework links censored-waveform reconstruction, state-aware endpoint routing, repeated-manoeuvre evidence, and learned reliability within one auditable screening workflow.")

    add_heading(doc, "REFERENCES")
    refs = [
        "[1] Global Initiative for Chronic Obstructive Lung Disease, Global Strategy for Prevention, Diagnosis and Management of COPD: 2026 Report, 2026.",
        "[2] B. L. Graham et al., Standardization of Spirometry 2019 Update: An Official American Thoracic Society and European Respiratory Society Technical Statement, Am. J. Respir. Crit. Care Med., vol. 200, no. 8, pp. e70-e88, 2019, doi: 10.1164/rccm.201908-1590ST.",
        "[3] National Center for Health Statistics, NHANES Spirometry Raw Curve Data Documentation, cycles 2007-2008, 2009-2010, and 2011-2012, Centers for Disease Control and Prevention.",
        "[4] J. G. Rijcken et al., Time constant histograms from the forced expired volume signal: a clinical evaluation, Eur. Respir. J., vol. 2, no. 6, pp. 536-545, 1989.",
        "[5] H. Steltner, M. Vogel, S. Sorichter, H. Matthys, J. Guttmann, and J. Timmer, Analysis of forced expired volume signals using multi-exponential functions, Med. Biol. Eng. Comput., vol. 39, no. 2, pp. 190-194, 2001, doi: 10.1007/BF02344803.",
        "[6] A. Tarantola, Inverse Problem Theory and Methods for Model Parameter Estimation. Philadelphia, PA, USA: SIAM, 2005.",
        "[7] R. El-Yaniv and Y. Wiener, On the foundations of noise-free selective classification, J. Mach. Learn. Res., vol. 11, pp. 1605-1641, 2010.",
        "[8] Y. Geifman and R. El-Yaniv, SelectiveNet: A deep neural network with an integrated reject option, in Proc. 36th Int. Conf. Machine Learning, 2019, pp. 2151-2159.",
        "[9] P. H. Quanjer et al., Multi-ethnic reference values for spirometry for the 3-95-yr age range: the Global Lung Function 2012 equations, Eur. Respir. J., vol. 40, no. 6, pp. 1324-1343, 2012, doi: 10.1183/09031936.00080312.",
        "[10] S. Stanojevic et al., ERS/ATS technical standard on interpretive strategies for routine lung function tests, Eur. Respir. J., vol. 60, 2101499, 2022, doi: 10.1183/13993003.01499-2021.",
        "[11] T. Fawcett, An introduction to ROC analysis, Pattern Recognit. Lett., vol. 27, no. 8, pp. 861-874, 2006.",
        "[12] B. Efron and R. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman and Hall, 1993.",
        "[13] J. Cohen, A coefficient of agreement for nominal scales, Educ. Psychol. Meas., vol. 20, no. 1, pp. 37-46, 1960.",
        "[14] C. J. Clopper and E. S. Pearson, The use of confidence or fiducial limits illustrated in the case of the binomial, Biometrika, vol. 26, no. 4, pp. 404-413, 1934.",
        "[15] P. J. Huber, Robust Statistics. New York, NY, USA: Wiley, 1981.",
        "[16] J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed. New York, NY, USA: Springer, 2006.",
        "[17] P. Sadani and M. Gupta, An Integrated Monitoring Device for Assessing and Predicting the Possibility of Occurrence of at Least Five Major Diseases Including Breast Cancer, Liver Disease, Kidney Disease, Heart Disease, and Diabetes, Indian provisional patent specification, 2026.",
        "[18] R. Perez-Padilla et al., Reliability of FEV1/FEV6 to diagnose airflow obstruction compared with FEV1/FVC: the PLATINO longitudinal study, PLoS ONE, vol. 8, no. 8, e67960, 2013, doi: 10.1371/journal.pone.0067960.",
        "[19] S. P. Bhatt et al., FEV1/FEV6 to diagnose airflow obstruction: comparisons with computed tomography and morbidity indices, Ann. Am. Thorac. Soc., vol. 11, no. 3, pp. 335-341, 2014, doi: 10.1513/AnnalsATS.201308-251OC.",
        "[20] Y. Wang et al., Deep learning for spirometry quality assurance with spirometric indices and curves, Respir. Res., vol. 23, art. 98, 2022, doi: 10.1186/s12931-022-02014-9.",
        "[21] S. Bodduluri et al., Deep neural network analyses of spirometry for structural phenotyping of chronic obstructive pulmonary disease, JCI Insight, vol. 5, no. 13, e132781, 2020, doi: 10.1172/jci.insight.132781.",
        "[22] D. Hill et al., Deep learning of suboptimal spirometry to predict respiratory outcomes and mortality, Research Square, preprint, 2025, doi: 10.21203/rs.3.rs-6296752/v1.",
        "[23] H. Steltner, M. Vogel, E. Sprung, J. Timmer, J. Guttmann, and S. Sorichter, Incomplete forced expiration: estimating vital capacity by a mathematical method, Respiration, vol. 71, no. 4, pp. 353-359, 2004, doi: 10.1159/000079639.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.16)
        p.paragraph_format.first_line_indent = Inches(-0.16)
        p.paragraph_format.space_after = Pt(1.2)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        set_font(p.add_run(ref), size=7.4)

    # A terminal continuous section balances the final two-column reference
    # page in Word-compatible renderers.
    balance_section = doc.add_section(WD_SECTION.CONTINUOUS)
    balance_section.top_margin = Inches(0.65)
    balance_section.bottom_margin = Inches(0.65)
    balance_section.left_margin = Inches(0.72)
    balance_section.right_margin = Inches(0.72)
    set_columns(balance_section, 1)
    balance_section.footer.is_linked_to_previous = True

    reorder_major_sections(doc)

    # Document metadata and compatibility settings
    doc.core_properties.title = "Physiology-Constrained Reconstruction of Censored Spirometry"
    doc.core_properties.subject = "IEEE-style research manuscript"
    doc.core_properties.author = "Mrityunjay Gupta"
    doc.core_properties.keywords = "spirometry, censored time series, inverse modeling, selective prediction, COPD screening"

    settings = doc.settings._element
    compat = settings.find(qn("w:compat"))
    if compat is None:
        compat = OxmlElement("w:compat")
        settings.append(compat)
    balance = OxmlElement("w:balanceSingleByteDoubleByteWidth")
    compat.append(balance)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
