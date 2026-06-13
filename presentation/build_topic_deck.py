#!/usr/bin/env python3
"""Build the ROGJ DeepSpeech2 VEPRAD presentation deck.

The deck is generated as native PowerPoint XML so the text boxes, tables, and
simple diagrams remain editable in PowerPoint. The architecture slide embeds a
single generated raster image copied from the earlier project folder.
"""

from __future__ import annotations

import html
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


OUT = Path(__file__).with_name("ROGJ_Deepspeech2_topic_presentation.pptx")
ARCH_IMAGE = Path(__file__).with_name("deepspeech2_architecture_generated_slide.png")

SLIDE_W = 12192000
SLIDE_H = 6858000
PX_W = 1920
PX_H = 1080
SLIDE_COUNT = 7

INK = "15171A"
INK_2 = "20282C"
CREAM = "F8F3E8"
SAND = "F2DFC7"
LINE = "D6C8B5"
MUTED = "6F6A5B"
RED = "E24A2A"
TEAL = "2C8C99"
GOLD = "F0B429"
GREEN = "5A7D3A"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def ex(x: float) -> int:
    return round(x / PX_W * SLIDE_W)


def ey(y: float) -> int:
    return round(y / PX_H * SLIDE_H)


def pt(size: float) -> int:
    return round(size * 100)


class Slide:
    def __init__(self, bg: str = CREAM):
        self.bg = bg
        self.parts: list[str] = []
        self.next_id = 2
        self.image_rels: list[tuple[str, str]] = []

    def _id(self) -> int:
        shape_id = self.next_id
        self.next_id += 1
        return shape_id

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        line: str | None = None,
        name: str = "Rectangle",
    ) -> None:
        shape_id = self._id()
        line_xml = (
            "<a:ln><a:noFill/></a:ln>"
            if line is None
            else f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        )
        self.parts.append(
            f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{ex(x)}" y="{ey(y)}"/><a:ext cx="{ex(w)}" cy="{ey(h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
          <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}
        </p:spPr>
      </p:sp>"""
        )

    def line(self, x: float, y: float, w: float, h: float, color: str = RED) -> None:
        self.rect(x, y, w, h, color, None, "Rule")

    def text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        lines: list[tuple[str, float, str, bool]],
        name: str = "Text",
        margin: float = 0,
    ) -> None:
        shape_id = self._id()
        paragraphs = []
        for text, size, color, bold in lines:
            bold_attr = ' b="1"' if bold else ""
            paragraphs.append(
                f'<a:p><a:r><a:rPr lang="hr-HR" sz="{pt(size)}"{bold_attr}>'
                f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
                f'<a:latin typeface="Aptos"/></a:rPr><a:t>{esc(text)}</a:t></a:r>'
                f'<a:endParaRPr lang="hr-HR" sz="{pt(size)}"/></a:p>'
            )
        inset = ex(margin)
        self.parts.append(
            f"""
      <p:sp>
        <p:nvSpPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
        <p:spPr>
          <a:xfrm><a:off x="{ex(x)}" y="{ey(y)}"/><a:ext cx="{ex(w)}" cy="{ey(h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln>
        </p:spPr>
        <p:txBody><a:bodyPr wrap="square" lIns="{inset}" tIns="{inset}" rIns="{inset}" bIns="{inset}"/><a:lstStyle/>{''.join(paragraphs)}</p:txBody>
      </p:sp>"""
        )

    def label(self, number: str, text: str, y: float, color: str = RED) -> None:
        self.rect(70, y, 42, 42, color, None, "Kicker marker")
        self.text(81, y + 6, 26, 24, [(number, 16, CREAM if color != GOLD else INK, True)], "Kicker number")
        self.text(130, y + 4, 540, 34, [(text, 16, color, True)], "Kicker label")

    def image(self, x: float, y: float, w: float, h: float, target: str, name: str) -> None:
        shape_id = self._id()
        rid = f"rId{len(self.image_rels) + 2}"
        self.image_rels.append((rid, target))
        self.parts.append(
            f"""
      <p:pic>
        <p:nvPicPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
        <p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
        <p:spPr>
          <a:xfrm><a:off x="{ex(x)}" y="{ey(y)}"/><a:ext cx="{ex(w)}" cy="{ey(h)}"/></a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>"""
        )

    def table(
        self,
        x: float,
        y: float,
        widths: list[float],
        row_h: float,
        headers: list[str],
        rows: list[list[str]],
        header_fill: str,
        font_size: float = 11,
        first_col_bold: bool = True,
    ) -> None:
        total_w = sum(widths)
        self.rect(x, y, total_w, row_h * (len(rows) + 1), CREAM, LINE, "Table background")
        self.rect(x, y, total_w, row_h, header_fill, None, "Table header")
        for row_i in range(len(rows)):
            if row_i % 2 == 1:
                self.rect(x, y + row_h * (row_i + 1), total_w, row_h, SAND, None, "Table stripe")
        cx = x
        for header, width in zip(headers, widths):
            self.text(cx + 8, y + 9, width - 16, row_h - 12, [(header, font_size, CREAM if header_fill != GOLD else INK, True)], "Header")
            cx += width
        for row_i, row in enumerate(rows):
            cx = x
            for col_i, (value, width) in enumerate(zip(row, widths)):
                cell_lines = [(part, font_size, INK if col_i == 0 else INK_2, first_col_bold and col_i == 0) for part in value.split("\n")]
                self.text(cx + 8, y + row_h * (row_i + 1) + 8, width - 16, row_h - 10, cell_lines, "Cell")
                cx += width

    def xml(self) -> str:
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{self.bg}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(self.parts)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""

    def rels(self) -> str:
        rels = [
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
        ]
        for rid, target in self.image_rels:
            rels.append(
                f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n  '
            + "\n  ".join(rels)
            + "\n</Relationships>\n"
        )


def title(slide: Slide, kicker: str, claim: str, support: str | None = None) -> None:
    slide.text(70, 46, 580, 32, [(kicker, 15, RED, True)], "Kicker")
    slide.text(70, 82, 1180, 62, [(claim, 25, INK, True)], "Title")
    if support:
        slide.text(70, 150, 1180, 46, [(support, 11.8, MUTED, False)], "Support")
    slide.line(70, 210, 116, 7, RED)


def footer(slide: Slide, text: str) -> None:
    slide.text(70, 1038, 1450, 24, [(text, 8.5, MUTED, False)], "Footer")


def slide1() -> Slide:
    s = Slide()
    s.rect(0, 0, 505, 1080, INK)
    s.rect(505, 0, 28, 1080, RED)
    s.text(66, 72, 320, 38, [("ROGJ 2025/26", 22, GOLD, True)])
    s.text(66, 136, 330, 52, [("Tema 12", 34, CREAM, True)])
    s.text(
        66,
        215,
        340,
        150,
        [
            ("Učenje akustičkih", 23, SAND, False),
            ("modela govora", 23, SAND, False),
            ("za raspoznavanje", 23, SAND, False),
        ],
    )
    s.text(
        610,
        105,
        1050,
        118,
        [("DeepSpeech2 na VEPRAD-u", 41, INK, True)],
    )
    s.text(
        616,
        292,
        1050,
        72,
        [
            (
                "Zadatak: na temelju baze snimljenog i transkribiranog govora naučiti akustičke modele i testirati rad sustava.",
                23,
                INK_2,
                False,
            )
        ],
    )
    s.line(616, 418, 880, 9, RED)
    s.text(616, 462, 900, 70, [("VEPRAD -> splitovi -> DeepSpeech2 trening -> KenLM dekodiranje -> WER/CER", 21, INK, True)])
    stats = [
        ("3", "trenirana modela", TEAL),
        ("6", "eval subsetova", GOLD),
        ("0.118", "najbolji test WER", RED),
    ]
    x = 616
    for value, label, color in stats:
        s.rect(x, 580, 245, 116, color)
        s.text(x + 22, 596, 180, 44, [(value, 30, CREAM if color != GOLD else INK, True)])
        s.text(x + 22, 646, 200, 25, [(label, 13, CREAM if color != GOLD else INK, False)])
        x += 285
    s.text(
        616,
        770,
        980,
        130,
        [
            ("U prezentaciji: kako smo izolirali test skup, kako je definiran CV split, koje DS2 konfiguracije smo trenirali i koliko KenLM mijenja rezultate.", 19, MUTED, False)
        ],
    )
    s.text(616, 968, 900, 32, [("Repo: PaddleSpeech DeepSpeech2 pipeline + VEPRAD manifesti + rezultatne metrike", 13, MUTED, False)])
    return s


def slide2() -> Slide:
    s = Slide()
    title(
        s,
        "02 · PODACI I SPLITOVI",
        "Test skup je odvojen prije cross-validacije.",
        "VEPRAD se tretira kao govor + transkript, s govornicima kao važnom osi generalizacije.",
    )
    s.text(70, 250, 430, 34, [("Permanentni test skup", 18, INK, True)])
    s.line(70, 288, 90, 6, RED)
    steps = [
        ("1", "3 govornika se potpuno izdvajaju za test", RED),
        ("2", "iz svih ostalih govornika uzima se oko 10% utterancea za test", TEAL),
        ("3", "test audio + test JSONL žive u data/test i ne ulaze u CV", GOLD),
    ]
    y = 320
    for num, text, color in steps:
        s.rect(70, y, 42, 42, color)
        s.text(83, y + 7, 25, 22, [(num, 15, CREAM if color != GOLD else INK, True)])
        s.text(130, y + 2, 430, 58, [(text, 12.5, INK_2, True)])
        y += 92
    s.text(70, 610, 520, 70, [("sm04 je ostavljen izvan aktivnog split poola jer nije bilo dovoljno jasno što točno predstavlja; time smanjujemo rizik od curenja podataka.", 12.2, MUTED, False)])

    s.text(650, 250, 500, 34, [("Jedan CV split", 18, INK, True)])
    s.line(650, 288, 90, 6, TEAL)
    s.table(
        650,
        320,
        [230, 150, 150, 150, 170],
        56,
        ["Subset", "utt.", "gov.", "h", "uloga"],
        [
            ["train", "3057", "19", "4.72", "učenje modela"],
            ["val", "1205", "22", "1.81", "odabir/usporedba"],
            ["val seen", "349", "19", "0.54", "isti govornici"],
            ["val unseen", "856", "3", "1.26", "novi govornici"],
            ["test", "993", "25", "1.50", "zamrznuta provjera"],
            ["test seen", "486", "22", "0.74", "isti govornici"],
            ["test unseen", "507", "3", "0.76", "novi govornici"],
        ],
        TEAL,
        10,
    )
    s.text(
        650,
        760,
        770,
        82,
        [("Namjera splitova je mjeriti dvije stvari odvojeno: koliko model generalizira na nove rečenice istih govornika i koliko generalizira na potpuno neviđene govornike.", 14, MUTED, False)],
    )
    s.rect(70, 880, 1490, 74, SAND, LINE)
    s.text(96, 898, 1438, 38, [("Reproducibilnost: permanentni test skup ne diramo nakon stvaranja; za nove validacijske eksperimente generiramo nove CV splitove iz non-test poola.", 14.5, INK, True)])
    footer(s, "Izvori u repozitoriju: data/test/*.meta.json, data/cross_validation_splits/cv_paper_small_001, README.")
    return s


def slide3() -> Slide:
    s = Slide()
    title(
        s,
        "03 · PIPELINE",
        "Pipeline je sada jedan Python tok, ne stari shell recipe.",
        "Skripta poziva PaddleSpeech Python funkcije direktno: split/manifesti, trening, averaging, eval i metrike.",
    )
    x0, y0 = 92, 292
    stages = [
        ("VEPRAD", "audio + txt", INK),
        ("manifesti", "JSONL rel. paths", TEAL),
        ("features", "fbank_kaldi + CMVN", GOLD),
        ("vocab", "31 char, train-only", GREEN),
        ("DS2", "CTC trening", RED),
        ("eval", "WER/CER po subsetu", INK),
    ]
    x = x0
    for i, (head, sub, color) in enumerate(stages):
        s.rect(x, y0, 202, 92, color)
        s.text(x + 18, y0 + 16, 170, 28, [(head, 18, CREAM if color != GOLD else INK, True)])
        s.text(x + 18, y0 + 53, 168, 20, [(sub, 11.5, CREAM if color != GOLD else INK, False)])
        if i < len(stages) - 1:
            s.text(x + 214, y0 + 26, 36, 32, [("→", 24, RED, True)])
        x += 248

    s.text(92, 470, 430, 30, [("Preprocessing odluke", 18, INK, True)])
    s.line(92, 508, 90, 6, RED)
    s.text(
        92,
        530,
        570,
        190,
        [
            ("• 16 kHz, 25 ms window, 10 ms shift", 13.5, INK_2, False),
            ("• 161-bin fbank_kaldi + CMVN", 13.5, INK_2, False),
            ("• character CTC vocab built only from train", 13.5, INK_2, False),
            ("• Croatian letters covered: 0 unknown chars on val/test", 13.5, INK_2, False),
            ("• no data augmentation in these runs", 13.5, INK_2, False),
        ],
    )
    s.text(750, 470, 470, 30, [("Transkripti i event tagovi", 18, INK, True)])
    s.line(750, 508, 90, 6, TEAL)
    s.text(
        750,
        530,
        650,
        190,
        [
            ("ASR target je normalizirano polje text.", 14, INK_2, True),
            ("Event tagovi poput <uzdah> ostaju u text_raw/non_speech_events, ali se ne treniraju kao CTC znakovi jer bi se u character modu rastavili na <, u, z, ...", 13, MUTED, False),
        ],
    )
    s.rect(92, 800, 1340, 78, INK)
    s.text(
        120,
        817,
        1280,
        42,
        [("Programski tok: scripts/run_ds2_pipeline.py generira PaddleSpeech podatke, trenira model, računa avg_1 checkpoint i evaluira val/test seen/unseen podskupove.", 13.5, CREAM, True)],
    )
    footer(s, "Pipeline datoteke: scripts/run_ds2_pipeline.py, scripts/create_cv_split.py, scripts/train_kenlm_lm.py.")
    return s


def slide4() -> Slide:
    s = Slide()
    title(
        s,
        "04 · DEEPSPEECH2",
        "DeepSpeech2 pretvara akustičke značajke u vjerojatnosti znakova kroz CTC.",
        "Slika prikazuje paper-style tok; desni stupac navodi što smo točno mogli reproducirati bez mijenjanja PaddleSpeech sourcea.",
    )
    s.image(54, 238, 1130, 635, "../media/deepspeech2_architecture_generated_slide.png", "DeepSpeech2 architecture")
    s.rect(1232, 242, 610, 446, INK)
    s.text(1260, 266, 560, 36, [("Što smo zadržali", 18, GOLD, True)])
    s.text(
        1260,
        320,
        542,
        166,
        [
            ("• konvolucijski front-end", 13.5, CREAM, False),
            ("• bidirectional offline recurrent layers", 13.5, CREAM, False),
            ("• character CTC targets", 13.5, CREAM, False),
            ("• beam search decoding path", 13.5, CREAM, False),
            ("• external word-level KenLM scorer", 13.5, CREAM, False),
        ],
    )
    s.text(1260, 520, 560, 34, [("Razlike od paper konfiguracije", 17, GOLD, True)])
    s.text(
        1260,
        568,
        542,
        94,
        [
            ("BatchNorm, exact spectrogram path, Nesterov optimizer and lookahead conv are not exposed cleanly in this PaddleSpeech DS2 path.", 12.6, SAND, False)
        ],
    )
    s.rect(1232, 728, 610, 96, SAND, LINE)
    s.text(1260, 750, 560, 48, [("Zaključak: model je paper-close approximation, ne bit-for-bit reprodukcija Deep Speech 2 rada.", 13.5, INK, True)])
    footer(s, "Architecture image: presentation/deepspeech2_architecture_generated_slide.png; exact configs: conf/deepspeech2_paper_*.yaml.")
    return s


def slide5() -> Slide:
    s = Slide()
    title(
        s,
        "05 · MODELI I HIPERPARAMETRI",
        "Trenirali smo paper-small konfiguraciju i dvije dublje varijante.",
        "Cilj nije bio samo dobiti jedan score, nego vidjeti što se mijenja s dubinom i tipom rekurentne ćelije.",
    )
    s.table(
        70,
        260,
        [280, 190, 170, 160, 150, 170, 230],
        62,
        ["Run", "recurrent", "layers", "hidden", "params", "epochs", "reason"],
        [
            ["paper_small_001", "GRU", "5 biRNN", "650", "36.21M", "20", "najbliže small DS2\nbez source edita"],
            ["paper_7gru_001", "GRU", "7 biRNN", "650", "50.73M", "20", "+2 sloja,\nisti cell"],
            ["paper_7lstm_001", "LSTM", "7 biRNN", "650", "67.61M", "20", "usporedba cella\ns istom dubinom"],
        ],
        RED,
        10.5,
    )
    s.text(70, 560, 450, 30, [("Zajedničko u treningu", 18, INK, True)])
    s.line(70, 598, 90, 6, TEAL)
    s.text(
        70,
        622,
        620,
        198,
        [
            ("• 2 convolution layers, no FC layers", 13.5, INK_2, False),
            ("• fbank_kaldi 161 + CMVN, no augmentation", 13.5, INK_2, False),
            ("• batch size 32, 8 workers", 13.5, INK_2, False),
            ("• lr 5e-4, exponential decay 0.93, grad clip 5.0", 13.5, INK_2, False),
            ("• avg_1 checkpoint used for final eval", 13.5, INK_2, False),
        ],
    )
    s.text(780, 560, 450, 30, [("Zašto ovako", 18, INK, True)])
    s.line(780, 598, 90, 6, GOLD)
    s.text(
        780,
        622,
        650,
        198,
        [
            ("Paper-small run drži hidden size 650 i GRU kao najbližu izvedivu varijantu unutar PaddleSpeecha.", 13.5, INK_2, False),
            ("Dublji GRU testira samu dubinu. LSTM testira je li druga recurrent cell bolja pod istim splitom i decoding protokolom.", 13.5, MUTED, False),
        ],
    )
    s.rect(70, 880, 1370, 78, SAND, LINE)
    s.text(96, 898, 1320, 36, [("Optimizer napomena: za razliku od DS2 papera, ovaj PaddleSpeech trainer koristi Adam + LR decay; to eksplicitno navodimo kao razliku od paper reprodukcije.", 13, INK, True)])
    footer(s, "Konfiguracije: conf/deepspeech2_paper_small.yaml, conf/deepspeech2_paper_7gru.yaml, conf/deepspeech2_paper_7lstm.yaml.")
    return s


def slide6() -> Slide:
    s = Slide()
    title(
        s,
        "06 · KENLM DEKODIRANJE",
        "KenLM je vanjski word-level scorer koji dramatično pomaže na vremenskim prognozama.",
        "LM se trenira iz train transkripata, ali prije toga izbacujemo rečenice koje se previše poklapaju s val/test tekstom.",
    )
    s.rect(70, 260, 500, 210, INK)
    s.text(96, 282, 450, 32, [("LM corpus filter", 18, GOLD, True)])
    s.text(
        96,
        328,
        430,
        106,
        [
            ("train rows: 3057", 13.5, CREAM, True),
            ("kept for KenLM: 2290", 13.5, CREAM, False),
            ("excluded: 767 exact/fuzzy held-out matches", 12.4, SAND, False),
        ],
    )
    s.rect(610, 260, 500, 210, TEAL)
    s.text(636, 282, 450, 32, [("Scorer", 18, CREAM, True)])
    s.text(
        636,
        328,
        430,
        106,
        [
            ("word-level KenLM 5-gram", 13.0, CREAM, True),
            ("is_character_based = 0", 12.4, CREAM, False),
            ("max_order = 5", 12.4, CREAM, False),
            ("dict_size = 1181", 12.4, CREAM, False),
        ],
    )
    s.rect(1150, 260, 500, 210, GOLD)
    s.text(1176, 282, 450, 32, [("Sweep grid", 18, INK, True)])
    s.text(
        1176,
        328,
        430,
        106,
        [
            ("beam 5/10/20/40 at a=2.5 b=0.3", 12.2, INK, True),
            ("beam 20 alpha: 2.0 / 3.0", 12.2, INK, False),
            ("beam 20 beta: 0.0 / 0.6", 12.2, INK, False),
        ],
    )

    s.text(70, 540, 520, 30, [("Najbolji validation-selected decoder", 18, INK, True)])
    s.line(70, 578, 90, 6, RED)
    s.table(
        70,
        612,
        [290, 250, 170, 170, 170, 170],
        58,
        ["Model", "best decoder", "val WER", "test WER", "seen", "unseen"],
        [
            ["paper-small GRU", "beam 40\na=2.5 b=0.3", "0.088", "0.118", "0.075", "0.160"],
            ["7-layer GRU", "beam 40\na=2.5 b=0.3", "0.094", "0.127", "0.083", "0.171"],
            ["7-layer LSTM", "beam 40\na=2.5 b=0.3", "0.123", "0.151", "0.095", "0.206"],
        ],
        RED,
        10.3,
    )
    s.rect(70, 844, 1220, 58, SAND, LINE)
    s.text(94, 858, 1170, 24, [("U sva tri modela isti decoder pobjeđuje na validation setu: beam 40, alpha 2.5, beta 0.3.", 12.2, INK, True)])
    footer(s, "Machine-readable sweep: results/kenlm_sweep_all_models/metrics.json.")
    return s


def slide7() -> Slide:
    s = Slide()
    title(
        s,
        "07 · REZULTATI",
        "Najbolji rezultat: paper-small GRU + KenLM.",
        "Validation odabire beam 40; KenLM smanjuje WER nekoliko puta u odnosu na no-LM CTC decoding.",
    )
    s.table(
        70,
        250,
        [280, 160, 160, 170, 160, 170, 170],
        60,
        ["Model", "No LM", "Fixed LM", "Best test", "Best val", "Seen", "Unseen"],
        [
            ["paper-small GRU", "0.453", "0.155", "0.118", "0.088", "0.075", "0.160"],
            ["7-layer GRU", "0.429", "0.142", "0.127", "0.094", "0.083", "0.171"],
            ["7-layer LSTM", "0.535", "0.175", "0.151", "0.123", "0.095", "0.206"],
        ],
        TEAL,
        10.2,
    )
    callouts = [
        ("Best model", "paper-small GRU", "0.118 test WER", RED),
        ("LM effect", "0.453 → 0.118", "best model test WER", TEAL),
        ("Speaker gap", "0.075 vs 0.160", "seen vs unseen test WER", GOLD),
    ]
    x = 90
    for head, value, note, color in callouts:
        s.rect(x, 548, 360, 120, color)
        s.text(x + 24, 566, 300, 24, [(head, 13, CREAM if color != GOLD else INK, True)])
        s.text(x + 24, 596, 310, 30, [(value, 18.5, CREAM if color != GOLD else INK, True)])
        s.text(x + 24, 636, 300, 22, [(note, 10.5, CREAM if color != GOLD else INK, False)])
        x += 405
    s.text(70, 735, 610, 34, [("Zaključci", 19, INK, True)])
    s.line(70, 773, 90, 6, RED)
    s.text(
        70,
        800,
        705,
        148,
        [
            ("• Sustav radi end-to-end na VEPRAD-u: splitovi, trening, dekodiranje i metrike su reproducibilni.", 13.5, INK_2, False),
            ("• Dublje nije automatski bolje: 5-layer GRU je bolji od 7GRU i 7LSTM u ovoj postavci.", 13.5, INK_2, False),
            ("• KenLM je ključan jer VEPRAD sadrži mnogo ponavljajućih vremenskih fraza.", 13.5, INK_2, False),
        ],
    )
    s.text(895, 735, 520, 34, [("Što bi bilo iduće", 19, INK, True)])
    s.line(895, 773, 90, 6, TEAL)
    s.text(
        895,
        800,
        620,
        148,
        [
            ("• više CV splitova za stabilniju procjenu", 13.5, INK_2, False),
            ("• n-gram order sweep za KenLM", 13.5, INK_2, False),
            ("• eventualno model/feature promjene koje zahtijevaju PaddleSpeech source edit", 13.5, INK_2, False),
        ],
    )
    footer(s, "Tracked results: results/kenlm_sweep_all_models/metrics.json and per-run summary.md files.")
    return s


def content_types() -> str:
    slide_overrides = "\n".join(
        f'  <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, SLIDE_COUNT + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
{slide_overrides}
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def package_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def app_props() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{SLIDE_COUNT}</Slides>
  <Notes>0</Notes>
</Properties>
"""


def core_props() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>ROGJ DeepSpeech2 VEPRAD Presentation</dc:title>
  <dc:subject>DeepSpeech2 acoustic model training on VEPRAD</dc:subject>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-06-13T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-06-13T00:00:00Z</dcterms:modified>
</cp:coreProperties>
"""


def presentation_xml() -> str:
    slide_ids = "\n".join(
        f'    <p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, SLIDE_COUNT + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>
{slide_ids}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""


def presentation_rels() -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    rels.extend(
        f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, SLIDE_COUNT + 1)
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n  '
        + "\n  ".join(rels)
        + "\n</Relationships>\n"
    )


def theme_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="ROGJ VEPRAD">
  <a:themeElements>
    <a:clrScheme name="ROGJ">
      <a:dk1><a:srgbClr val="{INK}"/></a:dk1>
      <a:lt1><a:srgbClr val="{CREAM}"/></a:lt1>
      <a:dk2><a:srgbClr val="{INK_2}"/></a:dk2>
      <a:lt2><a:srgbClr val="{SAND}"/></a:lt2>
      <a:accent1><a:srgbClr val="{RED}"/></a:accent1>
      <a:accent2><a:srgbClr val="{TEAL}"/></a:accent2>
      <a:accent3><a:srgbClr val="{GOLD}"/></a:accent3>
      <a:accent4><a:srgbClr val="{GREEN}"/></a:accent4>
      <a:accent5><a:srgbClr val="7B4A2D"/></a:accent5>
      <a:accent6><a:srgbClr val="{MUTED}"/></a:accent6>
      <a:hlink><a:srgbClr val="{TEAL}"/></a:hlink>
      <a:folHlink><a:srgbClr val="7B4A2D"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="ROGJ Fonts">
      <a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont>
      <a:minorFont><a:latin typeface="Aptos"/></a:minorFont>
    </a:fontScheme>
    <a:fmtScheme name="ROGJ Format">
      <a:fillStyleLst><a:solidFill><a:schemeClr val="accent1"/></a:solidFill></a:fillStyleLst>
      <a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="accent1"/></a:solidFill></a:ln></a:lnStyleLst>
      <a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>
      <a:bgFillStyleLst><a:solidFill><a:schemeClr val="lt1"/></a:solidFill></a:bgFillStyleLst>
    </a:fmtScheme>
  </a:themeElements>
</a:theme>
"""


def master_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>
"""


def master_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""


def layout_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
</p:sldLayout>
"""


def layout_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def build() -> None:
    if not ARCH_IMAGE.exists():
        raise SystemExit(f"Missing architecture image: {ARCH_IMAGE}")

    slides = [slide1(), slide2(), slide3(), slide4(), slide5(), slide6(), slide7()]
    files: dict[str, str | bytes] = {
        "[Content_Types].xml": content_types(),
        "_rels/.rels": package_rels(),
        "docProps/app.xml": app_props(),
        "docProps/core.xml": core_props(),
        "ppt/presentation.xml": presentation_xml(),
        "ppt/_rels/presentation.xml.rels": presentation_rels(),
        "ppt/theme/theme1.xml": theme_xml(),
        "ppt/slideMasters/slideMaster1.xml": master_xml(),
        "ppt/slideMasters/_rels/slideMaster1.xml.rels": master_rels(),
        "ppt/slideLayouts/slideLayout1.xml": layout_xml(),
        "ppt/slideLayouts/_rels/slideLayout1.xml.rels": layout_rels(),
        "ppt/media/deepspeech2_architecture_generated_slide.png": ARCH_IMAGE.read_bytes(),
    }
    for i, slide in enumerate(slides, start=1):
        files[f"ppt/slides/slide{i}.xml"] = slide.xml()
        files[f"ppt/slides/_rels/slide{i}.xml.rels"] = slide.rels()

    with ZipFile(OUT, "w", ZIP_DEFLATED) as pptx:
        for name, data in files.items():
            pptx.writestr(name, data)
    print(OUT)


if __name__ == "__main__":
    build()
