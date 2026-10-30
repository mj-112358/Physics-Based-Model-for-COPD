from __future__ import annotations

from copy import deepcopy

from docx import Document
from docx.oxml.ns import qn

from build_predictx_ieee_paper import OUT, build as build_source


ORIGINAL_ORDER = [
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


FINAL_ORDER = [
    "I  INTRODUCTION",
    "II  RELATED WORK AND TECHNICAL GAP",
    "III  DATA AND STUDY DESIGN",
    "IV  SIGNAL RECONSTRUCTION",
    "V  PHYSIOLOGY-CONSTRAINED CENSORED-CURVE MODEL",
    "VI  MULTI-MANOEUVRE SELECTIVE INFERENCE",
    "XIV  MATHEMATICAL ANALYSIS OF CENSORING",
    "XV  OPTIMIZATION AND MODEL IDENTIFICATION",
    "VII  EXPERIMENTAL PROTOCOL",
    "VIII  RESULTS",
    "IX  ABLATION AND MODEL-SELECTION EVIDENCE",
    "X  ROBUSTNESS AND WITHIN-PROGRAM TRANSPORT",
    "XVI  UNCERTAINTY AND SELECTIVE-RISK CALIBRATION",
    "XVII  REPRODUCIBILITY AND EXTENSION STUDIES",
    "XIII  DEPLOYMENT CONSIDERATIONS",
    "XI  DISCUSSION",
    "XII  LIMITATIONS",
    "XVIII  CONCLUSION",
    "APPENDIX A  SPIROMETRY INFERENCE ALGORITHM",
    "APPENDIX B  CLAIM TO EVIDENCE MAP",
    "REFERENCES",
]


RENUMBER = {
    "XIV  MATHEMATICAL ANALYSIS OF CENSORING": "VII  MATHEMATICAL ANALYSIS OF CENSORING",
    "XV  OPTIMIZATION AND MODEL IDENTIFICATION": "VIII  OPTIMIZATION AND MODEL IDENTIFICATION",
    "VII  EXPERIMENTAL PROTOCOL": "IX  EXPERIMENTAL PROTOCOL",
    "VIII  RESULTS": "X  RESULTS",
    "IX  ABLATION AND MODEL-SELECTION EVIDENCE": "XI  ABLATION AND MODEL-SELECTION EVIDENCE",
    "X  ROBUSTNESS AND WITHIN-PROGRAM TRANSPORT": "XII  ROBUSTNESS AND WITHIN-PROGRAM TRANSPORT",
    "XVI  UNCERTAINTY AND SELECTIVE-RISK CALIBRATION": "XIII  UNCERTAINTY AND SELECTIVE-RISK CALIBRATION",
    "XVII  REPRODUCIBILITY AND EXTENSION STUDIES": "XIV  REPRODUCIBILITY AND EXTENSION STUDIES",
    "XIII  DEPLOYMENT CONSIDERATIONS": "XV  DEPLOYMENT CONSIDERATIONS",
    "XI  DISCUSSION": "XVI  DISCUSSION",
    "XII  LIMITATIONS": "XVII  LIMITATIONS",
}


PLAINER_PARAGRAPHS = {
    "The technical objective is to infer latent FVC from a two-to-five-second prefix while preserving session quality control and quantifying when the recovered endpoint supports action. We introduce a state-conditional censored-volume estimator in which censoring time enters the inverse objective, the forward trajectory obeys monotonicity and total-volume conservation, and the participant-level selector returns classification, repeat testing, or referral under an explicit coverage-risk definition. Agreement between the mechanistic and learned estimators supplies a frozen reliability coordinate whose incremental value is evaluated at matched coverage.":
    "The goal is to estimate the final FVC from the first two to five seconds of expiration and to decide when that estimate supports action. The model includes the observed stopping time, preserves monotone volume and total-volume conservation, and returns either a classification or a deferral. A separate ridge model provides an independent estimate. Their agreement is evaluated at the same coverage as ridge confidence ranking.",
    "The five parameters define the family of continuations permitted after the observed prefix. The optimizer returns the terminal FVC implied by the best-fitting continuation. Fisher-information analysis characterizes uncertainty in the fitted parameters, while endpoint error measures the quantity used in the screening ratio.":
    "The five parameters define the possible continuations after the recorded prefix. The optimizer selects the continuation that best fits the observed volume and reports its terminal FVC. Fisher-information analysis measures how strongly the prefix constrains the parameters, while endpoint error evaluates the recovered quantity used in the screening ratio.",
    "The state-specific ablation established the routing principle used by the research design. Direct measurement reached kappa 0.918 on complete Cycle F manoeuvres. For censored manoeuvres, physiology-constrained recovery increased obstruction detection after 2-s truncation from 1.3% to 40.7%. These complementary results support direct endpoints for observed plateaus and structural recovery for latent endpoints.":
    "The ablation supports a simple routing rule. Complete manoeuvres use their measured endpoint, which reached kappa 0.918 in Cycle F. Incomplete manoeuvres use recovery: after two-second truncation, detection increased from 1.3% to 40.7%. The workflow therefore applies each estimator to the measurement state it was designed for.",
    "The participant-level contribution is a selective action function over measurement state and repeated efforts. It records whether FVC was observed or recovered, the manoeuvres supporting session maxima, uncertainty relative to the decision boundary, and the reason for classification, repeat testing, or referral. The same risk-coverage surface supports direct comparison with mechanistic extrapolators, clinical quality control, FEV1/FEV6, and learned estimators.":
    "At participant level, the workflow records whether FVC was measured or recovered, which manoeuvres supplied the session values, how far the estimate lies from the decision boundary, and why the case was classified or deferred. This common action record allows direct comparison with clinical quality control, FEV1/FEV6, mathematical extrapolation, and learned estimators.",
}


def paragraph_text(element) -> str:
    if element.tag != qn("w:p"):
        return ""
    return "".join(node.text or "" for node in element.iter(qn("w:t"))).strip()


def replace_paragraph_text(paragraph, text: str) -> None:
    if not paragraph.runs:
        paragraph.add_run(text)
        return
    paragraph.runs[0].text = text
    for run in paragraph.runs[1:]:
        run.text = ""


def reorder_sections(doc: Document) -> None:
    body = doc._element.body
    children = list(body)
    terminal_sectpr = children[-1] if children and children[-1].tag == qn("w:sectPr") else None
    content = children[:-1] if terminal_sectpr is not None else children

    starts = {}
    for index, element in enumerate(content):
        text = paragraph_text(element)
        if text in ORIGINAL_ORDER:
            starts[text] = index
    missing = [name for name in ORIGINAL_ORDER if name not in starts]
    if missing:
        raise RuntimeError(f"Missing section headings: {missing}")

    prefix = [deepcopy(x) for x in content[: starts[ORIGINAL_ORDER[0]]]]
    ranges = {}
    for index, name in enumerate(ORIGINAL_ORDER):
        start = starts[name]
        end = starts[ORIGINAL_ORDER[index + 1]] if index + 1 < len(ORIGINAL_ORDER) else len(content)
        ranges[name] = [deepcopy(x) for x in content[start:end]]

    for child in list(body):
        body.remove(child)
    for element in prefix:
        body.append(element)
    for name in FINAL_ORDER:
        for element in ranges[name]:
            body.append(element)
    if terminal_sectpr is not None:
        body.append(deepcopy(terminal_sectpr))


def build() -> None:
    build_source()
    doc = Document(OUT)
    reorder_sections(doc)

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if text in RENUMBER:
            replace_paragraph_text(paragraph, RENUMBER[text])
        elif text in PLAINER_PARAGRAPHS:
            replace_paragraph_text(paragraph, PLAINER_PARAGRAPHS[text])

    doc.core_properties.author = "Mrityunjay Gupta"
    doc.core_properties.subject = "Full-depth IEEE-style research manuscript"
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
