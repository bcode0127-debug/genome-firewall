"""GENOME FIREWALL — Streamlit decision-support prototype.

HARD CONSTRAINT (verified by inspection): this app reads ONLY these committed
files and never touches data/raw/, data/interim/, or data/features/:
  - models/meropenem.joblib, models/ceftazidime.joblib, models/gentamicin.joblib
  - models/conformal.json
  - data/drug_properties.yaml
  - data/results/metrics.json
  - data/results/novelty.json
  - data/results/demo_genomes.parquet
"""
import json
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

DRUGS = ["meropenem", "ceftazidime", "gentamicin"]
POSITIVE_CLASS = "Resistant"
ST258 = "MLST.Klebsiella_pneumoniae.258"
JACCARD_THRESH = 0.1570
ABSENT_THRESH = 15
LAB_BANNER = ("⚠️ **Every result must be confirmed by standard laboratory "
              "testing.** This is research-prototype decision support, not a "
              "diagnostic device.")
COVERAGE_STATEMENT = (
    "**Coverage:** *Klebsiella pneumoniae* only; meropenem, ceftazidime, and "
    "gentamicin only. This system does not cover other species or the other 71 "
    "antibiotics in the source database."
)

MODEL_PATHS = {d: f"models/{d}.joblib" for d in DRUGS}
CONFORMAL_PATH = "models/conformal.json"
DRUG_PROPS_PATH = "data/drug_properties.yaml"
METRICS_PATH = "data/results/metrics.json"
NOVELTY_PATH = "data/results/novelty.json"
DEMO_PATH = "data/results/demo_genomes.parquet"

FEATURE_START_COL = 7  # demo parquet: first 7 cols are metadata, rest are features


# ----------------------------- data loading -----------------------------

@st.cache_resource
def load_models():
    return {d: joblib.load(MODEL_PATHS[d]) for d in DRUGS}


@st.cache_data
def load_json(path):
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_drug_props():
    with open(DRUG_PROPS_PATH) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_demo():
    df = pd.read_parquet(DEMO_PATH)
    df["genome_id"] = df["genome_id"].astype(str)
    return df


@st.cache_data
def feature_columns():
    return list(load_demo().columns[FEATURE_START_COL:])


def inner_logreg(model):
    """Reach the LogisticRegression under CalibratedClassifierCV(FrozenEstimator)."""
    return model.calibrated_classifiers_[0].estimator.estimator


# ----------------------------- core logic -----------------------------

def compile_determinants(drug_props, drug):
    return [re.compile(p, re.IGNORECASE)
            for p in drug_props[drug]["resistance_determinant_patterns"]]


def determinant_hits(ar_genes, patterns):
    """Word-boundary determinant match against a genome's AR gene list."""
    hits = []
    for gene in ar_genes:
        if any(p.search(str(gene)) for p in patterns):
            hits.append(str(gene))
    return sorted(set(hits))


def verdict_and_confidence(model, feature_vec, conformal, drug):
    """Returns (verdict, prob_resistant, is_no_call, reason)."""
    prob1 = float(model.predict_proba(feature_vec.reshape(1, -1))[:, 1][0])
    q_r = conformal[drug]["q_resistant"]
    q_s = conformal[drug]["q_susceptible"]
    include_r = prob1 >= (1 - q_r)
    include_s = (1 - prob1) >= (1 - q_s)
    set_size = int(include_r) + int(include_s)
    if set_size == 1:
        if include_r:
            return "LIKELY TO FAIL", prob1, False, None  # resistant -> drug fails
        return "LIKELY TO WORK", prob1, False, None      # susceptible -> drug works
    if set_size == 0:
        return ("NO-CALL", prob1, True,
                "Conformal prediction set is EMPTY — neither class met its "
                f"class-conditional confidence threshold (alpha={conformal[drug]['alpha']}).")
    return ("NO-CALL", prob1, True,
            "Conformal prediction set contains BOTH labels — evidence is ambiguous.")


def top_contributing_features(model, feature_vec, feature_cols, k=8):
    """Per-genome contributions = coef * feature_value for present features."""
    lr = inner_logreg(model)
    coef = lr.coef_[0]
    present = np.where(feature_vec == 1)[0]
    contribs = [(feature_cols[i], float(coef[i])) for i in present]
    # rank by absolute contribution toward the resistant class
    contribs.sort(key=lambda x: abs(x[1]), reverse=True)
    return contribs[:k]


def evidence_category(ar_genes, det_patterns, model, feature_vec, feature_cols, det_hit_list):
    """(i) known determinant, (ii) statistical association only, (iii) no signal."""
    if det_hit_list:
        return ("i", "Known resistance determinant detected",
                f"A gene in this genome matches this drug's known-determinant "
                f"list: {', '.join(det_hit_list)}.")
    # any AR gene present at all?
    top = top_contributing_features(model, feature_vec, feature_cols, k=8)
    resistant_leaning = [(f, w) for f, w in top if w > 0]
    if resistant_leaning:
        names = ", ".join(f"{f}" for f, _ in resistant_leaning[:5])
        return ("ii", "Statistical association only",
                f"No known determinant for this drug matched, but the model's top "
                f"resistance-leaning features for this genome are: {names}. These are "
                f"statistical associations, not curated causal determinants.")
    return ("iii", "No known resistance signal found",
            "No known determinant matched and no resistance-leaning features are "
            "present in this genome.")


NOVELTY_LIFT_TEXT = {
    "meropenem": "measured error-rate lift on flagged genomes: 0.067 → 0.147.",
    "gentamicin": "measured error-rate lift on flagged genomes: 0.137 → 0.213.",
    "ceftazidime": ("the novelty flag showed NO discrimination for this drug "
                    "(flagged error rate 0.000 vs 0.077 unflagged) — treat as "
                    "advisory noise here."),
}


# ----------------------------- OpenAI explainer -----------------------------

def openai_explanation(drug, verdict, confidence, evidence_label, gene_names, category):
    """Additive narration. Sends STRUCTURED RESULT ONLY. Never raises."""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, timeout=8.0)
        payload = {
            "drug": drug, "verdict": verdict,
            "calibrated_confidence": round(confidence, 3),
            "evidence_category": evidence_label,
            "supporting_genes": gene_names,
        }
        if category == "ii":
            # No known determinant matched. Do NOT let the model invent a
            # mechanism — the listed genes are statistical associations only.
            prompt = (
                "You are given a FIXED, already-decided antibiotic-susceptibility "
                "prediction as structured JSON. Do NOT change or second-guess the "
                "verdict. This is evidence category (ii): NO known resistance "
                "determinant for this drug was found. In exactly two sentences, "
                "state that no known determinant matched and that the listed genes "
                "are statistical associations with no established causal role for "
                "this drug. Do NOT describe a biological mechanism and do NOT imply "
                "the listed genes cause resistance to this drug. Structured "
                "result:\n" + json.dumps(payload)
            )
        else:
            prompt = (
                "You are given a FIXED, already-decided antibiotic-susceptibility "
                "prediction as structured JSON. Do NOT change or second-guess the "
                "verdict. In exactly two sentences, explain the likely biological "
                "mechanism in plain language for a clinician. Structured result:\n"
                + json.dumps(payload)
            )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=140,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


# ----------------------------- UI helpers -----------------------------

def verdict_color(verdict):
    return {"LIKELY TO FAIL": "#c0392b", "LIKELY TO WORK": "#1e8449",
            "NO-CALL": "#7f8c8d"}.get(verdict, "#7f8c8d")


def parse_amrfinder_tsv(uploaded, feature_cols):
    """Map an AMRFinderPlus TSV to the model's feature vector by family token.
    Returns (feature_vec, ar_gene_list). Presence/absence over the model vocab.
    """
    df = pd.read_csv(uploaded, sep="\t")
    symcol = next((c for c in df.columns
                   if c.lower() in ("element symbol", "gene symbol", "gene")), None)
    namecol = next((c for c in df.columns if c.lower() == "element name"), None)
    subcol = next((c for c in df.columns if c.lower() == "subtype"), None)
    if symcol is None:
        return None, None
    rows = df
    if subcol is not None:
        rows = df[df[subcol] == "AMR"]
    symbols = [str(s) for s in rows[symcol].dropna().tolist()]

    def norm(s):
        s = re.split(r"[_]", s.strip())[0]
        s = re.sub(r"(?i)^bla", "", s)
        s = re.sub(r"[^A-Za-z0-9]", "", s).lower()
        return re.sub(r"\d+$", "", s)

    norm_syms = {norm(s) for s in symbols}
    vec = np.zeros(len(feature_cols), dtype=float)
    for i, col in enumerate(feature_cols):
        col_norm = re.sub(r"[^a-z0-9]", "", col.lower())
        if any(ns and ns in col_norm for ns in norm_syms):
            vec[i] = 1.0

    # Determinant-facing gene names, built directly from the TSV. Strip the
    # 'bla' beta-lactamase prefix so the SAME word-boundary determinant regexes
    # (e.g. \bkpc\b) match AMRFinderPlus's 'blaKPC-3' style symbols — without
    # this, 'blaKPC-3' has no word boundary before 'KPC' and every uploaded
    # beta-lactamase silently fell through to evidence category (ii). Fall back
    # to the Element name column when a row's symbol is blank.
    gene_names = []
    for _, r in rows.iterrows():
        sym = str(r[symcol]).strip() if pd.notna(r[symcol]) else ""
        if sym and sym.lower() != "nan":
            gene_names.append(re.sub(r"(?i)^bla", "", sym))
        elif namecol is not None and pd.notna(r[namecol]):
            gene_names.append(str(r[namecol]).strip())
    return vec, gene_names


# ----------------------------- TAB 1: Report -----------------------------

def tab_report(models, conformal, drug_props, demo, feature_cols):
    st.subheader("Antibiotic-response report")
    st.info(LAB_BANNER)
    st.markdown(COVERAGE_STATEMENT)

    st.sidebar.header("Input")
    mode = st.sidebar.radio("Source", ["Demo genome", "Upload AMRFinderPlus TSV"])

    feature_vec = None
    ar_genes = []
    meta = {}

    if mode == "Demo genome":
        gid = st.sidebar.selectbox("Genome", demo["genome_id"].tolist())
        row = demo[demo["genome_id"] == gid].iloc[0]
        feature_vec = row[feature_cols].values.astype(float)
        ar_genes = list(row["ar_genes"])
        meta = {"genome_id": gid, "mlst": row["mlst"], "flagged": bool(row["flagged_novel"]),
                "truth": {d: row[d] for d in DRUGS}}
        badges = []
        if row["mlst"] == ST258:
            badges.append("🧬 **ST258 — held-out clone** (never seen in training)")
        if row["flagged_novel"]:
            badges.append("🔬 **unlike training data** (novelty-flagged; advisory only)")
        if badges:
            st.sidebar.markdown("  \n".join(badges))
        st.sidebar.caption(f"MLST: {row['mlst']}")
    else:
        st.sidebar.markdown(
            "Upload an **AMRFinderPlus TSV** (Module 01 output). Raw FASTA is "
            "out of scope for the hosted app — run FASTA→TSV locally with the "
            "`amrfinder --nucleotide ...` command in `context/feature_spec.md`."
        )
        st.sidebar.caption(
            "No TSV handy? Two examples ship in `examples/` in the repo.")
        up = st.sidebar.file_uploader("AMRFinderPlus .tsv", type=["tsv", "txt"])
        if up is not None:
            feature_vec, ar_genes = parse_amrfinder_tsv(up, feature_cols)
            if feature_vec is None:
                st.error("Could not find an 'Element symbol' column in that TSV.")
                return
            meta = {"genome_id": up.name, "mlst": "(uploaded)", "flagged": False,
                    "truth": None}

    if feature_vec is None:
        st.warning("Pick a demo genome or upload an AMRFinderPlus TSV to begin.")
        return

    st.markdown(f"**Genome:** `{meta['genome_id']}`  ·  MLST: `{meta['mlst']}`  ·  "
                f"{len(ar_genes)} AR gene annotations")

    metrics = load_json(METRICS_PATH)

    for drug in DRUGS:
        model = models[drug]
        det_patterns = compile_determinants(drug_props, drug)
        det_hits = determinant_hits(ar_genes, det_patterns)

        verdict, prob1, is_no_call, reason = verdict_and_confidence(
            model, feature_vec, conformal, drug)
        cat, cat_label, cat_detail = evidence_category(
            ar_genes, det_patterns, model, feature_vec, feature_cols, det_hits)

        # confidence = calibrated prob of the called class
        if verdict == "LIKELY TO FAIL":
            confidence = prob1
        elif verdict == "LIKELY TO WORK":
            confidence = 1 - prob1
        else:
            confidence = max(prob1, 1 - prob1)

        with st.container(border=True):
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(
                    f"### {drug.capitalize()} &nbsp; "
                    f"<span style='color:{verdict_color(verdict)};font-weight:700'>"
                    f"{verdict}</span>", unsafe_allow_html=True)
                st.markdown(f"**Evidence category ({cat}):** {cat_label}")
                st.caption(cat_detail)
                if det_hits:
                    st.markdown(f"**Supporting determinant genes:** "
                                f"{', '.join(det_hits)}")
                if is_no_call:
                    st.markdown(f"**No-call reason:** {reason}")
            with c2:
                if not is_no_call:
                    st.metric("Calibrated confidence", f"{confidence:.1%}")
                st.caption(f"P(resistant) = {prob1:.3f}  ·  "
                           f"alpha = {conformal[drug]['alpha']}")
                if meta.get("truth"):
                    st.caption(f"(held-out truth: {meta['truth'][drug]})")

            # Novelty: advisory only, never overrides
            if meta.get("flagged"):
                st.warning(f"**Novelty caution (advisory only):** this genome is "
                           f"unlike the training data. For {drug}, "
                           f"{NOVELTY_LIFT_TEXT[drug]} This does **not** override "
                           f"the verdict above.")

            # OpenAI explainer — additive, after verdict is fixed
            narration = openai_explanation(
                drug, verdict, confidence, cat_label, det_hits or [f for f, _ in
                top_contributing_features(model, feature_vec, feature_cols, 3)], cat)
            if narration:
                st.markdown("**AI-generated explanation** "
                            "*(describes the result; does not produce it)*")
                st.caption(narration)

    st.divider()
    st.info(LAB_BANNER)


# ----------------------------- TAB 2: Performance -----------------------------

def tab_performance(metrics, conformal):
    st.subheader("Held-out test-set performance")
    st.caption("Positive class = **Resistant** throughout. Test set = genomes in "
               "MLST sequence types held out entirely from training (ST258 included).")

    for drug in DRUGS:
        d = metrics["drugs"][drug]
        o = d["test_overall"]
        c = d["conformal"]
        st.markdown(f"#### {drug.capitalize()}  ·  n = {o['n']}")
        cols = st.columns(4)
        cols[0].metric("Balanced acc.", f"{o['balanced_accuracy']:.3f}")
        cols[1].metric("AUROC", f"{o['auroc']:.3f}")
        cols[2].metric("PR-AUC (Resistant)", f"{o['pr_auc_positive_class_resistant']:.3f}")
        cols[3].metric("Brier", f"{o['brier']:.3f}")
        cols = st.columns(4)
        cols[0].metric("Recall (Resistant)", f"{o['recall_resistant']:.3f}")
        cols[1].metric("Recall (Susceptible)", f"{o['recall_susceptible']:.3f}")
        cols[2].metric("F1 (Resistant)", f"{o['f1_resistant_positive']:.3f}")
        cols[3].metric("No-call rate", f"{c['no_call_rate']:.3f}")

        cov = c["coverage_overall"]
        st.markdown(
            f"**Conformal coverage** (alpha = {d['alpha']}, nominal = "
            f"{c['nominal']:.2f}): **{cov:.3f}** overall "
            f"(Resistant {c['coverage_resistant']:.3f}, "
            f"Susceptible {c['coverage_susceptible']:.3f})."
        )
        if drug == "gentamicin":
            st.caption("Gentamicin abstains on 28.9% of cases (225/779) at "
                       "alpha=0.05 (chosen empirically). Overall coverage is 0.891, "
                       "still short of the 0.90 nominal. Most abstentions come from "
                       "a single isotonic plateau at P=0.227 inside the both-labels "
                       "band — a property of the calibration curve, not per-case "
                       "uncertainty. Excluding both-label abstentions, singleton "
                       "coverage is 0.847.")

        # reliability plot
        rel = d["reliability"]
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
        ax.plot([b["mean_pred"] for b in rel], [b["obs_freq"] for b in rel],
                "o-", color="#2c3e50", label="model")
        ax.set_xlabel("Mean predicted P(resistant)")
        ax.set_ylabel("Observed fraction resistant")
        ax.set_title(f"{drug} reliability (test)")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        st.pyplot(fig)
        plt.close(fig)

        # per-group table
        st.markdown("**Per-MLST-group (test), ST258 first:**")
        by_group = d["test_by_group"]
        ordered = ([ST258] if ST258 in by_group else []) + \
                  sorted([g for g in by_group if g != ST258],
                         key=lambda g: -by_group[g]["n"])
        rows = []
        for g in ordered:
            gm = by_group[g]
            near_single = (gm["recall_resistant"] in (0.0, 1.0)) or \
                          (gm["recall_susceptible"] in (0.0, 1.0))
            rows.append({
                "MLST": g.replace("MLST.Klebsiella_pneumoniae.", "ST"),
                "n": gm["n"],
                "recall_R": round(gm["recall_resistant"], 3),
                "recall_S": round(gm["recall_susceptible"], 3),
                "bal_acc": round(gm["balanced_accuracy"], 3),
                "AUROC": None if gm["auroc"] is None else round(gm["auroc"], 3),
                "note": "near-single-class: bal_acc/AUROC not meaningful; read raw recall & n"
                        if near_single else "",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.divider()


# ----------------------------- TAB 3: Responsibility -----------------------------

def tab_responsibility(metrics):
    st.subheader("Responsible-AI requirements")

    st.markdown("#### 1. Defensive by construction")
    st.markdown(
        "There is **no generative or sequence-design component anywhere in the "
        "prediction path**. The system maps an existing genome's AMR gene "
        "presence/absence to a susceptibility verdict. It never designs, "
        "modifies, or suggests changes to an organism.")

    st.markdown("#### 2. Honest generalization")
    st.markdown(
        "Train/calibration/test are split by **MLST sequence type (GroupKFold)**. "
        "ST258 is held out by sequence type. Its clonal-complex relative **ST437 "
        "remains in training** at a minimum Mash distance of **0.0034** — closer "
        "than our own within-ST maximum of **0.0081**. MLST grouping does not "
        "separate single-locus variants within a clonal complex, and our own "
        "validation records `mlst_grouping_consistent: false`. At our recommended "
        "0.00576 threshold, ~200 ST437/ST258 pairs cross the split. Meropenem "
        "still holds on ST258 (bal. acc. 0.901, AUROC 0.890); **ceftazidime does "
        "not (0.484 / 0.542)**, where the clone is ~97% resistant and the model "
        "rides that base rate. Per-group results are on the Performance tab.")

    st.markdown("#### 3. Calibrated confidence and honest no-call")
    rows = []
    for drug in DRUGS:
        d = metrics["drugs"][drug]
        c = d["conformal"]
        rows.append({"drug": drug, "alpha": d["alpha"],
                     "coverage": round(c["coverage_overall"], 3),
                     "nominal": c["nominal"], "no_call_rate": round(c["no_call_rate"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown(
        "Isotonic calibration + class-conditional conformal prediction. "
        "Split conformal's guarantee assumes calibration and test are "
        "**exchangeable**; our MLST group split deliberately breaks that. "
        "Class-conditional coverage **misses on every drug** (meropenem "
        "susceptible 0.836, ceftazidime susceptible 0.822, gentamicin resistant "
        "0.860). We report the **measured** coverage, not the theoretical "
        "guarantee.")
    st.markdown(
        "**Gentamicin** abstains on **28.9% of cases (225/779)** at alpha=0.05, "
        "chosen empirically rather than derived. Overall coverage is **0.891, "
        "still short of 0.90**. Most of those abstentions come from a single "
        "isotonic calibration plateau at P=0.227 falling inside the both-labels "
        "band, so this is a property of the calibration curve, not per-case "
        "uncertainty. Excluding both-label abstentions, singleton coverage is "
        "**0.847**.")

    st.markdown("#### 4. Honest explanations")
    st.markdown(
        "Each result is labelled **(i) known determinant**, **(ii) statistical "
        "association only**, or **(iii) no known signal**. Feature importance is "
        "**associational, not causal** — the app says so on every category-(ii) "
        "result.")

    st.markdown("#### 5. Human oversight")
    st.markdown("A persistent banner states **every result must be confirmed by "
                "standard laboratory testing**.")

    st.divider()
    st.markdown("#### Novelty-gate validation — where it works and where it does not")
    rows = []
    for drug in DRUGS:
        nv = metrics["drugs"][drug]["novelty_validation"]
        rows.append({
            "drug": drug, "n_flagged": nv["n_flagged"],
            "err_flagged": None if nv["err_flagged"] is None else round(nv["err_flagged"], 3),
            "err_not_flagged": None if nv["err_not_flagged"] is None else round(nv["err_not_flagged"], 3),
            "errors_caught": f"{nv['errors_caught']}/{nv['total_errors']}",
            "discriminative": "yes" if nv["discriminative"] else "NO",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("The novelty flag is directionally useful for meropenem and "
               "gentamicin but showed NO discrimination for ceftazidime. It is "
               "advisory only and never overrides a verdict.")

    st.divider()
    st.markdown("#### Documented limitations")
    st.markdown(
        "- **Gene-presence encoding only.** Point substitutions (e.g. `gyrA` QRDR "
        "mutations) are **not featurized**; loss-of-function still appears as gene "
        "absence. Resistance driven purely by point mutations may be missed.\n"
        "- **Annotation provenance.** AMRFinderPlus was run end-to-end on a "
        "20-genome validation subset; the full 3,342-genome training matrix uses "
        "CARD/PATRIC specialty-gene annotations at scale (stated plainly, not "
        "presented as AMRFinderPlus output).\n"
        "- **Target-presence gate.** Implemented and validated offline "
        "(`scripts/gates_and_diagnostics.py`): it forces a NO-CALL when a drug's "
        "molecular target is absent. It **fires on 0/779 test genomes** because "
        "all three targets (PBPs, 16S rRNA) are essential genes present in every "
        "genome, so it is not wired into the live upload path.")

    st.divider()
    st.markdown("#### What we could not validate")
    st.markdown(
        "All reported metrics are computed on curated (CARD/PATRIC) annotations. "
        "The upload path consumes AMRFinderPlus output, and the two annotation "
        "sources share only **10.6% of gene names** on our 20-genome comparison. "
        "**No genome was scored end-to-end** from FASTA through AMRFinderPlus to "
        "verdict against its true phenotype. Our metrics describe the **demo "
        "path**; the **upload path is documented but unmeasured**.")


# ----------------------------- TAB 4: How it works -----------------------------

def tab_how(metrics):
    st.subheader("How it works")
    st.markdown(
        """
```
FASTA (assembled genome, out of scope for hosted app)
   │   amrfinder --nucleotide  (Module 01, runs locally)
   ▼
AMR gene presence/absence features  ─►  vocab_model.json (422 columns, fixed order)
   │
   ▼
Logistic regression per drug  (L2, C tuned by GroupKFold on MLST)
   │
   ▼
Isotonic calibration   (fit on the held-out calibration split)
   │
   ▼
Class-conditional conformal prediction   (per-drug alpha → prediction set)
   │
   ├─►  singleton set  →  LIKELY TO WORK / LIKELY TO FAIL + calibrated confidence
   └─►  empty / both   →  NO-CALL
   │
   ▼
Deterministic novelty gate (advisory only — never overrides)
   │
   ▼
Decision report  +  evidence category (i/ii/iii)  +  supporting genes
```
(A target-presence gate is implemented and validated offline but is **not**
wired into this upload path — see the Responsibility tab for why.)
        """)
    st.success(
        "**No language model touches the prediction path.** The optional OpenAI "
        "explainer on the Report tab runs *after* the verdict is fixed, receives "
        "only the structured result, and renders plain-language narration. If the "
        "key is missing or the call fails, the full structured report renders "
        "unchanged.")


# ----------------------------- TAB 5: Surveillance -----------------------------

def tab_surveillance(demo, drug_props):
    st.subheader("Resistance surveillance (public-health tracking view)")
    st.caption("Across the 40-genome demo cohort. This is the aggregate lineage "
               "view — how resistance and its determinants distribute across "
               "MLST sequence types.")

    demo = demo.copy()
    demo["ST"] = demo["mlst"].str.replace("MLST.Klebsiella_pneumoniae.", "ST", regex=False)
    demo["ST"] = demo["ST"].fillna("untyped")

    st.markdown("#### Resistance rate per drug, by sequence type")
    rows = []
    for st_name, sub in demo.groupby("ST"):
        row = {"ST": st_name, "n": len(sub)}
        for drug in DRUGS:
            r = (sub[drug] == "Resistant").mean()
            row[drug] = round(float(r), 2)
        rows.append(row)
    surv = pd.DataFrame(rows).sort_values("n", ascending=False)
    st.dataframe(surv, hide_index=True, width="stretch")

    st.markdown("#### Which determinants cluster in which lineage")
    det_rows = []
    for st_name, sub in demo.groupby("ST"):
        fams = {}
        for drug in DRUGS:
            patterns = compile_determinants(drug_props, drug)
            for _, r in sub.iterrows():
                for hit in determinant_hits(list(r["ar_genes"]), patterns):
                    fams[hit] = fams.get(hit, 0) + 1
        if fams:
            top = sorted(fams.items(), key=lambda x: -x[1])[:4]
            det_rows.append({"ST": st_name, "n": len(sub),
                             "top determinants": ", ".join(f"{k}×{v}" for k, v in top)})
    st.dataframe(pd.DataFrame(det_rows), hide_index=True, width="stretch")

    st.markdown("#### Multi-drug-resistant genomes")
    def n_resistant(r):
        return sum(1 for d in DRUGS if r[d] == "Resistant")
    demo["n_resistant"] = demo.apply(n_resistant, axis=1)
    mdr = int((demo["n_resistant"] == 3).sum())
    any_r = int((demo["n_resistant"] >= 1).sum())
    st.metric("Resistant to all 3 drugs", f"{mdr} / {len(demo)}")
    st.caption(f"{any_r}/{len(demo)} are resistant to at least one drug; "
               f"{mdr}/{len(demo)} to all three (meropenem + ceftazidime + gentamicin).")


# ----------------------------- main -----------------------------

def main():
    st.set_page_config(page_title="Genome Firewall", page_icon="🧬", layout="wide")
    st.title("🧬 Genome Firewall")
    st.caption("Research prototype — genotype-to-antibiotic-response decision "
               "support for *Klebsiella pneumoniae*.")

    models = load_models()
    conformal = load_json(CONFORMAL_PATH)
    drug_props = load_drug_props()
    metrics = load_json(METRICS_PATH)
    demo = load_demo()
    feature_cols = feature_columns()

    t1, t2, t3, t4, t5 = st.tabs(
        ["Report", "Performance", "Responsibility", "How it works", "Surveillance"])
    with t1:
        tab_report(models, conformal, drug_props, demo, feature_cols)
    with t2:
        tab_performance(metrics, conformal)
    with t3:
        tab_responsibility(metrics)
    with t4:
        tab_how(metrics)
    with t5:
        tab_surveillance(demo, drug_props)


if __name__ == "__main__":
    main()
