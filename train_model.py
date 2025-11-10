import os, re, json, unicodedata, pandas as pd, numpy as np, joblib
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR  = os.path.join(BASE_DIR, "csv")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# --------- Inputs ---------
PRIMARY_CSV      = os.path.join(DATA_DIR, "training.csv")                  # text,label,[stars]
AUG_REVIEWS_CSV  = os.path.join(CSV_DIR,  "tripadvisor_reviews_updated.csv")# optional
EMOJI_CSV        = os.path.join(CSV_DIR,  "full_emoji.csv")
SUGGESTIONS_CSV  = os.path.join(CSV_DIR,  "suggestions_dataset.csv")

# --------- Outputs ---------
VECT_PATH        = os.path.join(DATA_DIR, "vectorizer.pkl")
CLS_PATH         = os.path.join(DATA_DIR, "model.pkl")
REG_PATH         = os.path.join(DATA_DIR, "star_model.pkl")
LABELS_JSON      = os.path.join(DATA_DIR, "labels.json")
EMOJI_JSON       = os.path.join(DATA_DIR, "emoji_map.json")
SUGGESTIONS_JSON = os.path.join(DATA_DIR, "suggestions.json")
PREPARED_CSV     = os.path.join(DATA_DIR, "training_prepared.csv")

DEFAULT_LABELS = ["negative","neutral","positive"]

# ---------- Utils ----------
def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def basic_normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Emoji ----------
def load_emoji_map(path: str) -> Dict[str,str]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "emoji" not in cols or "name" not in cols:
        # try alternate guesses
        cand_emoji = next((c for c in df.columns if "emoji" in c.lower()), None)
        cand_name  = next((c for c in df.columns if "name"  in c.lower()), None)
        if not (cand_emoji and cand_name):
            return {}
        cols["emoji"], cols["name"] = cand_emoji, cand_name

    mp = {}
    for _, row in df[[cols["emoji"], cols["name"]]].dropna().iterrows():
        ch = str(row[cols["emoji"]])
        nm = f"emoji_{_slug(str(row[cols['name']]))}"
        if ch and nm:
            mp[ch] = nm
    return mp

def replace_emojis(text: str, emoji_map: Dict[str,str]) -> str:
    if not emoji_map or not text:
        return text or ""
    keys = sorted(emoji_map.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(map(re.escape, keys))) if keys else None
    return pattern.sub(lambda m: " " + emoji_map.get(m.group(0), "") + " ", text) if pattern else text

# ---------- Suggestions (flexible) ----------
def canon(name: str) -> str:
    n = (name or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "", n)

SYNONYMS = {
    "topic": {"topic","category","issue","issuetype"},
    "keywords": {"keywords","keyword","key_words","keys","phrases","triggers"},
    "suggested_action": {"suggestedaction","suggestedactions","action","actions","recommendation","recommendations","suggestion"},
    "severity": {"severity","impact","priority","level"},
    "follow_up_metric": {"followupmetric","follow_up_metric","followup","metric","kpi","measure","followupkpi","followupmeasure"},
}

def build_suggestions_json(csv_path: str, out_path: str) -> int:
    if not os.path.exists(csv_path):
        # write empty file so backend handles gracefully
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        return 0
    df = pd.read_csv(csv_path)
    raw = list(df.columns)
    canon_map = {canon(c): c for c in raw}
    cols = {}
    for need, alts in SYNONYMS.items():
        hit = None
        for a in alts:
            if a in canon_map:
                hit = canon_map[a]; break
        if not hit:
            # closest contains-match
            for k, v in canon_map.items():
                if any(a in k for a in alts):
                    hit = v; break
        if not hit:
            raise ValueError(f"CSV missing a required column for '{need}'. Headers found: {raw}")
        cols[need] = hit

    def parse_keywords(cell: str):
        s = str(cell or "").strip()
        if not s: return []
        # Prefer ';' then ','
        parts = s.replace("|",";")
        sep = ";" if ";" in parts else ","
        return [p.strip() for p in parts.split(sep) if p.strip()]

    rows = []
    for _, r in df[[cols["topic"], cols["keywords"], cols["suggested_action"], cols["severity"], cols["follow_up_metric"]]].fillna("").iterrows():
        rows.append({
            "topic": str(r[cols["topic"]]).strip(),
            "keywords": parse_keywords(r[cols["keywords"]]),
            "suggested_action": str(r[cols["suggested_action"]]).strip(),
            "severity": str(r[cols["severity"]]).strip().lower(),
            "follow_up_metric": str(r[cols["follow_up_metric"]]).strip(),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    return len(rows)

# ---------- Data loading ----------
def load_primary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["text","label","stars"])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    text_col  = cols.get("text")  or cols.get("review") or cols.get("review_text") or list(df.columns)[0]
    label_col = cols.get("label")
    star_col  = cols.get("stars") or cols.get("rating")
    out = pd.DataFrame({
        "text":  df[text_col].astype(str),
        "label": df[label_col].astype(str) if label_col in df.columns else pd.NA,
        "stars": df[star_col] if star_col in df.columns else pd.NA
    })
    return out

def load_tripadvisor(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["text","label","stars"])
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    txt = cols.get("review text") or cols.get("review_text") or cols.get("text")
    rat = cols.get("rating") or cols.get("stars")
    if not txt or not rat:
        return pd.DataFrame(columns=["text","label","stars"])
    tmp = pd.DataFrame({
        "text":  df[txt].astype(str),
        "stars": pd.to_numeric(df[rat], errors="coerce")
    })
    def to_label(x):
        if pd.isna(x): return pd.NA
        x = float(x)
        if x <= 2.0:  return "negative"
        if x >= 4.0:  return "positive"
        return "neutral"
    tmp["label"] = tmp["stars"].map(to_label)
    return tmp

def prepare_data(emoji_map: Dict[str,str]) -> pd.DataFrame:
    a = load_primary(PRIMARY_CSV)
    b = load_tripadvisor(AUG_REVIEWS_CSV)
    df = pd.concat([a, b], ignore_index=True)
    df["text"] = df["text"].astype(str).map(lambda s: basic_normalize(replace_emojis(s, emoji_map)))
    df = df.dropna(subset=["text"]).copy()
    if df["label"].notna().any():
        df["label"] = df["label"].str.lower().str.strip()
        df = df[df["label"].isin(DEFAULT_LABELS)]
    if "stars" in df.columns:
        df["stars"] = pd.to_numeric(df["stars"], errors="coerce").clip(1,5)
    # Export prepared copy for debugging
    df.to_csv(PREPARED_CSV, index=False)
    return df

# ---------- Train ----------
def train():
    # Build emoji map
    emoji_map = load_emoji_map(EMOJI_CSV)
    with open(EMOJI_JSON, "w", encoding="utf-8") as f:
        json.dump(emoji_map, f, ensure_ascii=False, indent=2)

    # Build suggestions.json (flexible)
    try:
        n_rules = build_suggestions_json(SUGGESTIONS_CSV, SUGGESTIONS_JSON)
        print(f"[Suggestions] Wrote {n_rules} rules to {SUGGESTIONS_JSON}")
    except Exception as e:
        print(f"[Suggestions] WARNING: {e}. Writing empty suggestions.json")
        with open(SUGGESTIONS_JSON, "w", encoding="utf-8") as f:
            json.dump([], f)

    # Prepare data
    df = prepare_data(emoji_map)
    if df.empty:
        raise SystemExit("No training data found.")

    # Ensure labels
    if not df["label"].notna().any():
        if not df["stars"].notna().any():
            raise SystemExit("No labels or stars available to derive labels.")
        def to_label(x):
            if x <= 2.0:  return "negative"
            if x >= 4.0:  return "positive"
            return "neutral"
        df["label"] = df["stars"].map(to_label)

    # Save label order
    with open(LABELS_JSON, "w") as f:
        json.dump(DEFAULT_LABELS, f)

    # Train classifier
    clf_df = df.dropna(subset=["text","label"]).copy()
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        clf_df["text"], clf_df["label"], test_size=0.2, random_state=42, stratify=clf_df["label"]
    )

    vect = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=150_000, lowercase=True)
    Xc_tr = vect.fit_transform(Xc_train)
    Xc_te = vect.transform(Xc_test)

    base = LinearSVC()
    cls  = CalibratedClassifierCV(base, cv=5)
    cls.fit(Xc_tr, yc_train)

    y_pred = cls.predict(Xc_te)
    print("\n[Classifier]\n", classification_report(yc_test, y_pred, digits=3))

    joblib.dump(vect, VECT_PATH)
    joblib.dump(cls,  CLS_PATH)

    # Train regressor if stars exist
    if df["stars"].notna().any():
        reg_df = df.dropna(subset=["text","stars"]).copy()
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            reg_df["text"], reg_df["stars"], test_size=0.2, random_state=42
        )
        Xr_tr = vect.transform(Xr_train)
        Xr_te = vect.transform(Xr_test)
        reg = LinearSVR(C=1.0)
        reg.fit(Xr_tr, yr_train)
        mae = mean_absolute_error(reg.predict(Xr_te), yr_test)
        print(f"\n[Regressor] MAE ≈ {mae:.3f} stars")
        joblib.dump(reg, REG_PATH)
    else:
        print("\n[Regressor] Skipped (no stars in data).")

    print("\nSaved artifacts:")
    for p in [VECT_PATH, CLS_PATH, REG_PATH, LABELS_JSON, EMOJI_JSON, SUGGESTIONS_JSON, PREPARED_CSV]:
        if os.path.exists(p): print(" -", p)

if __name__ == "__main__":
    train()
