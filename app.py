from flask import Flask, render_template, request, jsonify, send_file
from flask.json.provider import DefaultJSONProvider
import pandas as pd
import requests
import io
import time
import numpy as np
import os


class SafeJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return super().default(obj)


app = Flask(__name__)
app.json = SafeJSONProvider(app)

# Temporary: direct key yahan daal sakti ho
# API_KEY = "gsk_your_new_key_here"

# Better: env variable se lo

API_KEY = os.getenv("GROQ_API_KEY", "")

MODEL_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "llama-3.1-8b-instant"

df_store = {}
last_filtered = {}
last_analysis = {}


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def read_uploaded_file(file):
    filename = (file.filename or "").lower()

    if filename.endswith(".csv"):
        content = file.read().decode("utf-8", errors="ignore")
        return pd.read_csv(io.StringIO(content))

    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        data = file.read()
        return pd.read_excel(io.BytesIO(data))

    raise ValueError("Please upload a CSV or Excel file (.csv, .xlsx, .xls)")


def try_parse_dates(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(20)
            if sample.empty:
                continue

            date_hint = any(
                x in col.lower()
                for x in ["date", "time", "month", "year", "created", "updated", "start", "end", "joined", "dob"]
            )

            if date_hint:
                try:
                    parsed = pd.to_datetime(df[col], errors="coerce")
                    if parsed.notna().sum() >= max(1, len(df) // 5):
                        df[col] = parsed
                except Exception:
                    pass
    return df


def call_ai(system, user, max_tokens=400):
    if not API_KEY:
        return "AI is disabled because API key is missing."

    if len(user) > 2800:
        user = user[:2800] + "\n...(truncated)"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }

    for attempt in range(3):
        try:
            r = requests.post(MODEL_URL, headers=headers, json=payload, timeout=30)

            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()

            if r.status_code == 429:
                time.sleep(4 + attempt * 3)
                continue

            try:
                return f"Error: {r.json().get('error', {}).get('message', 'Unknown')}"
            except Exception:
                return f"Error: HTTP {r.status_code}"

        except Exception as e:
            if attempt < 2:
                time.sleep(3)
                continue
            return f"Error: {str(e)}"

    return "Rate limit reached. Wait a few seconds and retry."


def smart_detect(df):
    cl = {c: c.lower().replace(" ", "").replace("_", "") for c in df.columns}
    num = df.select_dtypes(include="number").columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    kw = {
        "value": ["dealvalue", "revenue", "sales", "amount", "value", "salary", "monthlysalary", "wage", "income", "price", "cost", "total", "profit"],
        "discount": ["discount", "disc", "reduction", "rebate"],
        "booking": ["booking", "advance", "deposit", "payment", "booked"],
        "branch": ["branch", "region", "city", "location", "area", "zone", "territory", "office", "state"],
        "emp": ["employee", "emp", "staff", "salesperson", "agent", "salesemployee", "person", "executive", "seller", "staffid"],
        "status": ["status", "dealstatus", "result", "outcome", "stage", "state"],
        "date": ["date", "month", "year", "time", "period", "created", "closed", "ordered", "start", "end", "joined"],
        "name": ["name", "fullname", "customername", "clientname", "staffname", "empname"],
    }

    def find(keywords, pool):
        return next((c for c in pool if any(k in cl[c] for k in keywords)), None)

    date_candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_candidates.append(col)
        elif any(k in cl[col] for k in kw["date"]):
            date_candidates.append(col)

    return {
        "value_col": find(kw["value"], num),
        "discount_col": find(kw["discount"], num),
        "booking_col": find(kw["booking"], num),
        "branch_col": find(kw["branch"], cat),
        "emp_col": find(kw["emp"], cat),
        "status_col": find(kw["status"], cat),
        "date_col": date_candidates[0] if date_candidates else None,
        "name_col": find(kw["name"], cat),
        "num_cols": num,
        "cat_cols": cat,
    }


def compute_kpis(df, det):
    kpis = {"total_records": int(len(df))}

    vc = det["value_col"]
    sc = det["status_col"]
    dc = det["discount_col"]
    bc = det["booking_col"]
    br = det["branch_col"]
    ec = det["emp_col"]

    if vc and vc in df.columns:
        s = df[vc].dropna()
        if not s.empty:
            kpis.update({
                "total_value": round(float(s.sum()), 0),
                "avg_value": round(float(s.mean()), 0),
                "max_value": round(float(s.max()), 0),
                "min_value": round(float(s.min()), 0),
                "value_col": vc
            })

    if sc and sc in df.columns:
        vc_counts = df[sc].astype(str).value_counts()
        win_vals = [v for v in vc_counts.index if any(k in str(v).lower() for k in ["won", "win", "closed", "success", "complete"])]

        if win_vals and len(df) > 0:
            won = int(vc_counts[win_vals].sum())
            kpis.update({
                "win_rate": round((won / len(df)) * 100, 1),
                "won_count": won
            })

        kpis["status_breakdown"] = {str(k): int(v) for k, v in vc_counts.head(6).items()}
        kpis["status_col"] = sc

    if dc and dc in df.columns:
        s = df[dc].dropna()
        if not s.empty:
            kpis.update({
                "avg_discount": round(float(s.mean()), 0),
                "total_discount": round(float(s.sum()), 0),
                "discount_col": dc
            })

    if bc and bc in df.columns:
        s = df[bc].dropna()
        if not s.empty:
            kpis.update({
                "avg_booking": round(float(s.mean()), 0),
                "booking_col": bc
            })

    if br and br in df.columns:
        kpis["branch_count"] = int(df[br].nunique())
        kpis["branch_col"] = br
        if vc and vc in df.columns and len(df) > 1:
            try:
                grp = df.groupby(br)[vc].sum()
                if not grp.empty:
                    kpis["top_branch"] = str(grp.idxmax())
            except Exception:
                pass

    if ec and ec in df.columns:
        kpis["emp_count"] = int(df[ec].nunique())
        kpis["emp_col"] = ec
        if vc and vc in df.columns and len(df) > 1:
            try:
                grp = df.groupby(ec)[vc].sum()
                if not grp.empty:
                    kpis["top_employee"] = str(grp.idxmax())
            except Exception:
                pass

    return kpis


def compute_charts(df, det):
    charts = {}

    vc = det["value_col"]
    br = det["branch_col"]
    ec = det["emp_col"]
    sc = det["status_col"]
    dc = det["discount_col"]
    dtc = det["date_col"]

    if br and vc and br in df.columns and vc in df.columns:
        try:
            g = df.groupby(br)[vc].sum().round(0).sort_values(ascending=False).head(10)
            if not g.empty:
                charts["branch_revenue"] = {
                    "labels": list(g.index.astype(str)),
                    "values": [float(v) for v in g.values],
                    "title": f"{vc} by {br}"
                }

            g2 = df.groupby(br).size().sort_values(ascending=False).head(10)
            if not g2.empty:
                charts["branch_deals"] = {
                    "labels": list(g2.index.astype(str)),
                    "values": [int(v) for v in g2.values],
                    "title": f"Count by {br}"
                }
        except Exception:
            pass

    if ec and vc and ec in df.columns and vc in df.columns:
        try:
            g = df.groupby(ec)[vc].sum().round(0).sort_values(ascending=False).head(10)
            if not g.empty:
                charts["emp_revenue"] = {
                    "labels": list(g.index.astype(str)),
                    "values": [float(v) for v in g.values],
                    "title": f"{vc} by {ec}"
                }

            g2 = df.groupby(ec).size().sort_values(ascending=False).head(10)
            if not g2.empty:
                charts["emp_deals"] = {
                    "labels": list(g2.index.astype(str)),
                    "values": [int(v) for v in g2.values],
                    "title": f"Count by {ec}"
                }
        except Exception:
            pass

    if sc and sc in df.columns:
        try:
            g = df[sc].astype(str).value_counts().head(8)
            if not g.empty:
                charts["status_pie"] = {
                    "labels": list(g.index.astype(str)),
                    "values": [int(v) for v in g.values],
                    "title": f"Distribution of {sc}"
                }
        except Exception:
            pass

    if dtc and dtc in df.columns:
        try:
            df2 = df.copy()
            df2["_m"] = pd.to_datetime(df2[dtc], errors="coerce").dt.to_period("M").astype(str)
            df2 = df2[df2["_m"] != "NaT"]

            if vc and vc in df2.columns:
                g = df2.groupby("_m")[vc].sum().round(0).sort_index().tail(12)
                if not g.empty:
                    charts["monthly_revenue"] = {
                        "labels": list(g.index.astype(str)),
                        "values": [float(v) for v in g.values],
                        "title": f"Monthly {vc}"
                    }

            g2 = df2.groupby("_m").size().sort_index().tail(12)
            if not g2.empty:
                charts["monthly_count"] = {
                    "labels": list(g2.index.astype(str)),
                    "values": [int(v) for v in g2.values],
                    "title": "Monthly Count"
                }
        except Exception:
            pass

    if dc and vc and dc in df.columns and vc in df.columns:
        try:
            s = df[[dc, vc]].dropna().head(300)
            if not s.empty:
                charts["scatter"] = {
                    "x": [float(v) for v in s[dc].round(0).values],
                    "y": [float(v) for v in s[vc].round(0).values],
                    "title": f"{dc} vs {vc}",
                    "xlabel": dc,
                    "ylabel": vc
                }
        except Exception:
            pass

    if not charts:
        num_cols = det["num_cols"]
        cat_cols = det["cat_cols"]

        if num_cols:
            for col in num_cols[:3]:
                s = df[col].dropna()
                if not s.empty:
                    charts[f"hist_{col}"] = {
                        "labels": ["min", "mean", "max"],
                        "values": [float(s.min()), float(s.mean()), float(s.max())],
                        "title": f"Stats of {col}"
                    }

        elif cat_cols:
            for col in cat_cols[:2]:
                g = df[col].astype(str).value_counts().head(10)
                if not g.empty:
                    charts[f"cat_{col}"] = {
                        "labels": list(g.index.astype(str)),
                        "values": [int(v) for v in g.values],
                        "title": f"Top values in {col}"
                    }

    return charts


def auto_insights(df, det, kpis):
    ins = []

    vc = det["value_col"]
    br = det["branch_col"]
    ec = det["emp_col"]
    dc = det["discount_col"]
    dtc = det["date_col"]

    if vc and vc in df.columns:
        s = df[vc].dropna()
        if not s.empty:
            ins.append(f"Total {vc}: {s.sum():,.0f}")
            ins.append(f"Average {vc}: {s.mean():,.0f}")
            ins.append(f"Highest {vc}: {s.max():,.0f}")

    if br and vc and br in df.columns and vc in df.columns and len(df) > 1:
        try:
            top = df.groupby(br)[vc].sum()
            if not top.empty:
                ins.append(f"Top {br}: {top.idxmax()} ({top.max():,.0f})")
        except Exception:
            pass

    if ec and vc and ec in df.columns and vc in df.columns and len(df) > 1:
        try:
            top = df.groupby(ec)[vc].sum()
            if not top.empty:
                ins.append(f"Top {ec}: {top.idxmax()} ({top.max():,.0f})")
        except Exception:
            pass

    if "win_rate" in kpis:
        ins.append(f"Win Rate: {kpis['win_rate']}% ({kpis['won_count']} won / {len(df)} total)")

    if dc and dc in df.columns:
        s = df[dc].dropna()
        if not s.empty:
            ins.append(f"Avg {dc}: {s.mean():,.0f}")

    if dtc and dtc in df.columns:
        try:
            df2 = df.copy()
            df2["_m"] = pd.to_datetime(df2[dtc], errors="coerce").dt.to_period("M").astype(str)
            df2 = df2[df2["_m"] != "NaT"]
            if not df2.empty:
                best = df2.groupby("_m").size().idxmax()
                ins.append(f"Busiest Month: {best}")
        except Exception:
            pass

    miss = int(df.isnull().sum().sum())
    ins.append(f"Missing Values: {miss}" if miss else "Data Quality: Clean")

    return ins


def build_col_stats(df, num_cols):
    col_stats = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) == 0:
            col_stats[c] = {
                "mean": None,
                "min": None,
                "max": None,
                "median": None,
                "missing": int(df[c].isnull().sum()),
                "total": None
            }
        else:
            col_stats[c] = {
                "mean": round(float(s.mean()), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
                "median": round(float(s.median()), 2),
                "missing": int(df[c].isnull().sum()),
                "total": round(float(s.sum()), 0)
            }
    return col_stats


def build_filter_options(df, det):
    filter_options = {}
    for col in det["cat_cols"]:
        try:
            unique_count = int(df[col].nunique(dropna=True))
            if 1 < unique_count <= 60:
                values = sorted(df[col].dropna().astype(str).unique().tolist())
                filter_options[col] = values
        except Exception:
            pass
    return filter_options


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    try:
        df = read_uploaded_file(file)
        df = try_parse_dates(df)

        if df.empty:
            return jsonify({"error": "Uploaded file is empty"}), 400

        df_store["data"] = df
        last_filtered["data"] = df.copy()
        last_analysis["filename"] = file.filename

        det = smart_detect(df)
        kpis = compute_kpis(df, det)
        charts = compute_charts(df, det)
        insights = auto_insights(df, det, kpis)
        num_cols = det["num_cols"]
        col_stats = build_col_stats(df, num_cols)

        try:
            describe_txt = df.describe(include="all").fillna("").astype(str).head(10).to_string()
        except Exception:
            describe_txt = "Summary unavailable"

        analysis = call_ai(
            "You are a data analyst. Give a short plain-text summary in max 5 lines. No markdown.",
            f"Rows: {df.shape[0]}\nColumns: {list(df.columns)}\nSummary:\n{describe_txt}\nGive key findings and one recommendation.",
            260
        )
        last_analysis["text"] = analysis

        missing = {c: int(v) for c, v in df.isnull().sum().items() if int(v) > 0}
        filter_options = build_filter_options(df, det)

        date_range = {}
        if det["date_col"] and det["date_col"] in df.columns:
            try:
                dates = pd.to_datetime(df[det["date_col"]], errors="coerce").dropna()
                if not dates.empty:
                    date_range = {
                        "min": str(dates.min().date()),
                        "max": str(dates.max().date()),
                        "col": det["date_col"]
                    }
            except Exception:
                pass

        response_data = {
            "success": True,
            "filename": file.filename,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "total_rows": int(df.shape[0]),
            "filtered_rows": int(df.shape[0]),
            "col_names": list(df.columns),
            "analysis": analysis,
            "col_stats": col_stats,
            "charts": charts,
            "kpis": kpis,
            "insights": insights,
            "missing": missing,
            "detected": det,
            "filter_options": filter_options,
            "date_range": date_range,
            "table_data": df.head(300).fillna("").to_dict(orient="records"),
            "table_cols": list(df.columns),
            "all_numeric": num_cols,
            "all_cat": det["cat_cols"],
            "ai_enabled": bool(API_KEY),
        }

        return jsonify(make_json_safe(response_data))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/filter", methods=["POST"])
def apply_filter():
    if "data" not in df_store:
        return jsonify({"error": "Upload file first"}), 400

    body = request.json or {}
    base_df = df_store["data"]
    df = base_df.copy()

    for col, val in body.get("filters", {}).items():
        if col in df.columns and val not in [None, "", "__ALL__"]:
            df = df[df[col].astype(str) == str(val)]

    dr = body.get("date_range", {})
    date_col = dr.get("col")
    if date_col and date_col in df.columns:
        try:
            date_series = pd.to_datetime(df[date_col], errors="coerce")
            if dr.get("from"):
                df = df[date_series >= pd.to_datetime(dr["from"])]
                date_series = pd.to_datetime(df[date_col], errors="coerce")
            if dr.get("to"):
                df = df[date_series <= pd.to_datetime(dr["to"])]
        except Exception:
            pass

    last_filtered["data"] = df.copy()

    det = smart_detect(df)
    kpis = compute_kpis(df, det)
    charts = compute_charts(df, det)
    insights = auto_insights(df, det, kpis)
    col_stats = build_col_stats(df, det["num_cols"])

    response_data = {
        "rows": int(len(df)),
        "total_rows": int(len(base_df)),
        "filtered_rows": int(len(df)),
        "kpis": kpis,
        "charts": charts,
        "insights": insights,
        "col_stats": col_stats,
        "table_cols": list(df.columns),
        "table_data": df.head(300).fillna("").to_dict(orient="records")
    }

    return jsonify(make_json_safe(response_data))


@app.route("/chart-data", methods=["POST"])
def chart_data_api():
    if "data" not in df_store:
        return jsonify({"error": "Upload file first"}), 400

    body = request.json or {}
    df = last_filtered.get("data", df_store["data"])

    cat_col = body.get("cat")
    num_col = body.get("num")
    agg = body.get("agg", "mean")

    if not cat_col or cat_col not in df.columns:
        return jsonify({"error": "Invalid category column"}), 400

    if agg != "count" and (not num_col or num_col not in df.columns):
        return jsonify({"error": "Invalid numeric column"}), 400

    try:
        fn = {"mean": "mean", "sum": "sum", "count": "count"}.get(agg, "mean")

        if agg == "count":
            grp = df.groupby(cat_col).size().sort_values(ascending=False).head(10)
        else:
            grp = getattr(df.groupby(cat_col)[num_col], fn)().round(0).sort_values(ascending=False).head(10)

        return jsonify(make_json_safe({
            "labels": list(grp.index.astype(str)),
            "values": list(grp.values)
        }))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required"}), 400

    if "data" not in df_store:
        return jsonify({"error": "Upload file first"}), 400

    df = last_filtered.get("data", df_store["data"])
    det = smart_detect(df)
    q_lower = question.lower()
    col_info = f"Columns: {list(df.columns)}, Rows: {df.shape[0]}"

    list_kw = ["show", "list", "all", "who", "which", "name", "staff", "employee", "display", "give", "find"]
    top_kw = ["top", "bottom", "best", "worst", "rank", "highest", "lowest"]
    stat_kw = ["average", "mean", "max", "min", "total", "count", "how many", "sum", "percent", "rate"]

    is_top = any(k in q_lower for k in top_kw)
    is_stat = any(k in q_lower for k in stat_kw)
    is_list = any(k in q_lower for k in list_kw)

    if is_top:
        num_cols = det["num_cols"]
        rel_col = next((c for c in num_cols if any(w in c.lower() for w in q_lower.split())), num_cols[0] if num_cols else None)
        n = next((int(w) for w in q_lower.split() if w.isdigit()), 5)
        asc = any(k in q_lower for k in ["bottom", "lowest", "worst", "least", "minimum"])
        top_df = (df.nsmallest(n, rel_col) if asc else df.nlargest(n, rel_col)) if rel_col else df.head(n)

        system = "List these records clearly numbered with key values. No explanation. Plain text."
        user = f"{col_info}\n\nData:\n{top_df.to_string(index=False)}\n\nQ: {question}"

    elif is_stat:
        num_cols = det["num_cols"]
        stats = []

        for col in num_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            if any(w in col.lower() for w in q_lower.split()) or any(k in q_lower for k in ["all", "total", "every"]):
                stats.append(f"{col}: total={s.sum():,.0f}, mean={s.mean():,.0f}, max={s.max():,.0f}, min={s.min():,.0f}")

        if not stats and num_cols:
            stats = [f"{col}: mean={df[col].dropna().mean():,.0f}, max={df[col].dropna().max():,.0f}" for col in num_cols[:5] if not df[col].dropna().empty]

        for col in det["cat_cols"][:5]:
            if col.lower() in q_lower or any(w in col.lower() for w in q_lower.split()):
                stats.append(f"{col}:\n{df[col].astype(str).value_counts().head(8).to_string()}")

        system = "Answer with exact numbers. 2-3 short sentences. Direct. No code."
        user = f"{col_info}\n\nStats:\n{chr(10).join(stats)}\n\nQ: {question}"

    elif is_list:
        data_str = df.head(50).to_string(index=False)[:4500]
        system = "List ONLY the relevant values asked. Numbered. No stats. No explanation."
        user = f"{col_info}\n\nData:\n{data_str}\n\nQ: {question}"

    else:
        data_str = df.head(15).to_string(index=False)
        system = "You are a data analyst. Answer directly in max 4 lines. Plain text. No code."
        user = f"{col_info}\n\nSample:\n{data_str}\n\nQ: {question}"

    answer = call_ai(system, user, 450)

    has_filter = False
    filter_count = int(len(df))

    try:
        if API_KEY:
            fq = call_ai(
                "Return ONLY valid pandas query string or 'none'. No markdown.",
                f"Columns:{list(df.columns)}, dtypes:{dict(df.dtypes.astype(str))}\nQ:{question}",
                70
            ).strip().strip("\"'")

            if fq.lower() not in ["none", ""] and len(fq) < 300:
                filtered = df.query(fq)
                last_filtered["data"] = filtered
                has_filter = True
                filter_count = int(len(filtered))
    except Exception:
        pass

    return jsonify(make_json_safe({
        "answer": answer,
        "has_filter": has_filter,
        "filter_count": int(filter_count),
        "ai_enabled": bool(API_KEY)
    }))


@app.route("/download-filtered")
def download_filtered():
    df = last_filtered.get("data", df_store.get("data"))
    if df is None:
        return "No data", 400

    out = io.StringIO()
    df.to_csv(out, index=False)
    out.seek(0)

    return send_file(
        io.BytesIO(out.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="filtered_data.csv"
    )


@app.route("/download-report")
def download_report():
    df = last_filtered.get("data", df_store.get("data"))
    if df is None:
        return "No data", 400

    det = smart_detect(df)
    kpis = compute_kpis(df, det)
    ins = auto_insights(df, det, kpis)

    try:
        stats_txt = df.describe(include="all").fillna("").astype(str).to_string()
    except Exception:
        stats_txt = "Statistics unavailable"

    report = f"""DATALENS — AI ANALYTICS REPORT
{'=' * 60}
File: {last_analysis.get('filename', 'data.csv')}
Rows: {df.shape[0]} | Cols: {df.shape[1]}
Columns: {', '.join(df.columns)}
{'=' * 60}
AI ANALYSIS
{last_analysis.get('text', '')}
{'=' * 60}
KEY INSIGHTS
{chr(10).join('• ' + i for i in ins)}
{'=' * 60}
STATISTICS
{stats_txt}
{'=' * 60}
SAMPLE (first 20 rows)
{df.head(20).to_string(index=False)}
{'=' * 60}
Generated by DataLens AI"""

    return send_file(
        io.BytesIO(report.encode()),
        mimetype="text/plain",
        as_attachment=True,
        download_name="datalens_report.txt"
    )


@app.route("/sample")
def sample():
    csv = """deal_id,deal_date,branch,sales_employee_id,deal_status,deal_value,discount_value,booking_amount
D0000001,2025-01-15,Mumbai,E0043,Won,1500000,75000,150000
D0000002,2025-01-22,Delhi,E0042,Won,2200000,154000,220000
D0000003,2025-02-10,Pune,E0051,Lost,900000,63000,90000
D0000004,2025-02-18,Mumbai,E0060,Won,1850000,129500,185000
D0000005,2025-03-05,Hyderabad,E0011,Won,1200000,84000,120000
D0000006,2025-03-12,Delhi,E0054,Won,750000,52500,75000
D0000007,2025-03-25,Bangalore,E0053,Lost,980000,68600,98000
D0000008,2025-04-08,Mumbai,E0043,Won,1650000,115500,165000
D0000009,2025-04-15,Pune,E0042,Won,1100000,77000,110000
D0000010,2025-05-02,Hyderabad,E0060,Lost,850000,59500,85000
D0000011,2025-05-18,Delhi,E0011,Won,1950000,136500,195000
D0000012,2025-06-01,Mumbai,E0051,Won,2100000,147000,210000
D0000013,2025-06-14,Bangalore,E0054,Won,1300000,91000,130000
D0000014,2025-07-03,Pune,E0053,Lost,760000,53200,76000
D0000015,2025-07-20,Delhi,E0043,Won,1750000,122500,175000
D0000016,2025-08-05,Hyderabad,E0042,Won,1400000,98000,140000
D0000017,2025-08-19,Mumbai,E0060,Won,1900000,133000,190000
D0000018,2025-09-03,Bangalore,E0011,Lost,670000,46900,67000
D0000019,2025-09-22,Delhi,E0051,Won,2050000,143500,205000
D0000020,2025-10-10,Pune,E0054,Won,1150000,80500,115000"""

    return send_file(
        io.BytesIO(csv.encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="sample_sales.csv"
    )


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)