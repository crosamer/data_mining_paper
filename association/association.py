import pandas as pd
import time
from itertools import combinations
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import matplotlib.pyplot as plt

# CONFIG
INPUT_PATH = r"D:/College/Semester_5/Data Mining/coding/tugas paper/association/associationBaru.csv"
OUTPUT_DIR = os.path.dirname(INPUT_PATH)
MIN_SUPPORT = 0.12
MIN_CONFIDENCE = 0.3
TOP_K = 20 

# 1. LOAD FILE (delimiter ; )
print("Loading:", INPUT_PATH)
df_raw = pd.read_csv(INPUT_PATH, sep=';')

# show columns and small peek
print("Columns found:", list(df_raw.columns))
print("First 5 rows:")
print(df_raw.head(5))

# -----------------------------
# 2. Detect format:
#    - If first row values are 0/1 and column count > 1 => one-hot binary matrix
#    - Otherwise assume transactions of items per row or grouped dataset (Member/Date/Item)
# -----------------------------
def looks_like_binary_matrix(df):
    # exclude non-numeric columns first
    # Check if all columns (or at least >1) have only {0,1} values (ignoring NaN)
    numeric_like = True
    col_checks = 0
    for col in df.columns:
        vals = df[col].dropna().unique()
        # if empty, skip
        if len(vals) == 0:
            continue
        # try cast to numeric
        try:
            vals_num = pd.to_numeric(vals, errors='coerce')
            if pd.isna(vals_num).any():
                numeric_like = False
                break
            unique_vals = set(vals_num.astype(int).astype(str).tolist())
            # Accept if unique_vals subset of {'0','1'}
            if not unique_vals.issubset({'0','1'}):
                numeric_like = False
                break
            col_checks += 1
        except Exception:
            numeric_like = False
            break
    return numeric_like and col_checks >= 2

is_binary_matrix = looks_like_binary_matrix(df_raw)

# -----------------------------
# 3. Prepare transactions and binary table
# -----------------------------
transactions = None
df_onehot = None

if is_binary_matrix:
    print("Detected: binary one-hot matrix format (rows = transactions, cols = items).")
    # Ensure columns are item names
    # Convert values to bool
    df_onehot = df_raw.astype(int).astype(bool)
    # Convert to list of transactions
    transactions = []
    for _, row in df_onehot.iterrows():
        items = row[row == True].index.tolist()
        transactions.append(items)
else:
    print("Detected: non-binary format. Attempting to parse as (Member/Date/Item) or CSV of transactions.")
    # Check common column names for member/date/item
    cols_lower = [c.lower() for c in df_raw.columns]
    # Try to find item column
    item_col = None
    for candidate in ["item", "itemdescription", "itemname", "itemDescription", "item_name", "itemdescription "]:
        for c in df_raw.columns:
            if candidate.lower() in c.lower():
                item_col = c
                break
        if item_col:
            break
    # If we found an item col and possibly a transaction id (member/date), group into transactions
    if item_col is not None:
        # try to find member/transaction column
        trans_col = None
        for candidate in ["member", "invoice", "bill", "id", "customer", "member_number", "memberno", "transaction"]:
            for c in df_raw.columns:
                if candidate.lower() in c.lower():
                    trans_col = c
                    break
            if trans_col:
                break
        if trans_col is None:
            # no transaction id: treat each row as a single transaction containing the item (rare)
            print(f"Item column detected: {item_col}. No transaction ID column found. Treating each row as single-item transaction.")
            transactions = df_raw[item_col].dropna().astype(str).apply(lambda x: [x]).tolist()
        else:
            print(f"Grouping by transaction column: {trans_col} (item column: {item_col})")
            df_raw[item_col] = df_raw[item_col].astype(str)
            df_raw[trans_col] = df_raw[trans_col].astype(str)
            transactions = df_raw.groupby(trans_col)[item_col].apply(list).tolist()
    else:
        # fallback: assume each row is already list of 0/1 but maybe read as strings without header
        # Try to treat header as item names and rows as binary values separated by commas/semicolons
        print("Could not auto-detect item column. Trying to interpret file as rows of 0/1 with header as items.")
        # if all columns are numeric 0/1 after removing first header row, treat as one-hot
        try:
            # attempt to coerce to numeric
            df_num = df_raw.apply(pd.to_numeric, errors='coerce')
            if looks_like_binary_matrix(df_num):
                df_onehot = df_num.astype(int).astype(bool)
                transactions = []
                for _, row in df_onehot.iterrows():
                    items = row[row == True].index.tolist()
                    transactions.append(items)
            else:
                raise ValueError("Fallback numeric detection failed.")
        except Exception as e:
            raise RuntimeError("Cannot parse input file format automatically. Please provide file with columns (Member/Invoice/Bill, Item) or one-hot matrix.") from e

# At this point we must have transactions
if transactions is None:
    raise RuntimeError("Failed to derive transactions from input file.")

print("Total transactions:", len(transactions))
# Optional: quick frequency table
from collections import Counter
item_counts = Counter()
for t in transactions:
    item_counts.update(t)

print("Unique items before filtering:", len(item_counts))

# -----------------------------
# 4. FILTER RARE ITEMS (important for performance)
# -----------------------------
# Keep items that appear at least MIN_FREQ times (based on min_support)
min_freq = max(2, int(MIN_SUPPORT * len(transactions)))  # at least 2 occurrences
keep_items = {item for item, cnt in item_counts.items() if cnt >= min_freq}
print(f"Filtering items with frequency < {min_freq}. Items kept:", len(keep_items))

# rebuild transactions with filtered items and drop empty transactions
transactions_f = [[it for it in t if it in keep_items] for t in transactions]
transactions_f = [t for t in transactions_f if len(t) > 0]
print("Transactions after filtering (non-empty):", len(transactions_f))

# Build one-hot DataFrame from filtered transactions
te = TransactionEncoder()
te_ary = te.fit(transactions_f).transform(transactions_f)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_).astype(bool)

# Save a quick frequency csv
freq_df = pd.DataFrame.from_records(list(item_counts.items()), columns=["item", "raw_count"])
freq_df = freq_df[freq_df["item"].isin(keep_items)].sort_values("raw_count", ascending=False)
freq_df.to_csv(os.path.join(OUTPUT_DIR, "item_frequencies_filtered.csv"), index=False)

# -----------------------------
# Helper: evaluation and pretty-print rules
# -----------------------------
def evaluate_and_save(rules_df, name):
    if rules_df is None or len(rules_df) == 0:
        print(f"{name}: no rules found.")
        return {"Rules": 0, "Avg_Confidence": 0, "Avg_Lift": 0, "Runtime_sec": 0}
    # ensure antecedents/consequents are strings for CSV
    out = rules_df.copy()
    # if using mlxtend association_rules output: antecedents/consequents are frozensets
    if "antecedents" in out.columns and "consequents" in out.columns:
        out["antecedents_str"] = out["antecedents"].apply(lambda s: ", ".join(sorted(list(s))) if isinstance(s, (set, frozenset)) else str(s))
        out["consequents_str"] = out["consequents"].apply(lambda s: ", ".join(sorted(list(s))) if isinstance(s, (set, frozenset)) else str(s))
    else:
        # ECLAT/AIS rule format: try itemset/lhs/rhs columns
        if "lhs" in out.columns and "rhs" in out.columns:
            out["antecedents_str"] = out["lhs"].apply(lambda s: ", ".join(sorted(list(s))) if isinstance(s, (set, frozenset)) else str(s))
            out["consequents_str"] = out["rhs"].apply(lambda s: ", ".join(sorted(list(s))) if isinstance(s, (set, frozenset)) else str(s))
        elif "antecedents" in out.columns and "consequents" not in out.columns:
            # maybe columns 'itemA' and 'itemB'
            if "itemA" in out.columns and "itemB" in out.columns:
                out["antecedents_str"] = out["itemA"].astype(str)
                out["consequents_str"] = out["itemB"].astype(str)
            else:
                out["antecedents_str"] = out.index.astype(str)
                out["consequents_str"] = out.index.astype(str)
        else:
            out["antecedents_str"] = out.index.astype(str)
            out["consequents_str"] = out.index.astype(str)

    # compute metrics if not present
    if "confidence" not in out.columns:
        # try compute confidence where possible: confidence = support(itemA+itemB)/support(itemA)
        if "support" in out.columns and "antecedents_str" in out.columns:
            # rough compute: treat antecedents_str single-item
            confs = []
            lifts = []
            for idx, row in out.iterrows():
                a = row["antecedents_str"]
                b = row["consequents_str"]
                # compute support counts
                sup_ab = row["support"]
                # support a:
                cnt_a = item_counts[a] if a in item_counts else None
                if cnt_a:
                    sup_a = cnt_a / len(transactions_f)
                    conf = sup_ab / sup_a if sup_a > 0 else 0
                else:
                    conf = 0
                confs.append(conf)
                # lift approximate:
                cnt_b = item_counts[b] if b in item_counts else None
                if cnt_b:
                    sup_b = cnt_b / len(transactions_f)
                    lift = conf / sup_b if sup_b > 0 else 0
                else:
                    lift = 0
                lifts.append(lift)
            out["confidence"] = confs
            out["lift"] = lifts

    # average metrics
    avg_conf = float(out["confidence"].mean()) if "confidence" in out.columns else 0.0
    avg_lift = float(out["lift"].mean()) if "lift" in out.columns else 0.0

    # save CSV
    save_path = os.path.join(OUTPUT_DIR, f"rules_{name}.csv")
    out.to_csv(save_path, index=False)
    print(f"{name}: saved {len(out)} rules to {save_path}")

    return {"Rules": len(out), "Avg_Confidence": avg_conf, "Avg_Lift": avg_lift, "Runtime_sec": None}

# -----------------------------
# 5. RUN FP-GROWTH (utama)
# -----------------------------
print("\nRunning FP-Growth...")
t0 = time.time()
freq_fp = fpgrowth(df_onehot, min_support=MIN_SUPPORT, use_colnames=True)
rules_fp = association_rules(freq_fp, metric="confidence", min_threshold=MIN_CONFIDENCE)
t_fp = time.time() - t0
print("FP-Growth: frequent itemsets =", len(freq_fp), ", rules =", len(rules_fp), ", time = %.3fs" % t_fp)

fp_metrics = evaluate_and_save(rules_fp, "FP_Growth")
fp_metrics["Runtime_sec"] = t_fp

# print top rules
if rules_fp is not None and len(rules_fp)>0:
    print("\nTop FP-Growth rules by lift:")
    display_fp = rules_fp.sort_values(by="lift", ascending=False).head(TOP_K)
    for _, r in display_fp.iterrows():
        print(f"{set(r['antecedents'])} -> {set(r['consequents'])} | support={r['support']:.3f} conf={r['confidence']:.3f} lift={r['lift']:.3f}")

# -----------------------------
# 6. RUN ECLAT (pairs only - pembanding)
# -----------------------------
print("\nRunning ECLAT (pair-based)...")
t0 = time.time()
# Build TID sets for filtered items
item_tid = {item: set() for item in df_onehot.columns}
for tid, trans in enumerate(transactions_f):
    for item in set(trans):
        if item in item_tid:
            item_tid[item].add(tid)

min_count = max(1, int(MIN_SUPPORT * len(transactions_f)))
eclat_rules = []
items_list = list(item_tid.keys())
for i in range(len(items_list)):
    a = items_list[i]
    tids_a = item_tid[a]
    if len(tids_a) < min_count:
        continue
    for j in range(i+1, len(items_list)):
        b = items_list[j]
        tids_b = item_tid[b]
        inter = tids_a & tids_b
        if len(inter) >= min_count:
            support = len(inter) / len(transactions_f)
            # compute confidence A->B
            conf = len(inter) / len(tids_a) if len(tids_a)>0 else 0
            # compute support of b
            sup_b = len(tids_b) / len(transactions_f) if len(tids_b)>0 else 0
            lift = conf / sup_b if sup_b>0 else 0
            eclat_rules.append({"lhs": frozenset([a]), "rhs": frozenset([b]), "support": support, "confidence": conf, "lift": lift})

t_eclat = time.time() - t0
print("ECLAT: rules =", len(eclat_rules), ", time = %.3fs" % t_eclat)
df_eclat = pd.DataFrame(eclat_rules)
eclat_metrics = evaluate_and_save(df_eclat, "ECLAT")
eclat_metrics["Runtime_sec"] = t_eclat

if len(df_eclat) > 0:
    print("\nTop ECLAT rules by lift:")
    for _, r in df_eclat.sort_values(by="lift", ascending=False).head(TOP_K).iterrows():
        print(f"{set(r['lhs'])} -> {set(r['rhs'])} | support={r['support']:.3f} conf={r['confidence']:.3f} lift={r['lift']:.3f}")

# -----------------------------
# 7. RUN AIS (simple pair-based)
# -----------------------------
print("\nRunning AIS (simple pair-based)...")
t0 = time.time()
pair_counts = {}
for trans in transactions_f:
    unique = sorted(set(trans))
    for a, b in combinations(unique, 2):
        pair = (a, b)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

ais_rules = []
for (a, b), cnt in pair_counts.items():
    if cnt >= min_count:
        support = cnt / len(transactions_f)
        # confidence a->b = support(pair) / support(a)
        sup_a = len(item_tid[a]) / len(transactions_f) if a in item_tid else 0
        conf = (cnt / len(transactions_f)) / sup_a if sup_a>0 else 0
        sup_b = len(item_tid[b]) / len(transactions_f) if b in item_tid else 0
        lift = conf / sup_b if sup_b>0 else 0
        ais_rules.append({"lhs": frozenset([a]), "rhs": frozenset([b]), "support": support, "confidence": conf, "lift": lift})

t_ais = time.time() - t0
print("AIS: rules =", len(ais_rules), ", time = %.3fs" % t_ais)
df_ais = pd.DataFrame(ais_rules)
ais_metrics = evaluate_and_save(df_ais, "AIS")
ais_metrics["Runtime_sec"] = t_ais

if len(df_ais) > 0:
    print("\nTop AIS rules by lift:")
    for _, r in df_ais.sort_values(by="lift", ascending=False).head(TOP_K).iterrows():
        print(f"{set(r['lhs'])} -> {set(r['rhs'])} | support={r['support']:.3f} conf={r['confidence']:.3f} lift={r['lift']:.3f}")

# -----------------------------
# 8. SUMMARY TABLE
# -----------------------------
summary = pd.DataFrame([
    {"Algorithm": "FP-Growth", **fp_metrics},
    {"Algorithm": "ECLAT", **eclat_metrics},
    {"Algorithm": "AIS", **ais_metrics},
])
print("\n\n=== SUMMARY METRICS ===")
print(summary.to_string(index=False))

# Save summary
summary.to_csv(os.path.join(OUTPUT_DIR, "association_summary.csv"), index=False)
print("Saved summary to", os.path.join(OUTPUT_DIR, "association_summary.csv"))

print("\nAll rule CSV files saved in:", OUTPUT_DIR)

# ============================================
# 9. GRAPHICS / PLOTTING
# ============================================
import matplotlib.pyplot as plt

# ----- A. Grafik Perbandingan Waktu Eksekusi -----
plt.figure(figsize=(8,5))
plt.bar(
    ["FP-Growth", "ECLAT", "AIS"],
    [fp_metrics["Runtime_sec"], eclat_metrics["Runtime_sec"], ais_metrics["Runtime_sec"]]
)
plt.title("Perbandingan Waktu Eksekusi")
plt.xlabel("Algoritma")
plt.ylabel("Waktu (detik)")
plt.grid(axis='y')
plt.show()

# ----- B. Grafik Jumlah Rules -----
plt.figure(figsize=(8,5))
plt.bar(
    ["FP-Growth", "ECLAT", "AIS"],
    [fp_metrics["Rules"], eclat_metrics["Rules"], ais_metrics["Rules"]]
)
plt.title("Jumlah Association Rules")
plt.xlabel("Algoritma")
plt.ylabel("Jumlah Rules")
plt.grid(axis='y')
plt.show()

# ----- C1. Grafik Rata-rata Confidence -----
plt.figure(figsize=(8,5))
plt.bar(
    ["FP-Growth", "ECLAT", "AIS"],
    [fp_metrics["Avg_Confidence"], eclat_metrics["Avg_Confidence"], ais_metrics["Avg_Confidence"]]
)
plt.title("Rata-rata Confidence")
plt.xlabel("Algoritma")
plt.ylabel("Confidence")
plt.grid(axis='y')
plt.show()

# ----- C2. Grafik Rata-rata Lift -----
plt.figure(figsize=(8,5))
plt.bar(
    ["FP-Growth", "ECLAT", "AIS"],
    [fp_metrics["Avg_Lift"], eclat_metrics["Avg_Lift"], ais_metrics["Avg_Lift"]]
)
plt.title("Rata-rata Lift")
plt.xlabel("Algoritma")
plt.ylabel("Lift")
plt.grid(axis='y')
plt.show()

# ----- D. Grafik Frequent Itemsets FP-Growth -----
plt.figure(figsize=(8,5))
plt.bar(["FP-Growth"], [len(freq_fp)])
plt.title("Jumlah Frequent Itemsets FP-Growth")
plt.xlabel("FP-Growth")
plt.ylabel("Jumlah Itemset")
plt.grid(axis='y')
plt.show()

# ============================================
# 10. GRAPHICS – TOP RULES BY LIFT (FP/ECLAT/AIS)
# ============================================
print("=== ECLAT columns ===")
print(df_eclat.columns)
print(df_eclat.head())

print("\n=== AIS columns ===")
print(df_ais.columns)
print(df_ais.head())

def convert_pair_rules(df):
    """Robust converter: mendukung format item1/item2, atau rule langsung."""
    if df is None or len(df) == 0:
        return None

    df2 = df.copy()

    # Jika sudah ada rule lengkap
    if "antecedents" in df2.columns and "consequents" in df2.columns:
        return df2

    # Jika ECLAT/AIS hanya punya item1-item2
    if "item1" in df2.columns and "item2" in df2.columns:
        df2["antecedents"] = df2["item1"].apply(lambda x: {str(x)})
        df2["consequents"] = df2["item2"].apply(lambda x: {str(x)})
        return df2

    # Jika hanya frequent itemsets → tidak bisa jadi rule
    if "itemsets" in df2.columns:
        print("Data hanya frequent itemsets — tidak bisa membuat rules.")
        return None

    print("Format tidak dikenali, skip.")
    return None


df_eclat_rules = convert_pair_rules(df_eclat)
df_ais_rules = convert_pair_rules(df_ais)


def plot_top_rules(title, rules_df, top_k=10):
    if rules_df is None or len(rules_df) == 0:
        print(f"{title}: no rules to plot.")
        return

    # Pastikan rules sudah di-sort berdasarkan lift
    df_top = rules_df.sort_values(by="lift", ascending=False).head(top_k)

    # Buat label aturan: "A, B -> C"
    labels = [
        f"{', '.join(sorted(list(r['antecedents'])))} → {', '.join(sorted(list(r['consequents'])))}"
        for _, r in df_top.iterrows()
    ]
    lifts = df_top["lift"].tolist()

    plt.figure(figsize=(10, 7))
    plt.barh(labels, lifts)
    plt.title(title)
    plt.xlabel("Lift")
    plt.ylabel("Rule")
    plt.gca().invert_yaxis()   # rule terbaik di atas
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


# --- Grafik Top Rules FP-Growth ---
plot_top_rules("Top FP-Growth Rules by Lift", rules_fp, TOP_K)

# --- Grafik Top Rules ECLAT ---
plot_top_rules("Top ECLAT Rules by Lift", df_eclat_rules, TOP_K)

# --- Grafik Top Rules AIS ---
plot_top_rules("Top AIS Rules by Lift", df_ais_rules, TOP_K)


