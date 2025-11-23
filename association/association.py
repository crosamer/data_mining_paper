import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import time
from itertools import combinations

# Load Dataset
df = pd.read_csv("D:/College/Semester_5/Data Mining/coding/tugas paper/association/market20k.csv", sep=";")

df = df[["BillNo", "Itemname"]]

# =============================
# FILTER ITEM RARE
# =============================
item_counts = df["Itemname"].value_counts()
df = df[df["Itemname"].isin(item_counts[item_counts > 20].index)]

print("Unique Item After Filtering:", df["Itemname"].nunique())

# =============================
# GROUP BY TRANSACTION
# =============================
transactions = df.groupby("BillNo")["Itemname"].apply(list).tolist()
print("Total Transactions:", len(transactions))

# =============================
# ENCODE
# =============================
start_enc = time.time()
te = TransactionEncoder()
te_data = te.fit_transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)
encoding_time = time.time() - start_enc

# =============================
# EVALUASI MODEL
# =============================
def evaluate_rules(rules, runtime):
    if len(rules) == 0:
        return {
            "Rules": 0,
            "Avg_Confidence": 0,
            "Avg_Lift": 0,
            "Runtime_sec": runtime
        }

    return {
        "Rules": len(rules),
        "Avg_Confidence": rules["confidence"].mean(),
        "Avg_Lift": rules["lift"].mean(),
        "Runtime_sec": runtime
    }

results = {}

# =============================
# 1. FP-GROWTH (ALGORITMA UTAMA)
# =============================
start = time.time()
freq_fp = fpgrowth(df_encoded, min_support=0.02, use_colnames=True)
rules_fp = association_rules(freq_fp, metric="confidence", min_threshold=0.3)
runtime_fp = time.time() - start
results["FP-Growth"] = evaluate_rules(rules_fp, runtime_fp)

# =============================
# 2. ECLAT (PEMBANDING 1)
# =============================
def eclat(df_bin, min_support=0.02):
    min_count = int(min_support * len(df_bin))
    frequent = {}

    # 1-itemset
    for item in df_bin.columns:
        support = df_bin[item].sum()
        if support >= min_count:
            frequent[frozenset([item])] = support

    # 2-itemset only (agar tetap cepat)
    items = list(frequent.keys())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            new_item = items[i] | items[j]
            cols = list(new_item)
            support = (df_bin[cols].sum(axis=1) == len(cols)).sum()
            if support >= min_count:
                frequent[new_item] = support

    rows = [{"itemsets": k, "support": v / len(df_bin)} for k, v in frequent.items()]
    return pd.DataFrame(rows)

start = time.time()
freq_eclat = eclat(df_encoded, min_support=0.02)

# Buat aturan
rules_eclat_list = []
for _, row in freq_eclat.iterrows():
    itemset = row["itemsets"]
    supp_ab = row["support"]

    if len(itemset) < 2:
        continue

    A = frozenset([list(itemset)[0]])
    B = itemset - A

    supp_a = freq_eclat[freq_eclat["itemsets"] == A]["support"].values
    supp_b = freq_eclat[freq_eclat["itemsets"] == B]["support"].values

    if len(supp_a) == 0 or len(supp_b) == 0:
        continue

    conf = supp_ab / supp_a[0]
    lift = conf / supp_b[0]

    rules_eclat_list.append({
        "lhs": A,
        "rhs": B,
        "confidence": conf,
        "lift": lift
    })

rules_eclat = pd.DataFrame(rules_eclat_list)
runtime_eclat = time.time() - start
results["ECLAT"] = evaluate_rules(rules_eclat, runtime_eclat)

# =============================
# 3. AIS (PEMBANDING 2)
# =============================
def ais_algo(transactions, min_support=0.02):
    min_count = int(min_support * len(transactions))

    # Single item support
    item_sup = {}
    for t in transactions:
        for item in t:
            item_sup[item] = item_sup.get(item, 0) + 1

    L1 = {frozenset([k]): v for k, v in item_sup.items() if v >= min_count}

    # Pair generation
    pair_sup = {}
    for t in transactions:
        t_unique = sorted(set(t))
        for pair in combinations(t_unique, 2):
            pair_sup[frozenset(pair)] = pair_sup.get(frozenset(pair), 0) + 1

    L2 = {k: v for k, v in pair_sup.items() if v >= min_count}

    rows = [{"itemsets": k, "support": v / len(transactions)} for k, v in L2.items()]
    return pd.DataFrame(rows)

start = time.time()
freq_ais = ais_algo(transactions, min_support=0.02)

rules_ais_list = []
for _, row in freq_ais.iterrows():
    itemset = row["itemsets"]
    supp = row["support"]

    A = frozenset([list(itemset)[0]])
    B = itemset - A

    rules_ais_list.append({
        "lhs": A,
        "rhs": B,
        "confidence": 1.0,
        "lift": 1.0 / supp
    })

rules_ais = pd.DataFrame(rules_ais_list)
runtime_ais = time.time() - start

results["AIS"] = evaluate_rules(rules_ais, runtime_ais)

# =============================
# OUTPUT HASIL
# =============================
print("\n===============================")
print("   PERBANDINGAN ALGORITMA")
print("===============================")

for algo, res in results.items():
    print(f"\n{algo}:")
    for k, v in res.items():
        print(f"   {k}: {v}")
