"""
Restaurant dataset: cleaning -> EDA -> modelling.
Run section by section (works well in Jupyter / VS Code '# %%' cells).
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels
"""
# %% 0. Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", context="notebook")
PATH = "sienna_cafe.csv"

# %% 1. Load  (utf-8-sig strips the BOM that corrupts the first column name)
df = pd.read_csv(PATH, encoding="utf-8-sig")
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
print(df.shape)
df.info()

# %% 2. Cleaning
# 2a. Basic hygiene
assert df["restaurant_id"].is_unique
for c in df.select_dtypes("object").columns.union(df.select_dtypes("string").columns):
    df[c] = df[c].str.strip()

# 2b. Country names (Zomato country-code mapping)
COUNTRY = {1: "India", 14: "Australia", 30: "Brazil", 37: "Canada", 94: "Indonesia",
           148: "New Zealand", 162: "Philippines", 166: "Qatar", 184: "Singapore",
           189: "South Africa", 191: "Sri Lanka", 208: "Turkey", 214: "UAE",
           215: "United Kingdom", 216: "United States"}
df["country"] = df["country_code"].map(COUNTRY)

# 2c. Currency: fix garbled symbol + the Philippines rows labelled 'Botswana Pula'
df["currency"] = df["currency"].str.replace(r"Pounds\(.*\)", "Pounds(GBP)", regex=True)
df.loc[df["country"] == "Philippines", "currency"] = "Philippine Peso(PHP)"
df["currency_code"] = df["currency"].str.extract(r"\(([^)]+)\)")

# 2d. Convert cost to USD so countries are comparable.
#     APPROXIMATE rates -- replace with rates you trust.
USD_PER_UNIT = {"Rs.": 0.012, "$": 1.0, "GBP": 1.3, "R$": 0.19, "AED": 0.272,
                "R": 0.055, "TL": 0.05, "P": 0.073, "IDR": 0.000064,
                "QR": 0.275, "LKR": 0.0033, "PHP": 0.018}
# NZ shares the "$" symbol with US -> handle by country
df["fx"] = df["currency_code"].map(USD_PER_UNIT)
df.loc[df["country"] == "New Zealand", "fx"] = 0.60
df["cost_for_two_usd"] = df["average_cost_for_two"] * df["fx"]
df.loc[df["average_cost_for_two"] == 0, "cost_for_two_usd"] = np.nan  # 0 = missing

# 2e. Ratings: 0.0 means "Not rated", not a real score
df["rating"] = df["aggregate_rating"].replace(0, np.nan)
df["is_rated"] = df["rating"].notna()

# 2f. Coordinates: (0,0) is a placeholder, not a location
bad_geo = (df["longitude"] == 0) & (df["latitude"] == 0)
df.loc[bad_geo, ["longitude", "latitude"]] = np.nan

# 2g. Yes/No -> bool
for c in ["has_table_booking", "has_online_delivery", "is_delivering_now"]:
    df[c] = df[c].eq("Yes")

# 2h. Drop zero-information column
if df["switch_to_order_menu"].nunique() == 1:
    df = df.drop(columns="switch_to_order_menu")

# 2i. Cuisines: fill missing, build list column
df["cuisines"] = df["cuisines"].fillna("Unknown")
df["cuisine_list"] = df["cuisines"].str.split(r"\s*,\s*")
df["n_cuisines"] = df["cuisine_list"].apply(len)
df["primary_cuisine"] = df["cuisine_list"].str[0]

# 2j. Cost outliers: flag with a per-country IQR rule (don't silently delete)
def iqr_flag(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return (s < q1 - 3 * i) | (s > q3 + 3 * i)
df["cost_outlier"] = df.groupby("country")["cost_for_two_usd"].transform(iqr_flag)

# 2k. Quality report
print(df.isna().sum()[lambda s: s > 0])
print("Rated:", df.is_rated.mean().round(3), "| cost outliers:", df.cost_outlier.sum())

# %% 3. Univariate EDA
fig, ax = plt.subplots(2, 2, figsize=(13, 9))
sns.histplot(df["rating"].dropna(), bins=30, kde=True, ax=ax[0, 0]).set_title("Rating (rated only)")
sns.histplot(np.log1p(df["votes"]), bins=40, ax=ax[0, 1]).set_title("log(1+Votes)")
sns.histplot(np.log10(df["cost_for_two_usd"].dropna()), bins=40, ax=ax[1, 0]).set_title("log10 cost for two (USD)")
df["price_range"].value_counts().sort_index().plot.bar(ax=ax[1, 1], title="Price range")
plt.tight_layout(); plt.show()

# %% 4. Geography
top_c = df["country"].value_counts()
print(top_c)
# India dominates (~90%), so analyse India separately from the rest
india = df[df.country == "India"]; world = df[df.country != "India"]

city_stats = (df.groupby(["country", "city"])
                .agg(n=("restaurant_id", "size"), mean_rating=("rating", "mean"),
                     median_cost_usd=("cost_for_two_usd", "median"),
                     delivery_rate=("has_online_delivery", "mean"))
                .query("n >= 30").sort_values("mean_rating", ascending=False))
print(city_stats.head(15))

plt.figure(figsize=(8, 8))
sns.scatterplot(data=df.dropna(subset=["longitude"]), x="longitude", y="latitude",
                hue="country", s=8, legend=False)
plt.title("Restaurant locations"); plt.show()

# %% 5. Cuisines
cx = df.explode("cuisine_list")
cuisine_stats = (cx.groupby("cuisine_list")
                   .agg(n=("restaurant_id", "nunique"), mean_rating=("rating", "mean"),
                        median_cost_usd=("cost_for_two_usd", "median"))
                   .query("n >= 50").sort_values("mean_rating", ascending=False))
print(cuisine_stats.head(15)); print(cuisine_stats.tail(10))

top = cx["cuisine_list"].value_counts().head(15)
top.sort_values().plot.barh(figsize=(8, 6), title="Most common cuisines")
plt.show()

# Cuisine co-occurrence (which cuisines are paired together)
from itertools import combinations
from collections import Counter
pairs = Counter(p for l in df["cuisine_list"] for p in combinations(sorted(set(l)), 2))
print(pd.Series(pairs).sort_values(ascending=False).head(15))

# %% 6. Rating drivers
rated = df[df.is_rated]
print(rated.groupby("has_table_booking")["rating"].agg(["mean", "count"]))
print(rated.groupby("has_online_delivery")["rating"].agg(["mean", "count"]))
print(rated.groupby("price_range")["rating"].agg(["mean", "median", "count"]))

# Significance tests
for col in ["has_table_booking", "has_online_delivery"]:
    a, b = rated.loc[rated[col], "rating"], rated.loc[~rated[col], "rating"]
    u = stats.mannwhitneyu(a, b)
    print(f"{col}: diff={a.mean()-b.mean():+.3f}, MWU p={u.pvalue:.2e}")

# Correlations (Spearman: robust to skew)
num = rated[["rating", "votes", "cost_for_two_usd", "price_range", "n_cuisines"]]
sns.heatmap(num.corr("spearman"), annot=True, cmap="vlag", center=0); plt.show()

# Confounding check: is delivery's effect real once you control for country/price?
import statsmodels.formula.api as smf  # pip install statsmodels
m = smf.ols("rating ~ has_table_booking + has_online_delivery + C(price_range)"
            " + np.log1p(votes) + n_cuisines + C(country)", data=rated).fit()
print(m.summary().tables[1])

# %% 7. Value-for-money & hidden gems
rated = rated.copy()
# Bayesian-shrunk rating so 4.9 with 5 votes doesn't beat 4.5 with 2000 votes
C = rated["rating"].mean(); m_votes = rated["votes"].quantile(0.5)
rated["weighted_rating"] = (rated["votes"] / (rated["votes"] + m_votes)) * rated["rating"] \
                           + (m_votes / (rated["votes"] + m_votes)) * C
gems = rated[(rated.votes.between(20, 200)) & (rated.rating >= 4.2)] \
        .sort_values("weighted_rating", ascending=False)
print(gems[["restaurant_name", "city", "primary_cuisine", "rating", "votes"]].head(20))

# %% 8. Chains
chains = (df.groupby("restaurant_name")
            .agg(n_outlets=("restaurant_id", "size"), mean_rating=("rating", "mean"),
                 total_votes=("votes", "sum"))
            .query("n_outlets >= 5").sort_values("n_outlets", ascending=False))
print(chains.head(15))
df["is_chain"] = df.groupby("restaurant_name")["restaurant_id"].transform("size") >= 5
print(df[df.is_rated].groupby("is_chain")["rating"].mean())

# %% 9. Delhi/NCR-style locality analysis (works for any city)
CITY = "New Delhi"
loc = (df[df.city == CITY].groupby("locality")
         .agg(n=("restaurant_id", "size"), mean_rating=("rating", "mean"),
              median_cost=("cost_for_two_usd", "median"))
         .query("n >= 15").sort_values("mean_rating", ascending=False))
print(loc.head(10))

# %% 10. Predict rating (baseline model)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

feat_num = ["votes", "cost_for_two_usd", "price_range", "n_cuisines"]
feat_bool = ["has_table_booking", "has_online_delivery"]
feat_cat = ["country", "primary_cuisine"]
X = rated[feat_num + feat_bool + feat_cat].copy()
X[feat_bool] = X[feat_bool].astype(int)
X["primary_cuisine"] = X["primary_cuisine"].where(
    X["primary_cuisine"].map(X["primary_cuisine"].value_counts()) >= 30, "Other")
y = rated["rating"]

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), feat_cat)],
                        remainder="passthrough")
model = Pipeline([("pre", pre), ("gb", GradientBoostingRegressor(random_state=0))])
Xtr, Xte, ytr, yte = train_test_split(X.fillna(X.median(numeric_only=True)), y,
                                      test_size=0.2, random_state=0)
model.fit(Xtr, ytr)
print("R^2 test:", model.score(Xte, yte))
print("CV R^2:", cross_val_score(model, Xtr, ytr, cv=5).mean())
# NOTE: 'votes' is partly a consequence of popularity; try dropping it to see
# how much the *intrinsic* features explain.