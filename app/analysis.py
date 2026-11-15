import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD

TOPIC_KEYWORDS = {
    "Price & affordability": ["price", "cost", "expensive", "sasta", "mehnga", "kitna", "rate", "پیس", "قیمت"],
    "Discounts & offers": ["discount", "offer", "sale", "coupon", "promo", "kam", "off"],
    "Delivery & order status": ["delivery", "deliver", "shipping", "parcel", "order", "waiting", "days", "kitne din", "late"],
    "Availability & location": ["available", "availability", "karachi", "lahore", "stock", "size available", "mil", "in stock"],
    "Quality & fabric": ["quality", "fabric", "cloth", "material", "lawn", "cotton", "kapra", "soft"],
    "Sizing & fit": ["size", "sizing", "fit", "small", "large", "medium", "measurement"],
    "Purchase interest": ["buy", "order", "book", "purchase", "inbox", "dm", "send", "details", "interested"],
}
ACTION_MAP = {
    "Price & affordability": "Test a clear price card and value-focused product copy; review repeated price objections.",
    "Discounts & offers": "Test a limited, clearly explained bundle or discount and compare conversion outcomes.",
    "Delivery & order status": "Publish delivery timelines and improve order-status replies for concerned shoppers.",
    "Availability & location": "Show available sizes, stock status and city-wise delivery coverage in post captions.",
    "Quality & fabric": "Create close-up fabric, stitching and wash-care videos to answer quality questions.",
    "Sizing & fit": "Add a measurement chart and model-fit details to product posts.",
    "Purchase interest": "Use a clear call-to-action and route buying questions to a quick-reply workflow.",
    "General engagement": "Review these comments manually and use them to shape the next content test.",
}

def detect_topic(text):
    t = str(text).lower()
    scores = {topic: sum(1 for word in words if word in t) for topic, words in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] else "General engagement"

def analyze_comments(df):
    data = df.copy()
    data["topic_detected"] = data["comment_text"].apply(detect_topic)
    texts = data["comment_text"].astype(str).tolist()
    # K-means discovers text-similarity groups; keyword topics provide human-readable labels.
    n_clusters = min(5, max(2, len(data) // 8))
    if len(data) < 3:
        data["cluster"] = 0
        cluster_labels = {0: "Early sample"}
    else:
        vec = TfidfVectorizer(ngram_range=(1,2), max_features=1500, stop_words="english")
        matrix = vec.fit_transform(texts)
        n_clusters = min(n_clusters, len(data)-1)
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        data["cluster"] = model.fit_predict(matrix)
        cluster_labels = {}
        for cid, group in data.groupby("cluster"):
            words = " ".join(group["comment_text"].astype(str)).lower()
            cluster_labels[int(cid)] = group["topic_detected"].value_counts().index[0]
    topic_counts = data["topic_detected"].value_counts().to_dict()
    total = len(data)
    topic_rows = [{"topic": k, "count": int(v), "share": round(v/total*100,1),
                   "action": ACTION_MAP[k]} for k,v in topic_counts.items()]
    cluster_rows = []
    for cid, group in data.groupby("cluster"):
        cluster_rows.append({"cluster": int(cid), "label": cluster_labels.get(int(cid), "Audience group"),
                             "count": len(group), "examples": group["comment_text"].head(2).tolist()})
    platform_rows = []
    if "language_type" in data.columns:
        platform_rows = [{"name": str(k), "count": int(v)} for k,v in data["language_type"].fillna("Unspecified").value_counts().items()]
    return {"total": total, "topics": topic_rows, "clusters": cluster_rows,
            "platform_rows": platform_rows, "sample": data.head(12).to_dict(orient="records")}

def analyze_posts(df):
    data = df.copy()
    for col in ["likes", "comments_count", "shares", "views"]:
        if col not in data: data[col] = 0
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    data["engagement"] = data["likes"] + data["comments_count"] + data["shares"]
    data["engagement_rate"] = np.where(data["views"] > 0, data["engagement"]/data["views"]*100, np.nan)
    # Robust outlier detection on available engagement signals.
    features = np.log1p(data[["likes","comments_count","shares","views"]].clip(lower=0))
    if len(data) >= 5 and features.nunique().sum() > 1:
        model = IsolationForest(contamination="auto", random_state=42)
        data["unusual"] = model.fit_predict(features) == -1
    else:
        data["unusual"] = False
    content = data.groupby("content_type", dropna=False).agg(
        posts=("post_id","count"), avg_likes=("likes","mean"),
        avg_comments=("comments_count","mean"), avg_engagement=("engagement","mean")
    ).reset_index().sort_values("avg_engagement", ascending=False)
    return {"total": len(data), "content": content.round(1).to_dict(orient="records"),
            "unusual": data[data["unusual"]].head(10).to_dict(orient="records"),
            "top_posts": data.sort_values("engagement", ascending=False).head(5).to_dict(orient="records")}
