#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySpark数据处理流程 - 为 ecommerce_classifier_v2_2.py 准备训练数据

Author: Bai, Gengyuan

任务：
1. 使用PySpark处理原始数据（events.csv, item_properties）
2. 构建候选集（prefix, item covis, cat covis, popularity, user history）
3. 生成训练所需的所有特征
4. 保存为parquet供后续训练使用

数据窗口：
- 训练：2015-05-01 到 2015-07-01（2个月）
- 验证：2015-07-01 到 2015-08-01（1个月）
- Session gap: 30分钟
"""

import sys
from pathlib import Path
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lag, unix_timestamp, when, concat, lit
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

# 配置
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2015-05-01"
TRAIN_END = "2015-07-01"
VALID_START = "2015-07-01"
VALID_END = "2015-08-01"
SESSION_GAP_MINUTES = 30

print("=" * 80)
print("PySpark数据处理流程开始")
print("=" * 80)
print(f"训练窗口: [{TRAIN_START}, {TRAIN_END})")
print(f"验证窗口: [{VALID_START}, {VALID_END})")
print(f"Session gap: {SESSION_GAP_MINUTES} 分钟")
print()

# ============================================================================
# STEP 1: 初始化Spark会话
# ============================================================================
print("STEP 1: 初始化Spark会话...")

spark = SparkSession.builder \
    .appName("Ecommerce_Training_Data_Preparation") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.local.dir", "/tmp/spark-temp") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")
print("✓ Spark会话创建成功")
print()

# ============================================================================
# STEP 2: 加载和Sessionize Events
# ============================================================================
print("STEP 2: 加载events.csv并进行sessionization...")

events_path = f"file://{(DATA_DIR / 'events.csv').absolute()}"
events_df = spark.read.csv(events_path, header=True, inferSchema=True)

# 转换时间戳（从毫秒到秒）
events_df = events_df.withColumn(
    "ts", 
    F.from_unixtime(col("timestamp") / 1000).cast("timestamp")
)

# 过滤时间范围（包含训练和验证窗口）
events_df = events_df.filter(
    (col("ts") >= F.lit(TRAIN_START).cast("timestamp")) &
    (col("ts") < F.lit(VALID_END).cast("timestamp"))
)

# 重命名列
events_df = events_df.select(
    col("visitorid").cast("bigint").alias("user_id"),
    col("ts"),
    col("itemid").cast("bigint").alias("item_id"),
    col("event")
)

print(f"  加载事件数: {events_df.count():,}")

# Sessionization: 30分钟gap规则
window_user_time = Window.partitionBy("user_id").orderBy("ts")

events_df = events_df.withColumn(
    "prev_ts",
    lag("ts").over(window_user_time)
)

events_df = events_df.withColumn(
    "time_gap_sec",
    when(
        col("prev_ts").isNotNull(),
        unix_timestamp("ts") - unix_timestamp("prev_ts")
    ).otherwise(0)
)

events_df = events_df.withColumn(
    "is_new_session",
    when(
        (col("time_gap_sec") > SESSION_GAP_MINUTES * 60) | col("prev_ts").isNull(),
        1
    ).otherwise(0)
)

window_session = Window.partitionBy("user_id").orderBy("ts")
events_df = events_df.withColumn(
    "session_num",
    F.sum("is_new_session").over(window_session)
)

events_df = events_df.withColumn(
    "session_id",
    concat(col("user_id").cast("string"), lit("_"), col("session_num").cast("string"))
)

# 清理中间列
events_df = events_df.select("session_id", "user_id", "ts", "item_id", "event")

# Cache以加速后续操作
events_df.cache()

session_count = events_df.select("session_id").distinct().count()
print(f"  生成会话数: {session_count:,}")
print("✓ Sessionization完成")
print()

# ============================================================================
# STEP 3: 加载Item Category信息
# ============================================================================
print("STEP 3: 加载item properties并提取category...")

part1_path = f"file://{(DATA_DIR / 'item_properties_part1.csv').absolute()}"
part2_path = f"file://{(DATA_DIR / 'item_properties_part2.csv').absolute()}"

props1 = spark.read.csv(part1_path, header=True, inferSchema=True)
props2 = spark.read.csv(part2_path, header=True, inferSchema=True)

# 合并两部分
item_props = props1.union(props2)

# 转换时间戳
item_props = item_props.withColumn(
    "ts",
    F.from_unixtime(col("timestamp") / 1000).cast("timestamp")
)

# 只保留categoryid属性，并取最新的值
item_props = item_props.filter(col("property") == "categoryid")

item_props = item_props.select(
    col("itemid").cast("bigint").alias("item_id"),
    col("value").cast("bigint").alias("category_id"),
    col("ts")
)

# 对每个item取最新的category_id
window_item = Window.partitionBy("item_id").orderBy(F.desc("ts"))
item_props = item_props.withColumn("rn", F.row_number().over(window_item))
item_category = item_props.filter(col("rn") == 1).select("item_id", "category_id")

item_category.cache()
print(f"  Items with category: {item_category.count():,}")
print("✓ Category信息加载完成")
print()

# ============================================================================
# STEP 4: 提取Add-to-Cart事件
# ============================================================================
print("STEP 4: 提取add-to-cart事件...")

# Join events with category
atc_events = events_df.filter(col("event") == "addtocart") \
    .join(item_category, "item_id", "inner") \
    .select(
        col("session_id"),
        col("user_id"),
        col("ts").alias("atc_ts"),
        col("item_id"),
        col("category_id")
    )

atc_events.cache()

# 分割训练和验证集
atc_train = atc_events.filter(
    (col("atc_ts") >= F.lit(TRAIN_START).cast("timestamp")) &
    (col("atc_ts") < F.lit(TRAIN_END).cast("timestamp"))
)

atc_valid = atc_events.filter(
    (col("atc_ts") >= F.lit(VALID_START).cast("timestamp")) &
    (col("atc_ts") < F.lit(VALID_END).cast("timestamp"))
)

atc_train.cache()
atc_valid.cache()

n_atc_train = atc_train.count()
n_atc_valid = atc_valid.count()

print(f"  训练集ATC事件: {n_atc_train:,}")
print(f"  验证集ATC事件: {n_atc_valid:,}")
print("✓ ATC事件提取完成")
print()

# ============================================================================
# STEP 5: 构建候选集
# ============================================================================
print("STEP 5: 构建候选集...")

def build_candidates_spark(atc_df, split_name, train_cutoff_str):
    """
    为给定的ATC事件构建候选类别集
    包括：prefix, item covisitation, category covisitation, popularity, user history
    """
    print(f"  构建 {split_name} 候选集...")
    
    train_cutoff = F.lit(train_cutoff_str).cast("timestamp")
    
    # 1. Prefix candidates: 会话前缀中出现的所有类别
    prefix_cands = atc_df.alias("a") \
        .join(
            events_df.alias("se"),
            (col("a.session_id") == col("se.session_id")) & (col("se.ts") < col("a.atc_ts")),
            "inner"
        ) \
        .join(item_category.alias("ic"), col("se.item_id") == col("ic.item_id"), "inner") \
        .select(
            col("a.session_id"),
            col("a.atc_ts"),
            col("ic.category_id").alias("category_id")
        ).distinct()
    
    # 2. Item Co-visitation candidates
    # 计算item-item共现（在训练截止前）
    train_events = events_df.filter(col("ts") < train_cutoff)
    
    item_covis = train_events.alias("a") \
        .join(
            train_events.alias("b"),
            (col("a.session_id") == col("b.session_id")) & (col("a.item_id") < col("b.item_id")),
            "inner"
        ) \
        .groupBy(col("a.item_id").alias("item_a"), col("b.item_id").alias("item_b")) \
        .agg(F.count("*").alias("covis")) \
        .filter(col("covis") >= 3)
    
    # 为每个ATC的前缀items找共现items并转换为categories
    prefix_items = atc_df.alias("a") \
        .join(
            events_df.alias("se"),
            (col("a.session_id") == col("se.session_id")) & (col("se.ts") < col("a.atc_ts")),
            "inner"
        ) \
        .select(
            col("a.session_id"),
            col("a.atc_ts"),
            col("se.item_id")
        )
    
    itemcovis_cands = prefix_items.alias("pi") \
        .join(item_covis.alias("iv"), col("pi.item_id") == col("iv.item_a"), "inner") \
        .join(item_category.alias("ic2"), col("iv.item_b") == col("ic2.item_id"), "inner") \
        .groupBy(col("pi.session_id"), col("pi.atc_ts"), col("ic2.category_id")) \
        .agg(F.max("iv.covis").alias("max_covis")) \
        .withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("session_id", "atc_ts").orderBy(F.desc("max_covis"))
            )
        ) \
        .filter(col("rn") <= 15) \
        .select(col("session_id"), col("atc_ts"), col("ic2.category_id").alias("category_id"))
    
    # 3. Category Co-visitation candidates
    # 计算category-category共现
    train_events_with_cat = train_events.alias("se") \
        .join(item_category.alias("ic"), col("se.item_id") == col("ic.item_id"), "inner") \
        .select(col("se.session_id"), col("ic.category_id"))
    
    cat_covis = train_events_with_cat.alias("a") \
        .join(
            train_events_with_cat.alias("b"),
            (col("a.session_id") == col("b.session_id")) & (col("a.category_id") < col("b.category_id")),
            "inner"
        ) \
        .groupBy(col("a.category_id").alias("cat_a"), col("b.category_id").alias("cat_b")) \
        .agg(F.countDistinct("a.session_id").alias("cooccur")) \
        .filter(col("cooccur") >= 5)
    
    prefix_cats = atc_df.alias("a") \
        .join(
            events_df.alias("se"),
            (col("a.session_id") == col("se.session_id")) & (col("se.ts") < col("a.atc_ts")),
            "inner"
        ) \
        .join(item_category.alias("ic"), col("se.item_id") == col("ic.item_id"), "inner") \
        .select(
            col("a.session_id"),
            col("a.atc_ts"),
            col("ic.category_id")
        )
    
    catcovis_cands = prefix_cats.alias("pc") \
        .join(cat_covis.alias("cc"), col("pc.category_id") == col("cc.cat_a"), "inner") \
        .groupBy(col("pc.session_id"), col("pc.atc_ts"), col("cc.cat_b")) \
        .agg(F.max("cc.cooccur").alias("max_cooccur")) \
        .withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("session_id", "atc_ts").orderBy(F.desc("max_cooccur"))
            )
        ) \
        .filter(col("rn") <= 10) \
        .select(col("session_id"), col("atc_ts"), col("cat_b").alias("category_id"))
    
    # 4. Popularity candidates: top 20 全局最流行类别
    cat_pop = train_events.alias("se") \
        .join(item_category.alias("ic"), col("se.item_id") == col("ic.item_id"), "inner") \
        .groupBy("ic.category_id") \
        .agg(F.count("*").alias("cnt")) \
        .orderBy(F.desc("cnt")) \
        .limit(20)
    
    pop_cands = atc_df.alias("a").crossJoin(cat_pop.select(col("category_id").alias("pop_cat_id"))) \
        .select(col("a.session_id"), col("a.atc_ts"), col("pop_cat_id").alias("category_id"))
    
    # 5. User History candidates: 用户历史浏览的类别（最近10个）
    user_past_cats = train_events.alias("se") \
        .join(item_category.alias("ic"), col("se.item_id") == col("ic.item_id"), "inner") \
        .filter(col("se.ts") < train_cutoff) \
        .groupBy(col("se.user_id"), col("ic.category_id")) \
        .agg(F.max("se.ts").alias("last_seen"))
    
    userhist_cands = atc_df.alias("a") \
        .join(
            user_past_cats.alias("upc"),
            (col("a.user_id") == col("upc.user_id")) & (col("upc.last_seen") < col("a.atc_ts")),
            "inner"
        ) \
        .withColumn(
            "rn",
            F.row_number().over(
                Window.partitionBy("a.session_id", "a.atc_ts").orderBy(F.desc("upc.last_seen"))
            )
        ) \
        .filter(col("rn") <= 10) \
        .select(col("a.session_id"), col("a.atc_ts"), col("upc.category_id").alias("category_id"))
    
    # 合并所有候选
    all_candidates = prefix_cands \
        .union(itemcovis_cands) \
        .union(catcovis_cands) \
        .union(pop_cands) \
        .union(userhist_cands) \
        .distinct()
    
    n_cands = all_candidates.count()
    print(f"    {split_name}: {n_cands:,} 个候选")
    
    return all_candidates

# 构建训练和验证候选集
train_candidates = build_candidates_spark(atc_train, "train", TRAIN_END)
valid_candidates = build_candidates_spark(atc_valid, "valid", TRAIN_END)

train_candidates.cache()
valid_candidates.cache()

print("✓ 候选集构建完成")
print()

# ============================================================================
# STEP 5.5: 训练Category Embeddings (Word2Vec)
# ============================================================================
print("STEP 5.5: 训练Category Embeddings (Word2Vec)...")

# 提取每个session的category序列（用于训练Word2Vec）
cat_seqs_spark = events_df.filter(col("ts") < F.lit(TRAIN_END).cast("timestamp")) \
    .join(item_category, "item_id", "inner") \
    .select("session_id", "ts", "category_id") \
    .orderBy("session_id", "ts")

cat_seqs_spark = cat_seqs_spark.groupBy("session_id").agg(
    F.collect_list("category_id").alias("cat_sequence")
)

# 转换为Pandas以使用gensim
cat_seqs_pd = cat_seqs_spark.toPandas()

# 准备训练数据（转换为字符串列表）
sequences = [[str(cat) for cat in seq if cat is not None] for seq in cat_seqs_pd['cat_sequence']]
sequences = [seq for seq in sequences if len(seq) >= 2]  # 过滤短序列

print(f"  提取了 {len(sequences):,} 个category序列用于训练")

# 训练Word2Vec模型
w2v_model = Word2Vec(
    sentences=sequences,
    vector_size=16,
    window=5,
    min_count=3,
    workers=4,
    sg=1,
    epochs=10,
    seed=42
)

print(f"  训练了 {len(w2v_model.wv)} 个category的embeddings")

# 创建embedding查找字典
cat_embeddings = {int(cat): w2v_model.wv[cat] for cat in w2v_model.wv.index_to_key}

# 展示相似度检查
sample_cat = list(cat_embeddings.keys())[0]
similar = w2v_model.wv.most_similar(str(sample_cat), topn=5)
print(f"  示例: Category {sample_cat} 最相似的类别: {[(int(c), round(s, 3)) for c, s in similar]}")

print("✓ Word2Vec训练完成")
print()

# ============================================================================
# STEP 6: 特征工程
# ============================================================================
print("STEP 6: 特征工程...")

def build_features_spark(atc_df, candidates_df, split_name, train_cutoff_str):
    """
    为候选集构建所有训练所需的特征
    """
    print(f"  构建 {split_name} 特征...")
    
    train_cutoff = F.lit(train_cutoff_str).cast("timestamp")
    train_events = events_df.filter(col("ts") < train_cutoff)
    
    # Join ATC with candidates
    base = atc_df.alias("a") \
        .join(
            candidates_df.alias("c"),
            (col("a.session_id") == col("c.session_id")) & (col("a.atc_ts") == col("c.atc_ts")),
            "inner"
        ) \
        .select(
            col("a.session_id"),
            col("a.user_id"),
            col("a.atc_ts"),
            col("a.category_id").alias("true_category_id"),
            col("c.category_id").alias("cand_category_id")
        )
    
    # 1. Prefix统计特征
    prefix_events = base.alias("b") \
        .join(
            events_df.alias("se"),
            (col("b.session_id") == col("se.session_id")) & (col("se.ts") < col("b.atc_ts")),
            "left"
        ) \
        .join(item_category.alias("ic"), col("se.item_id") == col("ic.item_id"), "left")
    
    prefix_stats = prefix_events.groupBy(
        col("b.session_id"), col("b.atc_ts"), col("b.cand_category_id")
    ).agg(
        F.countDistinct(col("se.item_id")).alias("n_prefix_items"),
        F.count(col("se.item_id")).alias("n_prefix_events"),
        F.sum(when(col("ic.category_id") == col("b.cand_category_id"), 1).otherwise(0)).alias("cat_count_in_prefix"),
        F.max(
            when(col("ic.category_id") == col("b.cand_category_id"), 
                 unix_timestamp(col("b.atc_ts")) - unix_timestamp(col("se.ts")))
        ).alias("recency_sec"),
        F.min(col("se.ts")).alias("session_start"),
        F.countDistinct(col("ic.category_id")).alias("n_unique_cats_in_session")
    ).select(
        col("session_id"),
        col("atc_ts"),
        col("cand_category_id"),
        col("n_prefix_items"),
        col("n_prefix_events"),
        col("cat_count_in_prefix"),
        col("recency_sec"),
        col("session_start"),
        col("n_unique_cats_in_session")
    )
    
    # 2. Category全局流行度
    cat_pop = train_events.alias("se") \
        .join(item_category.alias("ic"), col("se.item_id") == col("ic.item_id"), "inner") \
        .groupBy(col("ic.category_id")) \
        .agg(F.count("*").alias("global_pop")) \
        .select(
            col("category_id"),
            col("global_pop")
        )
    
    # 3. User-Category亲和度
    user_cat_aff = train_events.alias("se") \
        .join(item_category.alias("ic"), col("se.item_id") == col("ic.item_id"), "inner") \
        .groupBy(col("se.user_id"), col("ic.category_id")) \
        .agg(
            F.count("*").alias("user_cat_interactions"),
            F.countDistinct(col("se.session_id")).alias("user_cat_sessions")
        ) \
        .select(
            col("user_id"),
            col("category_id"),
            col("user_cat_interactions"),
            col("user_cat_sessions")
        )
    
    # 4. User统计
    user_stats = train_events.groupBy("user_id", "session_id").agg(
        (F.max("ts").cast("long") - F.min("ts").cast("long")).alias("session_duration")
    ).groupBy("user_id").agg(
        F.countDistinct("session_id").alias("total_sessions"),
        F.avg("session_duration").alias("avg_session_duration")
    )
    
    # Join所有特征
    features = base.alias("base") \
        .join(
            prefix_stats.alias("ps"),
            (col("base.session_id") == col("ps.session_id")) &
            (col("base.atc_ts") == col("ps.atc_ts")) &
            (col("base.cand_category_id") == col("ps.cand_category_id")),
            "left"
        ) \
        .join(
            cat_pop.alias("cp"),
            col("base.cand_category_id") == col("cp.category_id"),
            "left"
        ) \
        .join(
            user_cat_aff.alias("uca"),
            (col("base.user_id") == col("uca.user_id")) &
            (col("base.cand_category_id") == col("uca.category_id")),
            "left"
        ) \
        .join(
            user_stats.alias("us"),
            col("base.user_id") == col("us.user_id"),
            "left"
        )
    
    # 计算派生特征
    features = features.select(
        col("base.session_id"),
        col("base.atc_ts"),
        col("base.cand_category_id").alias("category_id"),
        
        # Prefix特征
        F.coalesce(col("ps.n_prefix_items"), F.lit(0)).alias("n_prefix_items"),
        F.coalesce(col("ps.n_prefix_events"), F.lit(0)).alias("n_prefix_events"),
        F.coalesce(col("ps.cat_count_in_prefix"), F.lit(0)).alias("cat_count_in_prefix"),
        (F.coalesce(col("ps.cat_count_in_prefix"), F.lit(0)) / 
         F.greatest(F.coalesce(col("ps.n_prefix_events"), F.lit(1)), F.lit(1))).alias("cat_share_in_prefix"),
        F.coalesce(col("ps.recency_sec"), F.lit(999999)).alias("recency_sec"),
        F.log1p(F.coalesce(col("ps.recency_sec"), F.lit(999999))).alias("log_recency"),
        
        # 时间特征
        F.hour(col("base.atc_ts")).alias("hour_of_day"),
        F.dayofweek(col("base.atc_ts")).alias("day_of_week"),
        when(F.dayofweek(col("base.atc_ts")).isin([1, 7]), 1).otherwise(0).alias("is_weekend"),
        F.coalesce(unix_timestamp(col("base.atc_ts")) - unix_timestamp(col("ps.session_start")), F.lit(0)).alias("time_since_session_start"),
        F.coalesce(col("ps.n_unique_cats_in_session"), F.lit(0)).alias("session_cat_diversity"),
        
        # Category流行度
        F.coalesce(col("cp.global_pop"), F.lit(1)).alias("cat_popularity"),
        F.log1p(F.coalesce(col("cp.global_pop"), F.lit(1))).alias("log_cat_pop"),
        
        # User-Category亲和度
        F.coalesce(col("uca.user_cat_interactions"), F.lit(0)).alias("user_cat_hist"),
        F.log1p(F.coalesce(col("uca.user_cat_interactions"), F.lit(0))).alias("log_user_cat_hist"),
        F.coalesce(col("uca.user_cat_sessions"), F.lit(0)).alias("user_cat_sessions"),
        
        # User统计
        F.coalesce(col("us.total_sessions"), F.lit(0)).alias("user_total_sessions"),
        F.coalesce(col("us.avg_session_duration"), F.lit(0)).alias("user_avg_session_dur"),
        
        # Label
        when(col("base.true_category_id") == col("base.cand_category_id"), 1).otherwise(0).alias("y")
    )
    
    # 先 count 行数（在添加 embeddings 之前）
    n_rows = features.count()
    
    print(f"    {split_name}: {n_rows:,} 行基础特征")
    print(f"    添加 16 维 category embeddings...")
    
    # 广播 embedding 字典以提高性能
    emb_broadcast = spark.sparkContext.broadcast(cat_embeddings)
    
    # 定义 UDF 来获取 embedding 的某一维
    def get_embedding_dim(cat_id, dim_idx):
        emb_dict = emb_broadcast.value
        if cat_id in emb_dict:
            return float(emb_dict[cat_id][dim_idx])
        else:
            return 0.0
    
    # 注册 UDF
    from pyspark.sql.types import FloatType
    get_emb_udf = F.udf(get_embedding_dim, FloatType())
    
    # 逐个添加 embedding 维度
    for dim in range(16):
        features = features.withColumn(
            f'cat_emb_{dim}',
            get_emb_udf(col("category_id"), F.lit(dim))
        )
    
    print(f"    {split_name}: {n_rows:,} 行 x {len(features.columns)} 列特征（含embeddings）")
    
    return features

# 构建训练和验证特征
X_train_spark = build_features_spark(atc_train, train_candidates, "train", TRAIN_END)
X_valid_spark = build_features_spark(atc_valid, valid_candidates, "valid", TRAIN_END)

print("✓ 特征工程完成")
print()

# ============================================================================
# STEP 7: 保存为Parquet
# ============================================================================
print("STEP 7: 保存训练数据...")

train_output_path = f"file://{(OUTPUT_DIR / 'X_train_spark.parquet').absolute()}"
valid_output_path = f"file://{(OUTPUT_DIR / 'X_valid_spark.parquet').absolute()}"

X_train_spark.write.mode("overwrite").parquet(train_output_path)
X_valid_spark.write.mode("overwrite").parquet(valid_output_path)

print(f"  训练集保存至: {OUTPUT_DIR / 'X_train_spark.parquet'}")
print(f"  验证集保存至: {OUTPUT_DIR / 'X_valid_spark.parquet'}")
print("✓ 数据保存完成")
print()

# ============================================================================
# STEP 8: 统计摘要
# ============================================================================
print("=" * 80)
print("数据处理完成摘要")
print("=" * 80)

# 统计标签分布
train_label_dist = X_train_spark.groupBy("y").count().collect()
valid_label_dist = X_valid_spark.groupBy("y").count().collect()

train_pos = [r['count'] for r in train_label_dist if r['y'] == 1][0] if any(r['y'] == 1 for r in train_label_dist) else 0
train_total = sum(r['count'] for r in train_label_dist)
valid_pos = [r['count'] for r in valid_label_dist if r['y'] == 1][0] if any(r['y'] == 1 for r in valid_label_dist) else 0
valid_total = sum(r['count'] for r in valid_label_dist)

print(f"训练集:")
print(f"  总行数: {train_total:,}")
print(f"  正样本: {train_pos:,} ({train_pos/train_total*100:.2f}%)")
print(f"  负样本: {train_total - train_pos:,} ({(train_total-train_pos)/train_total*100:.2f}%)")

print(f"\n验证集:")
print(f"  总行数: {valid_total:,}")
print(f"  正样本: {valid_pos:,} ({valid_pos/valid_total*100:.2f}%)")
print(f"  负样本: {valid_total - valid_pos:,} ({(valid_total-valid_pos)/valid_total*100:.2f}%)")

print("\n特征列:")
feature_cols = [c for c in X_train_spark.columns if c not in ['session_id', 'atc_ts', 'category_id', 'y']]
print(f"  总特征数: {len(feature_cols)}")
print(f"    - 基础特征: 18")
print(f"    - Category Embeddings: 16")
print(f"  特征列表: {', '.join(feature_cols)}")

print("\n下一步:")
print("  1. 运行 ecommerce_classifier_v2_2.py（从LINE 398开始）")
print("  2. 读取 X_train_spark.parquet 和 X_valid_spark.parquet")
print("  3. 添加交互特征（cat_pop_x_user_hist, recency_x_cat_count 等5个）")
print("  4. 训练LightGBM/XGBoost/CatBoost模型")

print("\n" + "=" * 80)

# 停止Spark会话
spark.stop()
print("Spark会话已停止")

