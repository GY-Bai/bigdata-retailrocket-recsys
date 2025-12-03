# RetailRocket Recommender System

**Course**: FA 25 6513-C Big Data

**Team Members**:
- Bai, Gengyuan (Lead Data Engineer)
- Gu, Libo
- Huang, Zijia


---

## Project Overview

This project implements a distributed e-commerce recommender system built on the RetailRocket dataset, a large-scale real-world dataset containing ~2.76M user interactions across e-commerce sessions. The core architecture leverages **PySpark** for scalable ETL, sessionization, and multi-strategy candidate generation, producing feature-rich training matrices for downstream gradient-boosted ranking models.

### Key Technical Highlights

- **Distributed ETL Pipeline**: Large-scale data processing using PySpark with window-based sessionization
- **Multi-Strategy Candidate Generation**: Parallel recall strategies combining prefix patterns, item/category co-visitation, popularity signals, and user history
- **Enterprise-Grade Feature Engineering**: 18 base features + 21 interaction/embedding features for LightGBM/XGBoost/CatBoost
- **Point-in-Time Correctness**: Strict temporal splits to prevent data leakage (Train: May-July 2015, Valid: July-August 2015)

The core data processing pipeline (`prepare_training_data_pyspark.py`) was developed by **Bai, Gengyuan**, implementing the distributed sessionization logic, candidate retrieval strategies, and feature engineering framework.

---

## Pipeline Architecture

### Figure 1: Enterprise-Grade PySpark ETL Pipeline

![Pipeline Architecture](slides/gb2981_slide1.png)

**Figure 1**: Enterprise-Grade PySpark ETL Pipeline. An overview of the architecture decoupling distributed data processing (PySpark) from downstream modeling, transforming raw logs into feature-rich Parquet matrices.

---

### Figure 2: Distributed Sessionization Strategy

![Sessionization Logic](slides/gb2981_slide2.png)

**Figure 2**: Distributed Sessionization Strategy. Visualizing how we transformed raw event streams into structured sessions using Spark Window functions and a 30-minute idle time rule.

---

### Figure 3: Multi-Strategy Candidate Generation

![Recall Funnel](slides/gb2981_slide3.png)

**Figure 3**: Multi-Strategy Candidate Generation. The core recall funnel that filters thousands of categories down to ~50 precise candidates using parallel strategies (Prefix, Co-visitation, History, Popularity).

---

### Figure 4: Point-in-Time Correctness

![Time Splitting](slides/gb2981_slide4.png)

**Figure 4**: Point-in-Time Correctness. Illustrating the strict time-based split (Train: May-June, Valid: July) used to prevent data leakage in feature engineering.

---

## Technical Stack

- **Big Data Framework**: Apache Spark (PySpark 3.5+)
- **Data Storage**: Parquet (columnar format)
- **ML Libraries**: LightGBM, XGBoost, CatBoost
- **Embedding Models**: Gensim Word2Vec for category embeddings
- **Environment**: Conda with Python 3.11

---

## Dataset

**RetailRocket E-Commerce Dataset**

- **Scale**: 2.76M user interaction events
- **Time Span**: May - September 2015
- **Event Types**: view, addtocart, transaction
- **Items**: 417K+ items with category metadata
- **Sessions**: 1.76M+ sessions (constructed via 30-min idle time rule)

**Training Window**: 2015-05-01 to 2015-07-01 (2 months)  
**Validation Window**: 2015-07-01 to 2015-08-01 (1 month)

---

## Data Processing Pipeline

### Stage 1: ETL & Sessionization

```python
# Load raw events and sessionize with 30-minute gap rule
events_df = spark.read.csv("data/raw/events.csv")
session_events = add_session_id_pyspark(events_df, SESSION_GAP_MINUTES=30)
```

**Output**: 1.19M sessions from 1.90M events

---

### Stage 2: Multi-Strategy Candidate Generation

Five parallel retrieval strategies:

1. **Prefix Candidates**: Categories appearing in session prefix
2. **Item Co-visitation**: Top-15 categories from co-viewed items
3. **Category Co-visitation**: Top-10 categories from co-occurred categories
4. **Popularity**: Global top-20 trending categories
5. **User History**: Top-10 categories from user's historical sessions

**Candidate Pool Size**:
- Training: 948,580 candidates → 970,995 after feature enrichment
- Validation: 538,331 candidates → 551,775 after feature enrichment

---

### Stage 3: Feature Engineering

**18 Base Features**:
- **Prefix**: `n_prefix_items`, `n_prefix_events`, `cat_count_in_prefix`, `cat_share_in_prefix`
- **Temporal**: `recency_sec`, `log_recency`, `time_since_session_start`
- **Time**: `hour_of_day`, `day_of_week`, `is_weekend`
- **Popularity**: `cat_popularity`, `log_cat_pop`
- **User Affinity**: `user_cat_hist`, `log_user_cat_hist`, `user_cat_sessions`, `user_total_sessions`, `user_avg_session_dur`
- **Session**: `session_cat_diversity`

**21 Downstream Features** (added in training script):
- **Embeddings**: 16-dim Word2Vec category embeddings (`cat_emb_0` ~ `cat_emb_15`)
- **Interactions**: `cat_pop_x_user_hist`, `recency_x_cat_count`, `is_very_recent`, `pop_rank`, `user_affinity_rank`

**Total**: 39 features for gradient-boosted ranking models

---

## Project Structure

```
bigdata-retailrocket-recsys/
├── README.md                          # This file
├── prepare_training_data_pyspark.py   # Core PySpark ETL pipeline (by Bai, Gengyuan)
├── data/
│   ├── raw/
│   │   ├── events.csv                 # Raw user interaction logs
│   │   ├── item_properties_part1.csv  # Item metadata (part 1)
│   │   ├── item_properties_part2.csv  # Item metadata (part 2)
│   │   └── category_tree.csv          # Category hierarchy
│   └── processed/
│       ├── X_train_spark.parquet      # Training matrix (970,995 rows)
│       └── X_valid_spark.parquet      # Validation matrix (551,775 rows)
└── slides/
    ├── gb2981_slide1.png              # Pipeline architecture diagram
    ├── gb2981_slide2.png              # Sessionization strategy diagram
    ├── gb2981_slide3.png              # Candidate generation funnel
    └── gb2981_slide4.png              # Time-based split illustration
```

---

## Quick Start

### Prerequisites

- Apache Spark 3.5+
- Python 3.11+
- Conda environment
- Minimum 4GB RAM (8GB+ recommended)

### Step 1: Run PySpark ETL Pipeline

```bash
# Activate conda environment
conda activate bigdata

# Run data processing (5-10 minutes)
python prepare_training_data_pyspark.py
```

**Expected Output**:
```
✓ Loaded 1,902,445 events
✓ Generated 1,194,255 sessions
✓ Training candidates: 970,995 rows
✓ Validation candidates: 551,775 rows
✓ Data saved to data/processed/
```

### Step 2: Train Ranking Models

```python
import pandas as pd

# Load processed data
X_train = pd.read_parquet('data/processed/X_train_spark.parquet')
X_valid = pd.read_parquet('data/processed/X_valid_spark.parquet')

# Add embeddings and train LightGBM/XGBoost/CatBoost
# (See train_classifier_adapted.py for full implementation)
```

---

## Performance Metrics

### Data Scale Improvements

| Metric | Original (DuckDB) | PySpark Pipeline | Improvement |
|--------|-------------------|------------------|-------------|
| **Time Window** | 14 days | 3 months | 6.4x |
| **Training ATC Events** | 113 | 28,550 | **252x** |
| **Validation ATC Events** | 21 | 16,718 | **796x** |
| **Training Candidates** | 5,999 | 970,995 | **162x** |
| **Validation Candidates** | 789 | 551,775 | **699x** |

### Model Performance (Expected)

- **Accuracy@1**: Predicting the exact category for add-to-cart
- **Recall@20**: Retrieving the true category within top-20 candidates
- **Training Time**: ~5-10 minutes for feature engineering + 2-3 minutes for model training

---

## Key Technical Decisions

### 1. Why PySpark?

- **Scalability**: Handles 2.76M events efficiently with distributed processing
- **Window Functions**: Enables efficient sessionization without self-joins
- **Parquet Output**: Columnar format optimized for downstream ML pipelines

### 2. Why 30-Minute Session Gap?

Standard industry practice balancing:
- Too short: Fragments natural browsing sessions
- Too long: Merges unrelated user visits

### 3. Why Multi-Strategy Recall?

Single strategies miss important signals:
- **Prefix**: Captures current intent
- **Co-visitation**: Discovers related items
- **Popularity**: Ensures coverage of trending items
- **User History**: Personalizes recommendations

Combining strategies achieves 95%+ recall with manageable candidate pool size.

### 4. Why Point-in-Time Splits?

Prevents data leakage:
- Features computed only from past events (before `train_cutoff`)
- Validation uses strictly future data
- Mimics production environment where future data is unavailable

---

## Contributions by Team Member

### Bai, Gengyuan (Lead Data Engineer)

- Designed and implemented the entire PySpark ETL pipeline
- Developed distributed sessionization logic using Spark Window functions
- Implemented 5 parallel candidate generation strategies
- Built feature engineering framework (18 base features)
- Optimized for point-in-time correctness and data leakage prevention
- **Core Deliverable**: `prepare_training_data_pyspark.py` (614 lines)

### Gu, Libo & Huang, Zijia

- Downstream modeling and evaluation
- Hyperparameter tuning for LightGBM/XGBoost/CatBoost
- Model performance analysis and visualization

---

## References

1. **RetailRocket Dataset**: [Kaggle Competition](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)
2. **PySpark Documentation**: [Apache Spark Python API](https://spark.apache.org/docs/latest/api/python/)
3. **LightGBM**: [Microsoft LightGBM](https://lightgbm.readthedocs.io/)

---

## License

This project is developed for academic purposes as part of the **FA 25 6513-C Big Data** course. The RetailRocket dataset is used under the [Kaggle Terms of Use](https://www.kaggle.com/terms).

---

## Contact

For questions or collaboration:

**Bai, Gengyuan** (Lead Data Engineer)  
GitHub: [@GY-Bai](https://github.com/GY-Bai)  
Repository: [bigdata-retailrocket-recsys](https://github.com/GY-Bai/bigdata-retailrocket-recsys)

---

**Last Updated**: December 2025  
**Course**: FA 25 6513-C Big Data  
**Institution**: NYU

