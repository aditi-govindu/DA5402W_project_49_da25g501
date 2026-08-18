"""
Apache Spark Preprocessing & Feature Engineering Pipeline.
Demonstrates scalable data engineering using PySpark:
- Distributed text cleaning and normalization
- Spark ML Tokenization & StopWords removal
- Lexical feature extraction (TTR, word count, length, uppercase density)
- Stratified Train/Val/Test data partitioning
- Exporting to Parquet format
"""

import os
import argparse
from typing import Tuple
from src.utils.logger import get_logger
from src.utils.helpers import load_config

logger = get_logger("spark_preprocessing")

def get_spark_session(app_name: str = "AI-vs-Human-Text-Spark", master: str = "local[*]", log_level: str = "WARN"):
    """Initialize and return an Apache Spark session."""
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(log_level)
    return spark

def run_spark_preprocessing(
    raw_path: str = "data/raw/ai_vs_human_text.csv",
    output_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
):
    """
    Execute end-to-end Apache Spark distributed data preprocessing.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, FloatType
    from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

    logger.info("Initializing Apache Spark Session...")
    spark = get_spark_session()

    logger.info(f"Reading raw dataset with PySpark from: {raw_path}")
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(raw_path)
    
    # 1. Clean missing/null values
    df = df.filter(F.col("text").isNotNull() & (F.length(F.trim(F.col("text"))) > 0))
    df = df.filter(F.col("label").isNotNull())

    # 2. Text Normalization using Spark SQL functions
    # Remove URLs, HTML tags, special chars and lowercase
    cleaned_df = df.withColumn(
        "clean_text",
        F.lower(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(F.col("text"), r"https?://\S+|www\.\S+", ""),
                    r"<.*?>", ""
                ),
                r"[^\w\s\.\?!,]", ""
            )
        )
    ).withColumn("clean_text", F.trim(F.col("clean_text")))

    # 3. Feature Engineering with Spark
    # Text length (character count)
    # Word count
    # Uppercase character count / ratio
    # Punctuation count
    cleaned_df = cleaned_df.withColumn("char_length", F.length(F.col("text")))
    cleaned_df = cleaned_df.withColumn(
        "word_count",
        F.size(F.split(F.trim(F.col("clean_text")), r"\s+"))
    )
    cleaned_df = cleaned_df.withColumn(
        "avg_word_length",
        F.when(F.col("word_count") > 0, F.col("char_length") / F.col("word_count")).otherwise(0.0)
    )

    # 4. Spark ML Tokenizer & StopWords Remover
    tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="tokens", pattern=r"\W+", minTokenLength=2)
    tokenized_df = tokenizer.transform(cleaned_df)

    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    processed_df = remover.transform(tokenized_df)

    # Calculate Lexical Diversity (Type-Token Ratio = unique tokens / total tokens)
    def compute_ttr(tokens):
        if not tokens or len(tokens) == 0:
            return 0.0
        return float(len(set(tokens)) / len(tokens))

    ttr_udf = F.udf(compute_ttr, FloatType())
    processed_df = processed_df.withColumn("lexical_diversity", ttr_udf(F.col("tokens")))

    # 5. Label Encoding (AI-generated -> 1, Human-written -> 0)
    processed_df = processed_df.withColumn(
        "target",
        F.when(F.col("label") == "AI-generated", 1).otherwise(0).cast(IntegerType())
    )

    # 6. Stratified Split (Train, Validation, Test)
    logger.info(f"Splitting data: Train={train_ratio}, Val={val_ratio}, Test={test_ratio} with seed={seed}")
    
    ai_df = processed_df.filter(F.col("target") == 1)
    human_df = processed_df.filter(F.col("target") == 0)

    ai_train, ai_val, ai_test = ai_df.randomSplit([train_ratio, val_ratio, test_ratio], seed=seed)
    human_train, human_val, human_test = human_df.randomSplit([train_ratio, val_ratio, test_ratio], seed=seed)

    train_df = ai_train.union(human_train)
    val_df = ai_val.union(human_val)
    test_df = ai_test.union(human_test)

    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits as Parquet files
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    logger.info(f"Writing train set ({train_df.count()} records) to {train_path}...")
    train_df.toPandas().to_parquet(train_path, index=False)

    logger.info(f"Writing validation set ({val_df.count()} records) to {val_path}...")
    val_df.toPandas().to_parquet(val_path, index=False)

    logger.info(f"Writing test set ({test_df.count()} records) to {test_path}...")
    test_df.toPandas().to_parquet(test_path, index=False)

    logger.info("Apache Spark data preprocessing completed successfully.")
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Apache Spark Preprocessing")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_spark_preprocessing(
        raw_path=cfg["data"]["raw_path"],
        output_dir=cfg["data"]["processed_dir"],
        train_ratio=cfg["data"]["train_split"],
        val_ratio=cfg["data"]["val_split"],
        test_ratio=cfg["data"]["test_split"],
        seed=cfg["data"]["random_state"]
    )
