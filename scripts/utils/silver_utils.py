from pyspark.sql import functions as F
from datetime import datetime
import uuid
import builtins
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def check_schema(dataframe):
    try:
        logger.info(f"checking schema , {len(dataframe.columns)} columns: {dataframe.columns}")
        dataframe.printSchema()
    except Exception as e:
        logger.error(f"schema check failed with error {e}")
        raise


def cast_types(df, type_map):
    try:
        for colname, dtype in type_map.items():
            df = df.withColumn(colname, F.col(colname).cast(dtype))
        logger.info("columns casted successfully!")
        return df
    except Exception as e:
        logger.error(f"column cast failed with error {e}")
        raise


def rename_col(df, colname, new_name):
    try:
        df = df.withColumnRenamed(colname, new_name)
        logger.info(f"column renamed successfully: {colname} -> {new_name}")
        return df
    except Exception as e:
        logger.error(f"column rename failed with error {e}")
        raise


def null_profiling(df, df_name):
    try:
        logger.info(f"null profiling started for {df_name}")
        total = df.count()
        null_counts = df.select([
            F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
            for c in df.columns
        ])
        null_counts.display()
        for col_name, count in null_counts.collect()[0].asDict().items():
            if count is None:
                logger.info(f"{col_name} null profiling skipped — unsupported column type")
            else:
                logger.info(f"{col_name} has {(count / total) * 100:.2f}% nulls")
    except Exception as e:
        logger.error(f"null profiling failed for {df_name} with error {e}")
        raise


def handle_nulls_drop(df, drop_cols=[]):
    try:
        df = df.na.drop(subset=drop_cols)
        logger.info(f"Nulls dropped successfully for columns: {drop_cols}")
        return df
    except Exception as e:
        logger.error(f"nulls drop failed with error {e}")
        raise


def handle_nulls_fill(df, fill_cols={}):
    try:
        df = df.na.fill(fill_cols)
        logger.info(f"nulls filled successfully relative to rules {fill_cols}")
        return df
    except Exception as e:
        logger.error(f"nulls fill failed with error {e}")
        raise


def handle_duplicates(df, subset_cols):
    try:
        before = df.count()
        df = df.dropDuplicates(subset_cols)
        after = df.count()
        logger.info(f"Duplicates removed on {subset_cols}, before drop: {before}, after drop: {after}")
        return df
    except Exception as e:
        logger.error(f"duplicates handling failed with error {e}")
        raise


def standardize_strings(df, rules):
    try:
        for colname, rule in rules.items():
            df = df.withColumn(colname, rule(F.col(colname)))
            logger.info(f"string standardization applied to column: {colname}")
        return df
    except Exception as e:
        logger.error(f"string standardization failed with error {e}")
        raise


def profile_column(df, total_count, colname, rule, condition, table_name, warn_threshold):
    try:
        invalid_df = df.filter(~condition)
        invalid_count = invalid_df.count()
        valid_count = total_count - invalid_count

        if total_count == 0:
            invalid_pct = 0
            valid_pct = 0
        else:
            invalid_pct = builtins.round(invalid_count * 100 / total_count, 2)
            valid_pct = builtins.round(valid_count * 100 / total_count, 2)

        sample_invalids = [
            row[colname] for row in invalid_df.select(colname).limit(3).collect()
        ]

        if invalid_count == 0:
            status = "PASS"
        elif invalid_pct < warn_threshold:
            status = "WARN"
        else:
            status = "FAIL"

        result = {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "table_name":    table_name,
            "column":        colname,
            "rule":          rule,
            "total_rows":    total_count,
            "valid_count":   valid_count,
            "invalid_count": invalid_count,
            "valid_pct":     valid_pct,
            "invalid_pct":   invalid_pct,
            "status":        status,
            "sample_invalids": str(sample_invalids)
        }

        logger.info(
            f"[DQ] {table_name}.{colname} | {rule} = {invalid_count}/{total_count} invalid ({invalid_pct}%)"
        )
        return result

    except Exception as e:
        logger.error(f"[DQ ERROR] {table_name}.{colname} | {rule} → {str(e)}")
        return {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "table_name":    table_name,
            "column":        colname,
            "rule":          rule,
            "total_rows":    total_count,
            "valid_count":   None,
            "invalid_count": None,
            "valid_pct":     None,
            "invalid_pct":   None,
            "status":        "ERROR",
            "sample_invalids": None,
            "error_message": str(e)
        }


def build_dq_table(spark, df, checks, table_name, warn_threshold=5.0):
    total_count = df.count()
    results = []
    for colname, rule, condition in checks:
        result = profile_column(
            df, total_count, colname, rule, condition, table_name, warn_threshold
        )
        results.append(result)
    return spark.createDataFrame(results)


def add_silver_metadata(df):
    try:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:6]
        df = df.withColumn("ingestion_date", F.to_date(F.current_timestamp())) \
               .withColumn("pipeline_run_id", F.lit(run_id)) \
               .withColumn("batch_id", F.lit(f"batch_{run_id}")) \
               .withColumn("layer", F.lit("silver")) \
               .withColumn("updated_at", F.current_timestamp())
        logger.info("Silver metadata columns added successfully")
        return df
    except Exception as e:
        logger.error(f"metadata columns creation failed with error {e}")
        raise