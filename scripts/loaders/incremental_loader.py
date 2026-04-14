import logging
logging.basicConfig(level=logging.INFO)

def get_incremental(spark,table_name,key="id") :

    # 1. Read bronze table
    bronze_df = spark.read.table(f"workspace.ecommerce_bronze.{table_name}")
    logging.info("Step 1: Bronze table read successfully")

    # 2. Get last ingestion_time from silver
    try:
        last_ingestion_time = spark.sql(f"""
        SELECT MAX(ingestion_time) 
        FROM silver.{table_name}""").collect()[0][0]
        logging.info(f"Step 2: Last ingestion_time found = {last_ingestion_time}")

    except Exception:
        last_ingestion_time = None
        logging.info("Step 2: Silver table not found, first run detected")

    # 3. Filter new rows from bronze
    if last_ingestion_time is None:
        # First run = load everything
        df = bronze_df
        logging.info("Step 3: First run, loading all rows from bronze")
    else:
        # Subsequent runs = only new rows
        df = bronze_df.filter(bronze_df.ingestion_time > last_ingestion_time)
        logging.info(f"Step 3: Filtered new rows after {last_ingestion_time}")

    df = df.dropDuplicates([key])
    return df