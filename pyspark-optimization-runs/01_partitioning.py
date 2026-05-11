"""
PySpark Optimization: Partitioning Strategies
Covers repartition, coalesce, partitionBy, and partition tuning.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, spark_partition_id, count

spark = SparkSession.builder \
    .appName("Partitioning Optimization") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ------------------------------------------------------------
# 1. Check default partition count
# ------------------------------------------------------------
df = spark.range(1_000_000)
print(f"Default partitions: {df.rdd.getNumPartitions()}")

# ------------------------------------------------------------
# 2. repartition() — increases or decreases partitions (full shuffle)
#    Use when you need even distribution or want to increase parallelism.
# ------------------------------------------------------------
df_repartitioned = df.repartition(50)
print(f"After repartition(50): {df_repartitioned.rdd.getNumPartitions()}")

# Repartition by column — ensures all rows with same key land in same partition
sales = spark.createDataFrame([
    ("Alice", "NY", 100),
    ("Bob",   "CA", 200),
    ("Carol", "NY", 150),
    ("Dave",  "CA", 300),
], ["name", "state", "amount"])

sales_by_state = sales.repartition(4, col("state"))
sales_by_state.withColumn("partition_id", spark_partition_id()) \
    .groupBy("partition_id", "state").count().show()

# ------------------------------------------------------------
# 3. coalesce() — reduces partitions WITHOUT a full shuffle (narrow dependency)
#    Preferred over repartition() when only reducing partition count.
# ------------------------------------------------------------
df_large = spark.range(1_000_000).repartition(200)
df_small = df_large.coalesce(10)
print(f"After coalesce(10): {df_small.rdd.getNumPartitions()}")

# WARNING: coalesce can create uneven partitions. Check distribution:
df_small.withColumn("pid", spark_partition_id()) \
    .groupBy("pid").agg(count("*").alias("row_count")).orderBy("pid").show()

# ------------------------------------------------------------
# 4. partitionBy() — write data partitioned by column (partition pruning on read)
#    Critical for large datasets queried by a specific column.
# ------------------------------------------------------------
from pyspark.sql.functions import year, month, to_date, lit
import random

dates = ["2024-01-15", "2024-02-20", "2024-03-10", "2024-01-25"]
rows = [(d, random.randint(100, 999)) for d in dates * 250]
txn_df = spark.createDataFrame(rows, ["date", "amount"]) \
    .withColumn("date", to_date(col("date"))) \
    .withColumn("year",  year(col("date"))) \
    .withColumn("month", month(col("date")))

txn_df.write.mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet("/tmp/pyspark_opt/transactions_partitioned")

# Reading with partition pruning — Spark skips irrelevant partitions entirely
jan_df = spark.read.parquet("/tmp/pyspark_opt/transactions_partitioned") \
    .filter((col("year") == 2024) & (col("month") == 1))
jan_df.explain(mode="formatted")   # look for PartitionFilters in the plan

# ------------------------------------------------------------
# 5. Tune spark.sql.shuffle.partitions for your cluster size
#    Rule of thumb: 2-3x number of CPU cores across all executors.
# ------------------------------------------------------------
spark.conf.set("spark.sql.shuffle.partitions", "100")

df_agg = sales.groupBy("state").sum("amount")
df_agg.show()

print("Partitioning examples complete.")
spark.stop()
