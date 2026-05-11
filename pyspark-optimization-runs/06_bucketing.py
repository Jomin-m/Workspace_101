"""
PySpark Optimization: Bucketing
Pre-partitions data on disk to eliminate shuffle during joins and aggregations.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg

spark = SparkSession.builder \
    .appName("Bucketing Optimization") \
    .config("spark.sql.sources.bucketing.enabled", "true") \
    .enableHiveSupport() \
    .getOrCreate()

# ------------------------------------------------------------
# Sample data
# ------------------------------------------------------------
orders = spark.range(500_000) \
    .withColumn("customer_id", (col("id") % 1000).cast("int")) \
    .withColumn("amount", (col("id") * 2.5))

customers = spark.range(1_000) \
    .withColumn("customer_id", col("id").cast("int")) \
    .withColumn("region", (col("id") % 5).cast("string"))

# ------------------------------------------------------------
# 1. Write bucketed tables
#    Both tables bucketed on the same key with the same bucket count
#    → Spark can join them WITHOUT a shuffle.
# ------------------------------------------------------------
BUCKET_COUNT = 8

orders.write \
    .mode("overwrite") \
    .bucketBy(BUCKET_COUNT, "customer_id") \
    .sortBy("customer_id") \
    .saveAsTable("bucketed_orders")

customers.write \
    .mode("overwrite") \
    .bucketBy(BUCKET_COUNT, "customer_id") \
    .sortBy("customer_id") \
    .saveAsTable("bucketed_customers")

# ------------------------------------------------------------
# 2. Join bucketed tables — no Exchange (shuffle) node in the plan
# ------------------------------------------------------------
ord_tbl = spark.table("bucketed_orders")
cust_tbl = spark.table("bucketed_customers")

bucketed_join = ord_tbl.join(cust_tbl, "customer_id")
print("=== Bucketed Join plan (no Exchange expected) ===")
bucketed_join.explain(mode="formatted")
bucketed_join.groupBy("region").agg(avg("amount").alias("avg_amount")).show()

# ------------------------------------------------------------
# 3. Non-bucketed join for comparison — shows Exchange nodes
# ------------------------------------------------------------
ord_raw  = spark.range(500_000) \
    .withColumn("customer_id", (col("id") % 1000).cast("int")) \
    .withColumn("amount", (col("id") * 2.5))
cust_raw = spark.range(1_000) \
    .withColumn("customer_id", col("id").cast("int")) \
    .withColumn("region", (col("id") % 5).cast("string"))

plain_join = ord_raw.join(cust_raw, "customer_id")
print("=== Plain Join plan (Exchange nodes present) ===")
plain_join.explain(mode="formatted")

# ------------------------------------------------------------
# 4. Bucketing for aggregations
#    Grouping by the bucket key skips the shuffle aggregation phase.
# ------------------------------------------------------------
ord_tbl.groupBy("customer_id").agg(
    count("*").alias("order_count"),
    avg("amount").alias("avg_order")
).show(10)

# ------------------------------------------------------------
# 5. Rules of thumb for bucketing
#    - Use the same bucket count and column on both sides of a join
#    - Bucket count: 2x–4x expected number of executors is a good starting point
#    - Bucketing pays off only when tables are written ONCE and queried MANY times
#    - Not beneficial for streaming or one-off queries
# ------------------------------------------------------------

print("Bucketing examples complete.")
spark.stop()
