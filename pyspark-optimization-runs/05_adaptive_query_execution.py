"""
PySpark Optimization: Adaptive Query Execution (AQE)
Covers dynamic partition coalescing, skew join, runtime plan switching (Spark 3.x).
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, broadcast

spark = SparkSession.builder \
    .appName("AQE Optimization") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1") \
    .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5") \
    .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ------------------------------------------------------------
# 1. AQE Feature Overview
# ------------------------------------------------------------
aqe_settings = {
    "spark.sql.adaptive.enabled":                              "Master switch for AQE",
    "spark.sql.adaptive.coalescePartitions.enabled":           "Merge small shuffle partitions",
    "spark.sql.adaptive.coalescePartitions.minPartitionNum":   "Minimum partitions after coalescing",
    "spark.sql.adaptive.advisoryPartitionSizeInBytes":         "Target size for coalesced partitions",
    "spark.sql.adaptive.skewJoin.enabled":                     "Auto-split skewed join partitions",
    "spark.sql.adaptive.skewJoin.skewedPartitionFactor":       "Multiplier to detect a skewed partition",
    "spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes": "Min bytes for a partition to be skewed",
}
print("AQE Configuration:")
for key, desc in aqe_settings.items():
    try:
        val = spark.conf.get(key)
    except Exception:
        val = "not set"
    print(f"  {key.split('.')[-1]:50s} = {val:10s}  # {desc}")

# ------------------------------------------------------------
# 2. Dynamic partition coalescing
#    With 200 shuffle partitions set, many may end up nearly empty after a filter.
#    AQE merges them at runtime to avoid thousands of tiny tasks.
# ------------------------------------------------------------
df = spark.range(100_000) \
    .withColumn("category", (col("id") % 5).cast("string")) \
    .withColumn("value", (col("id") * 1.5))

# Without AQE: 200 shuffle tasks even if data fits in 3-4 partitions
spark.conf.set("spark.sql.adaptive.enabled", "false")
agg_no_aqe = df.groupBy("category").agg(avg("value").alias("avg_val"))
print("\n=== Without AQE ===")
agg_no_aqe.explain(mode="formatted")
print(f"Partitions (no AQE): {agg_no_aqe.rdd.getNumPartitions()}")

# With AQE: Spark merges small partitions automatically
spark.conf.set("spark.sql.adaptive.enabled", "true")
agg_aqe = df.groupBy("category").agg(avg("value").alias("avg_val"))
agg_aqe.show()
print(f"Partitions (with AQE): {agg_aqe.rdd.getNumPartitions()}")

# ------------------------------------------------------------
# 3. Runtime join strategy switching
#    AQE can switch a sort-merge join to a broadcast join at runtime
#    once it knows the actual size of each side after filtering.
# ------------------------------------------------------------
large = spark.range(500_000) \
    .withColumn("key", (col("id") % 1000).cast("string")) \
    .withColumn("val", col("id").cast("double"))

# After filter this becomes tiny — AQE detects this and broadcasts it
small_after_filter = spark.range(1_000_000) \
    .withColumn("key", (col("id") % 1000).cast("string")) \
    .withColumn("info", col("id").cast("string")) \
    .filter(col("id") < 200)   # leaves ~200 rows

spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")   # disable static threshold
runtime_join = large.join(small_after_filter, "key")
print("\n=== AQE runtime join strategy ===")
runtime_join.explain(mode="formatted")   # AQE may show BroadcastHashJoin at runtime
runtime_join.count()

# Re-enable threshold
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))

# ------------------------------------------------------------
# 4. AQE + Skew join (automatic at runtime)
#    No code changes needed — AQE splits oversized partitions mid-query.
# ------------------------------------------------------------
from pyspark.sql import Row
import random
random.seed(0)

skewed = spark.createDataFrame([
    Row(k="hot" if random.random() < 0.85 else f"k{random.randint(1,20)}", v=float(i))
    for i in range(100_000)
])
dim = spark.createDataFrame([Row(k="hot", label="popular")] +
                            [Row(k=f"k{i}", label=f"L{i}") for i in range(1, 21)])

skew_join = skewed.join(dim, "k")
print("\n=== AQE skew join plan ===")
skew_join.explain(mode="formatted")
skew_join.groupBy("label").count().show()

# ------------------------------------------------------------
# 5. When AQE doesn't help
#    - Streaming queries
#    - MapPartitions / RDD-level code (AQE is SQL-layer only)
#    - Static Spark 2.x clusters
# ------------------------------------------------------------

print("AQE examples complete.")
spark.stop()
