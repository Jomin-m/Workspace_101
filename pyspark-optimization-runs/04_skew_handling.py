"""
PySpark Optimization: Data Skew Handling
Covers salting, AQE skew join, skew detection, and repartitioning strategies.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, broadcast, count, rand, concat, lit,
    spark_partition_id, floor, expr
)
from pyspark.sql.types import IntegerType

spark = SparkSession.builder \
    .appName("Skew Handling Optimization") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5") \
    .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB") \
    .getOrCreate()

# ------------------------------------------------------------
# 1. Detect skew — inspect partition row counts
# ------------------------------------------------------------
def show_partition_skew(df, label=""):
    dist = df.withColumn("pid", spark_partition_id()) \
        .groupBy("pid").agg(count("*").alias("rows")) \
        .orderBy(col("rows").desc())
    print(f"\n--- Partition distribution: {label} ---")
    dist.show(20)

# Skewed dataset: key "hot" appears 90% of the time
from pyspark.sql import Row
import random
random.seed(42)

def make_skewed(n=200_000):
    rows = []
    for i in range(n):
        key = "hot" if random.random() < 0.9 else f"key_{random.randint(1, 50)}"
        rows.append(Row(join_key=key, value=float(i)))
    return rows

skewed_df = spark.createDataFrame(make_skewed()).repartition(20, col("join_key"))
show_partition_skew(skewed_df, "skewed input")

lookup = spark.createDataFrame([
    Row(join_key="hot",   label="popular"),
    *[Row(join_key=f"key_{i}", label=f"label_{i}") for i in range(1, 51)],
])

# ------------------------------------------------------------
# 2. Naive join — hot partition causes one task to process 90% of data
# ------------------------------------------------------------
naive = skewed_df.join(lookup, "join_key")
show_partition_skew(naive, "naive join output")

# ------------------------------------------------------------
# 3. Salting — artificially distribute hot keys across N buckets
#    Both sides must be modified: fact table adds random salt,
#    dimension table is exploded to carry all salt values.
# ------------------------------------------------------------
SALT_BUCKETS = 10

# Add random salt to the large (skewed) side
salted_large = skewed_df.withColumn(
    "salted_key",
    concat(col("join_key"), lit("_"), (rand() * SALT_BUCKETS).cast(IntegerType()).cast("string"))
)

# Explode the small side to match every salt bucket
from pyspark.sql.functions import array, explode

salt_array = array(*[lit(i) for i in range(SALT_BUCKETS)])
salted_small = lookup \
    .withColumn("salt", explode(salt_array)) \
    .withColumn("salted_key", concat(col("join_key"), lit("_"), col("salt").cast("string"))) \
    .drop("salt")

salted_join = salted_large.join(broadcast(salted_small), "salted_key") \
    .drop("salted_key")

show_partition_skew(salted_join, "salted join output")
salted_join.groupBy("label").count().show()

# ------------------------------------------------------------
# 4. AQE Skew Join (Spark 3.x) — automatic, no code change needed
#    AQE detects large partitions at runtime and splits them.
#    Enabled via spark.sql.adaptive.skewJoin.enabled=true (set above).
# ------------------------------------------------------------
print("\nAQE config:")
print("  adaptive.enabled               =", spark.conf.get("spark.sql.adaptive.enabled"))
print("  adaptive.skewJoin.enabled      =", spark.conf.get("spark.sql.adaptive.skewJoin.enabled"))
print("  skewedPartitionFactor          =", spark.conf.get("spark.sql.adaptive.skewJoin.skewedPartitionFactor"))

aqe_join = skewed_df.join(lookup, "join_key")
print("\n=== AQE Join plan (skew splits visible at runtime) ===")
aqe_join.explain(mode="formatted")
aqe_join.groupBy("label").count().show()

# ------------------------------------------------------------
# 5. Isolate hot keys — split join into two paths and union
#    Use when only a handful of keys are hot and domain is known.
# ------------------------------------------------------------
HOT_KEYS = ["hot"]

hot_df   = skewed_df.filter(col("join_key").isin(HOT_KEYS))
cold_df  = skewed_df.filter(~col("join_key").isin(HOT_KEYS))

hot_lookup  = lookup.filter(col("join_key").isin(HOT_KEYS))
cold_lookup = lookup.filter(~col("join_key").isin(HOT_KEYS))

# Hot side: broadcast the (small) matching lookup rows
hot_result  = hot_df.join(broadcast(hot_lookup), "join_key")
# Cold side: normal join — balanced keys, no skew
cold_result = cold_df.join(cold_lookup, "join_key")

final = hot_result.union(cold_result)
show_partition_skew(final, "isolated hot-key join")
final.groupBy("label").count().orderBy("label").show()

print("Skew handling examples complete.")
spark.stop()
