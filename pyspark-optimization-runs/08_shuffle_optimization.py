"""
PySpark Optimization: Shuffle Optimization
Covers reducing shuffles, tuning shuffle partitions, and sort-merge vs hash joins.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum, broadcast

spark = SparkSession.builder \
    .appName("Shuffle Optimization") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# ------------------------------------------------------------
# 1. What causes a shuffle?
#    - groupBy / agg
#    - join (sort-merge)
#    - distinct / dropDuplicates
#    - repartition (not coalesce)
#    - orderBy / sort (unless on partition key)
# ------------------------------------------------------------
df = spark.range(1_000_000) \
    .withColumn("key",   (col("id") % 50).cast("string")) \
    .withColumn("value", (col("id") * 1.5))

# Each of these triggers a shuffle (Exchange in the plan)
agg = df.groupBy("key").agg(spark_sum("value").alias("total"))
print("=== GroupBy shuffle (Exchange present) ===")
agg.explain(mode="formatted")

# ------------------------------------------------------------
# 2. Tune spark.sql.shuffle.partitions
#    Default is 200 — too high for small data, too low for very large data.
#    Heuristic: target ~128 MB per partition after the shuffle.
# ------------------------------------------------------------
spark.conf.set("spark.sql.shuffle.partitions", "20")
agg_tuned = df.groupBy("key").agg(spark_sum("value").alias("total"))
print(f"\nShuffle partitions: {agg_tuned.rdd.getNumPartitions()}")
agg_tuned.show(5)

# ------------------------------------------------------------
# 3. Avoid redundant shuffles — chain aggregations carefully
# ------------------------------------------------------------

# Bad: two separate shuffles
step1 = df.groupBy("key").agg(spark_sum("value").alias("total"))
step2 = step1.groupBy("key").agg(spark_sum("total").alias("grand_total"))  # extra shuffle
step2.explain(mode="formatted")

# Good: single-pass aggregation
single_pass = df.groupBy("key").agg(spark_sum("value").alias("grand_total"))
single_pass.explain(mode="formatted")

# ------------------------------------------------------------
# 4. Sort-merge join vs broadcast join (shuffle avoidance)
# ------------------------------------------------------------
orders = spark.range(500_000) \
    .withColumn("prod_id", (col("id") % 200).cast("int")) \
    .withColumn("amt", col("id").cast("double"))

products = spark.range(200) \
    .withColumn("prod_id", col("id").cast("int")) \
    .withColumn("category", (col("id") % 10).cast("string"))

# Sort-merge join: shuffles both sides
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
smj = orders.join(products, "prod_id")
print("\n=== Sort-Merge Join (2 shuffles) ===")
smj.explain(mode="formatted")

# Broadcast join: zero shuffle on large side
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", str(10 * 1024 * 1024))
bhj = orders.join(broadcast(products), "prod_id")
print("=== Broadcast Hash Join (0 shuffles on large side) ===")
bhj.explain(mode="formatted")

# ------------------------------------------------------------
# 5. Use reduceByKey / treeReduce for RDD-level aggregations (less shuffle data)
#    combineByKey aggregates locally before shuffle — less data sent over network.
# ------------------------------------------------------------
rdd = spark.sparkContext.parallelize(
    [(str(i % 50), float(i)) for i in range(500_000)]
)

# groupByKey — ships ALL values over the network (expensive)
# rdd.groupByKey().mapValues(sum) is NOT recommended

# reduceByKey — partial aggregation happens locally (recommended)
reduced = rdd.reduceByKey(lambda a, b: a + b)
print(f"\nreduceByKey result count: {reduced.count()}")

# ------------------------------------------------------------
# 6. Persist before multiple shuffles on the same base DataFrame
#    Avoids recomputing from source for each shuffle operation.
# ------------------------------------------------------------
base = df.filter(col("value") > 100_000).cache()
base.count()   # materialise

agg_a = base.groupBy("key").agg(spark_sum("value").alias("sum_val"))
agg_b = base.groupBy("key").agg(count("*").alias("cnt"))
agg_a.show(3)
agg_b.show(3)

base.unpersist()

print("Shuffle optimization examples complete.")
spark.stop()
