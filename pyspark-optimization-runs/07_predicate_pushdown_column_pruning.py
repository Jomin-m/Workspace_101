"""
PySpark Optimization: Predicate Pushdown & Column Pruning
Reduces data read from source by pushing filters and selecting only needed columns.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month

spark = SparkSession.builder \
    .appName("Predicate Pushdown and Column Pruning") \
    .getOrCreate()

# ------------------------------------------------------------
# Write a sample Parquet dataset with multiple columns and partitions
# ------------------------------------------------------------
from pyspark.sql import Row
import random, os

random.seed(7)
rows = [
    Row(
        id=i,
        name=f"user_{i}",
        age=random.randint(18, 70),
        country=random.choice(["US", "UK", "IN", "DE"]),
        score=round(random.uniform(0, 100), 2),
        active=random.choice([True, False]),
    )
    for i in range(500_000)
]
df = spark.createDataFrame(rows)

out_path = "/tmp/pyspark_opt/users_parquet"
df.write.mode("overwrite").parquet(out_path)

# ------------------------------------------------------------
# 1. Predicate Pushdown
#    Spark pushes filter conditions into the Parquet reader so only
#    matching row-groups are read — drastically reduces I/O.
# ------------------------------------------------------------
df_parquet = spark.read.parquet(out_path)

# Filter BEFORE any wide transformation — Spark can push it to the source
filtered = df_parquet.filter(col("country") == "US")
print("=== With Predicate Pushdown ===")
filtered.explain(mode="formatted")   # look for PushedFilters in FileScan

# Bad pattern: filter AFTER a wide operation prevents pushdown
from pyspark.sql.functions import upper
wide_then_filter = df_parquet.withColumn("country_up", upper(col("country"))) \
    .filter(col("country") == "US")   # pushdown still works on original column
print("=== Filter on original column after withColumn (pushdown still works) ===")
wide_then_filter.explain(mode="formatted")

# Anti-pattern: filtering on a derived column breaks pushdown
derived_filter = df_parquet.withColumn("country_up", upper(col("country"))) \
    .filter(col("country_up") == "US")   # PushedFilters will be empty
print("=== Filter on derived column (NO pushdown) ===")
derived_filter.explain(mode="formatted")

# ------------------------------------------------------------
# 2. Column Pruning
#    Parquet stores data by column; selecting only needed columns
#    means Spark reads only those column chunks from disk.
# ------------------------------------------------------------

# Bad: read all columns, only use two
all_cols = df_parquet.select("id", "country", "name", "age", "score", "active")
all_cols.filter(col("country") == "US").count()

# Good: project only what you need upfront
pruned = df_parquet.select("id", "country")
pruned.filter(col("country") == "US").count()
print("=== Column Pruning plan ===")
pruned.explain(mode="formatted")   # ReadSchema shows only selected columns

# ------------------------------------------------------------
# 3. Partition pruning (works with partitioned Parquet/ORC/Delta)
# ------------------------------------------------------------
from pyspark.sql.functions import to_date, lit
dates = ["2024-01-10", "2024-02-15", "2024-03-20"] * 50_000
dated = spark.createDataFrame(
    [(d, random.randint(1, 1000)) for d in dates], ["date", "amount"]
).withColumn("date", to_date(col("date"))) \
 .withColumn("month", month(col("date"))) \
 .withColumn("year",  year(col("date")))

dated.write.mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet("/tmp/pyspark_opt/dated_parquet")

dated_df = spark.read.parquet("/tmp/pyspark_opt/dated_parquet")

# Partition pruning: Spark only reads the year=2024/month=1 folder
jan = dated_df.filter((col("year") == 2024) & (col("month") == 1))
print("=== Partition Pruning plan ===")
jan.explain(mode="formatted")   # PartitionFilters: [year=2024, month=1]

# ------------------------------------------------------------
# 4. Combined best practice — filter early, select early, partition wisely
# ------------------------------------------------------------
result = spark.read.parquet(out_path) \
    .filter(col("country") == "US") \
    .filter(col("active") == True) \
    .select("id", "age", "score")

result.groupBy().avg("score").show()
print("=== Combined optimized plan ===")
result.explain(mode="formatted")

print("Predicate pushdown and column pruning examples complete.")
spark.stop()
