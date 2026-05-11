"""
PySpark Optimization: UDF Optimization
Covers avoiding Python UDFs, using Pandas UDFs (vectorized), and built-in functions.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, upper, regexp_replace, when, pandas_udf, udf
)
from pyspark.sql.types import StringType, DoubleType, LongType
import pandas as pd
import time

spark = SparkSession.builder \
    .appName("UDF Optimization") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

N = 500_000
df = spark.range(N) \
    .withColumn("name",  (col("id") % 1000).cast("string")) \
    .withColumn("value", (col("id") * 1.7))

# ------------------------------------------------------------
# 1. Prefer built-in SQL functions over Python UDFs
#    Built-ins run inside the JVM with Tungsten — no Python serialization overhead.
# ------------------------------------------------------------

# Bad: Python UDF for string upper
@udf(StringType())
def py_upper(s):
    return s.upper() if s else None

t0 = time.time()
df.withColumn("name_up", py_upper(col("name"))).count()
print(f"Python UDF upper:   {time.time() - t0:.2f}s")

# Good: built-in upper()
t0 = time.time()
df.withColumn("name_up", upper(col("name"))).count()
print(f"Built-in upper():   {time.time() - t0:.2f}s")

# ------------------------------------------------------------
# 2. Pandas UDF (vectorized UDF) — batched via Apache Arrow
#    ~10–100x faster than row-level Python UDFs.
#    Use when no built-in equivalent exists.
# ------------------------------------------------------------

@pandas_udf(DoubleType())
def scale_value(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)

t0 = time.time()
df.withColumn("scaled", scale_value(col("value"))).count()
print(f"Pandas UDF scale:   {time.time() - t0:.2f}s")

# Row-level Python UDF equivalent (much slower)
@udf(DoubleType())
def py_scale(v):
    return float(v) / 1000.0   # simplified — can't access series stats

t0 = time.time()
df.withColumn("scaled2", py_scale(col("value"))).count()
print(f"Python UDF scale:   {time.time() - t0:.2f}s")

# ------------------------------------------------------------
# 3. Pandas UDF for grouped aggregation (UDAF-style)
# ------------------------------------------------------------
from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf(DoubleType())
def weighted_avg(values: pd.Series, weights: pd.Series) -> float:
    return float((values * weights).sum() / weights.sum())

result = df.groupBy((col("id") % 10).alias("bucket")) \
    .agg(weighted_avg(col("value"), col("id").cast("double")).alias("w_avg"))
result.show(5)

# ------------------------------------------------------------
# 4. Replace complex conditional UDFs with when/otherwise
# ------------------------------------------------------------

# Bad
@udf(StringType())
def categorize(v):
    if v < 100: return "low"
    elif v < 500: return "mid"
    else: return "high"

# Good: runs entirely in JVM, no Python round-trip
df.withColumn("tier",
    when(col("value") < 100, "low")
    .when(col("value") < 500, "mid")
    .otherwise("high")
).show(5)

# ------------------------------------------------------------
# 5. Enable Arrow for faster Pandas UDF data transfer
#    spark.sql.execution.arrow.pyspark.enabled=true (set in builder above)
# ------------------------------------------------------------
print("\nArrow enabled:", spark.conf.get("spark.sql.execution.arrow.pyspark.enabled"))

# ------------------------------------------------------------
# 6. Cache intermediate results used across multiple UDF calls
# ------------------------------------------------------------
base = df.filter(col("value") > 500).cache()
base.count()

base.withColumn("scaled", scale_value(col("value"))).groupBy().avg("scaled").show()
base.withColumn("tier",
    when(col("value") < 800, "mid").otherwise("high")
).groupBy("tier").count().show()

base.unpersist()

print("UDF optimization examples complete.")
spark.stop()
