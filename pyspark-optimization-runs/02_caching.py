"""
PySpark Optimization: Caching and Persistence
Covers cache(), persist(), StorageLevel choices, and when to unpersist.
"""

from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
from pyspark.sql.functions import col, count, avg
import time

spark = SparkSession.builder \
    .appName("Caching Optimization") \
    .getOrCreate()

# ------------------------------------------------------------
# Sample data — simulate a dataset read from a slow source
# ------------------------------------------------------------
df = spark.range(2_000_000) \
    .withColumn("value", (col("id") % 100).cast("double")) \
    .withColumn("category", (col("id") % 10).cast("string"))

# ------------------------------------------------------------
# 1. cache() — shorthand for MEMORY_AND_DISK storage level
#    Best for DataFrames reused multiple times in the same job.
# ------------------------------------------------------------
df_cached = df.cache()
df_cached.count()   # materialise the cache

t0 = time.time()
df_cached.filter(col("category") == "3").groupBy("category").agg(avg("value")).show()
print(f"Cached query time: {time.time() - t0:.2f}s")

# ------------------------------------------------------------
# 2. persist() — explicit StorageLevel control
# ------------------------------------------------------------

# MEMORY_ONLY: fastest reads; data dropped if memory is full (no recompute safety net)
df_mem = df.persist(StorageLevel.MEMORY_ONLY)
df_mem.count()

# MEMORY_AND_DISK: spills to disk if memory is full — safer default
df_mem_disk = df.persist(StorageLevel.MEMORY_AND_DISK)
df_mem_disk.count()

# DISK_ONLY: useful for very large DataFrames where memory is precious
df_disk = df.persist(StorageLevel.DISK_ONLY)
df_disk.count()

# OFF_HEAP: stores data off the JVM heap — reduces GC pressure on large datasets
# (requires spark.memory.offHeap.enabled=true and spark.memory.offHeap.size)
# df_offheap = df.persist(StorageLevel.OFF_HEAP)

# ------------------------------------------------------------
# 3. StorageLevel comparison
# ------------------------------------------------------------
levels = {
    "MEMORY_ONLY":      StorageLevel.MEMORY_ONLY,
    "MEMORY_AND_DISK":  StorageLevel.MEMORY_AND_DISK,
    "MEMORY_ONLY_2":    StorageLevel.MEMORY_ONLY_2,      # 2x replicated
    "DISK_ONLY":        StorageLevel.DISK_ONLY,
}
print("\nStorageLevel overview:")
for name, level in levels.items():
    print(f"  {name:20s} | useDisk={level.useDisk} | useMemory={level.useMemory} "
          f"| replication={level.replication}")

# ------------------------------------------------------------
# 4. When NOT to cache
#    - DataFrame is used only once
#    - Dataset is too large and causes memory pressure / spilling
#    - After a wide transformation that won't be reused
# ------------------------------------------------------------

# ------------------------------------------------------------
# 5. Always unpersist when done — frees executor memory immediately
# ------------------------------------------------------------
df_cached.unpersist()
df_mem.unpersist()
df_mem_disk.unpersist()
df_disk.unpersist()

# ------------------------------------------------------------
# 6. Practical pattern: cache a filtered/joined base, run multiple aggregations
# ------------------------------------------------------------
base = spark.range(500_000) \
    .withColumn("region", (col("id") % 5).cast("string")) \
    .withColumn("sales",  (col("id") % 1000).cast("double")) \
    .filter(col("region").isin("0", "1", "2")) \
    .cache()

base.count()   # materialise once

agg1 = base.groupBy("region").agg(avg("sales").alias("avg_sales"))
agg2 = base.groupBy("region").agg(count("*").alias("tx_count"))

agg1.show()
agg2.show()

base.unpersist()

print("Caching examples complete.")
spark.stop()
