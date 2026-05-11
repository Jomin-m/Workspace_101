"""
PySpark Optimization: Serialization (Kryo)
Kryo serialization is faster and more compact than Java serialization for RDD operations.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import time

# ------------------------------------------------------------
# Kryo serialization applies to:
#   - RDD operations (map, filter, reduceByKey, etc.)
#   - Spark internals (shuffle data, task closures, broadcast vars)
# DataFrames/Datasets use their own Tungsten encoder and are unaffected.
# ------------------------------------------------------------

spark_kryo = SparkSession.builder \
    .appName("Kryo Serialization") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.kryo.registrationRequired", "false") \
    .config("spark.kryo.unsafe", "true") \
    .getOrCreate()

sc = spark_kryo.sparkContext

# ------------------------------------------------------------
# 1. Basic Kryo RDD benchmark — shuffle a large RDD
# ------------------------------------------------------------
N = 2_000_000
rdd = sc.parallelize([(str(i % 100), i) for i in range(N)], numSlices=20)

t0 = time.time()
result = rdd.reduceByKey(lambda a, b: a + b).collect()
print(f"Kryo reduceByKey time: {time.time() - t0:.2f}s  (keys={len(result)})")

# ------------------------------------------------------------
# 2. Broadcast variables — Kryo compresses broadcast payload
# ------------------------------------------------------------
lookup = {str(i): f"label_{i}" for i in range(100)}
bc = sc.broadcast(lookup)

mapped = rdd.mapValues(lambda v: bc.value.get(str(v % 100), "unknown"))
print(f"Broadcast map sample: {mapped.take(3)}")
bc.unpersist()

# ------------------------------------------------------------
# 3. Register custom classes for maximum Kryo efficiency
#    Registered classes get a small integer ID instead of full class name.
# ------------------------------------------------------------
# In Python, class registration is done in Java/Scala; here we show the
# equivalent configuration a Python app passes to the JVM:
#
#   .config("spark.kryo.classesToRegister",
#           "com.example.MyClass,com.example.AnotherClass")
#
# For pure Python objects in RDDs, cloudpickle handles serialization —
# Kryo affects only the JVM-side shuffle/broadcast buffers.

# ------------------------------------------------------------
# 4. Memory impact — Kryo uses ~10x less memory than Java serialization
#    visible via Storage UI (cached RDDs show smaller serialized size).
# ------------------------------------------------------------
large_rdd = sc.parallelize(range(500_000), 10)
large_rdd.persist()   # serialized by Kryo in executor memory
large_rdd.count()     # materialise

print(f"\nCached RDD partitions: {large_rdd.getNumPartitions()}")
large_rdd.unpersist()

# ------------------------------------------------------------
# 5. Configuration reference
# ------------------------------------------------------------
print("\nSerializer configs:")
configs = [
    "spark.serializer",
    "spark.kryo.registrationRequired",
    "spark.kryo.unsafe",
]
for c in configs:
    print(f"  {c} = {sc.getConf().get(c, 'not set')}")

# ------------------------------------------------------------
# 6. DataFrame path — Tungsten (no Kryo needed)
#    DataFrames use Tungsten binary format internally; Kryo is irrelevant here.
# ------------------------------------------------------------
df = spark_kryo.range(100_000) \
    .withColumn("v", (col("id") % 10).cast("double"))
df.groupBy("v").count().show()

print("Serialization examples complete.")
spark_kryo.stop()
