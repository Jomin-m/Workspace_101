"""
PySpark Optimization: Broadcast Joins
Covers broadcast(), auto-broadcast threshold, and when to use/avoid it.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

spark = SparkSession.builder \
    .appName("Broadcast Join Optimization") \
    .config("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)  # 10 MB
    .getOrCreate()

# ------------------------------------------------------------
# Sample data
# Large fact table (millions of rows in production)
# Small dimension table (fits comfortably in memory)
# ------------------------------------------------------------
orders_data = [(i, i % 100, float(i * 2.5)) for i in range(200_000)]
orders = spark.createDataFrame(orders_data, ["order_id", "product_id", "amount"])

products_data = [(i, f"Product_{i}", f"Category_{i % 10}") for i in range(100)]
products = spark.createDataFrame(products_data, ["product_id", "product_name", "category"])

# ------------------------------------------------------------
# 1. Default sort-merge join — causes a full shuffle on both sides
# ------------------------------------------------------------
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)   # disable auto-broadcast
join_no_broadcast = orders.join(products, "product_id")
print("=== Sort-Merge Join plan ===")
join_no_broadcast.explain(mode="formatted")

# ------------------------------------------------------------
# 2. Explicit broadcast join — small table sent to every executor
#    No shuffle on the large table; reduces network I/O dramatically.
# ------------------------------------------------------------
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 10 * 1024 * 1024)

join_broadcast = orders.join(broadcast(products), "product_id")
print("=== Broadcast Join plan ===")
join_broadcast.explain(mode="formatted")   # look for BroadcastHashJoin
join_broadcast.groupBy("category").count().show()

# ------------------------------------------------------------
# 3. Auto-broadcast via threshold (no hint needed)
#    Spark checks table size at planning time. Works well for Parquet/ORC
#    where statistics are embedded in the file metadata.
# ------------------------------------------------------------
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", 50 * 1024 * 1024)  # 50 MB
auto_join = orders.join(products, "product_id")
print("=== Auto-Broadcast plan ===")
auto_join.explain(mode="formatted")

# ------------------------------------------------------------
# 4. Broadcast join with a lookup dict (alternative for simple key lookups)
#    Avoids a Spark join altogether — useful for tiny mappings.
# ------------------------------------------------------------
category_map = {row["product_id"]: row["category"] for row in products.collect()}
from pyspark.sql.functions import udf

category_udf = udf(lambda pid: category_map.get(pid, "Unknown"), StringType())
orders_with_cat = orders.withColumn("category", category_udf(col("product_id")))
orders_with_cat.show(5)

# ------------------------------------------------------------
# 5. When NOT to broadcast
#    - Table > available executor memory / broadcast threshold
#    - Both sides are large (use sort-merge or bucketing instead)
#    - Broadcast serialization itself becomes a bottleneck (> 1 GB tables)
# ------------------------------------------------------------

# ------------------------------------------------------------
# 6. Broadcast join with a filter — push filters BEFORE broadcasting
#    to reduce the size of the broadcast payload.
# ------------------------------------------------------------
hot_categories = products.filter(col("category").isin("Category_1", "Category_2"))
filtered_join = orders.join(broadcast(hot_categories), "product_id")
filtered_join.groupBy("category").sum("amount").show()

print("Broadcast join examples complete.")
spark.stop()
