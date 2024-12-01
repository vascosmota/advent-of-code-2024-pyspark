from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.getOrCreate()

# Part I

# Read input file
input_raw = spark.read.csv('day01/input', header=False)
input_raw.show()

input_raw.columns
# ['_c0']

# Split the columns
input_split_res = input_raw.select(F.split(input_raw._c0, '   ', limit=2).alias('s'))
input_split_res['s']

input_split_res = input_split_res.withColumn("first_output", F.element_at('s', 1))
input_split_res = input_split_res.withColumn("second_output", F.element_at('s', 2))

input_split_res.show()

df_col_1 = input_split_res.select('first_output')
df_col_2 = input_split_res.select('second_output')

# Convert the text to numerical
df_col_1 = df_col_1.withColumn('first_output_num', df_col_1["first_output"].cast(IntegerType()))
df_col_2 = df_col_2.withColumn('second_output_num', df_col_2["second_output"].cast(IntegerType()))

# Sort each column
df_col_1_sorted = df_col_1.sort(F.asc('first_output_num')).withColumn('row_id',F.monotonically_increasing_id())
df_col_2_sorted = df_col_2.sort(F.asc('second_output_num')).withColumn('row_id',F.monotonically_increasing_id())

df_merge = df_col_1_sorted.join(df_col_2_sorted, on='row_id')
# df_merge = df_col_1_sorted.withColumn('second_output', df_col_2_sorted.second_output)
df_merge.show()

# Compute the absolute diff as a new column
df_merge = df_merge.withColumn('diff_abs', F.abs(df_merge['first_output_num'] - df_merge['second_output_num']))
df_merge.show()
# Sum the absolute diff

df_merge.agg(F.sum(df_merge['diff_abs'])).collect()


# Part II

# Similarity Score = Number * # of occurrences on the left list * # of occurrences on the right list
df_col_1_counts = (
    df_col_1_sorted
    .groupBy('first_output_num')
    .count()
    .withColumnRenamed('first_output_num', 'num')
    .withColumnRenamed('count', 'l_count')
)
df_col_1_counts.show()

df_col_2_counts = (
    df_col_2_sorted
    .groupBy('second_output_num')
    .count()
    .withColumnRenamed('second_output_num', 'num')
    .withColumnRenamed('count', 'r_count')
)

df_count_merge = (
    df_col_1_counts
    .join(df_col_2_counts, on='num')
    .fillna(0)
)
df_count_merge = df_count_merge.withColumn('sym_score', df_count_merge['num']* df_count_merge['l_count'] * df_count_merge['r_count'])
df_count_merge.show()

df_count_merge.agg(F.sum(df_count_merge['sym_score'])).collect()