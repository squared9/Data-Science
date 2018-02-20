/*
  Time & date operations in Apache Spark with Scala
*/
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

val spark = SparkSession.builder().getOrCreate

val df = spark.read.option("header", "true").
                    option("inferSchema", "true").
                    csv("data/CitiGroup2006_2008")

// show schema
df.printSchema

// show header
df.head(3)

df.select(month(df("Date"))).show
df.select(year(df("Date"))).show

// observe yearly averages; 2008 was a crisis, clearly to be seen from the data
val dfy = df.withColumn("Year", year(df("Date")))
val dfy_avg = dfy.groupBy("Year").mean()
dfy_avg.select($"Year", $"avg(Close)").show()

val dfy_min = dfy.groupBy("Year").min()
dfy_min.select($"Year", $"min(Close)").show()
