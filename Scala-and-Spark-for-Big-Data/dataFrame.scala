/*
  Data frame operations in Apache Spark with Scala
*/
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").
                    option("inferSchema", "true").
                    csv("data/CitiGroup2006_2008")

/*
  Basic operations
*/

// take first 5 records
val head = df.head(5)

// display data frame columns first
df.columns
println("------------")

// show first 5 rows
for (row <- head) {
  println(row)
}

// show data frame description
df.describe().show()

// show only column Volume
df.select("Volume").show()

// show columns Data and Close
df.select($"Date", $"Close").show()

// compute range of high-low values
val dfn = df.withColumn("Range", df("High") - df("Low"))

dfn.printSchema()
dfn.show()

// rename column
dfn.select(dfn("Range").as("Largest Daily Difference")).show()

/*
  DataFrame operations
*/

// Scala notation

// two identical statements
df.filter($"Close" > 480).show()
df.filter("Close > 480").show()

// multiple conditions
df.filter($"Close" < 480 && $"High" < 480).show()

// Spark SQL notation
df.filter("Close < 480 AND High < 480").show()

// Collecting values
val df_low = df.filter("Close < 480 AND High < 480").collect()

// this throws an error
// df.filter($"High" == 484.80).show()

df.filter($"High" === 484.40).show()

// correlation
df.select(corr("High", "Low").as("Correlation")).show()
