/*
  Data frame operations in Apache Spark with Scala
*/
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

// start a Spark session
val spark = SparkSession.builder().getOrCreate()

// load Netflix stock data
val df = spark.read.option("header", "true").
                    option("inferSchema", "true").
                    csv("data/Netflix_2011_2016.csv")


// display column names
df.columns
println("------------")

// show schema
df.printSchema

// show first 5 rows
val head = df.head(5)
for (row <- head) {
  println(row)
}

// show data frame description
df.describe().show()

// compute ratio of high price vs volume
val dfn = df.withColumn("HV Ratio", df("High") / df("Volume"))

dfn.printSchema()
dfn.show()

// rename column
dfn.select(dfn("HV Ratio").as("Largest Daily Difference")).show()

// show day with peak high price
df.orderBy($"High".desc).show(1)

// mean close price
df.select(mean("Close")).show()

// minimum and maximumum volume
df.select(min("Volume"), max("Volume")).show()

// number of days close price under $600
df.select($"Close" < 600).count
df.filter($"Close" < 600).count
df.filter("Close < 600").count

// percentage of time when high price > $500
df.filter($"High" > 500).count * 1.0 / df.count * 100.0

// Pearson correlation between high price and volume
df.select(corr("High", "Volume")).show

// maximal high price per year
val dfy = df.withColumn("Year", year(df("Date")))
dfy.printSchema
dfy.groupBy("Year").max("High").show

df.groupBy(year(df("Date"))).max("High").show

// average close price for each month
df.groupBy(month(df("Date"))).mean("Close").show
