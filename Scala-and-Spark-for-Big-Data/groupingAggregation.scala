/*
 Grouping and aggregation functions
*/

import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

val spark = SparkSession.builder().getOrCreate

val df = spark.read.option("header", "true").
                    option("inferSchema", "true").
                    csv("data/Sales.csv")

// show schema
df.printSchema

// show dataset
df.show

// grouping
df.groupBy("Company").mean().show
df.groupBy("Company").min().show
df.groupBy("Company").max().show
df.groupBy("Company").sum().show

// aggregate functions
df.select(sum("Sales")).show
df.select(countDistinct("Sales")).show
df.select(sumDistinct("Sales")).show
df.select(variance("Sales")).show
df.select(stddev("Sales")).show
df.select(collect_set("Sales")).show

// ordering results
df.orderBy("Sales").show
df.orderBy($"Sales".desc).show
