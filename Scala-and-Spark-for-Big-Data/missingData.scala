/*
 Handling missing/null data
*/

import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

val spark = SparkSession.builder().getOrCreate

val df = spark.read.option("header", "true").
                    option("inferSchema", "true").
                    csv("data/ContainsNull.csv")

// show schema
df.printSchema

// show dataset
df.show

// drop all rows with null values
df.na.drop().show

// drop all rows with at least 2 null values
df.na.drop(2).show

// filling-in default values to all int-typed columns
df.na.fill(100).show

// filling-in default values to all string-typed columns
df.na.fill("Missing").show

// filling-in default values to column Name
df.na.fill("Noname", Array("Name")).show

df.describe().show

// setting defaults for multiple columns
val dfs = df.na.fill(400.5, Array("Sales"))
dfs.na.fill("Unnamed", Array("Name")).show
