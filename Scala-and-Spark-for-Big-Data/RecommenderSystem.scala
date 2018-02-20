// Databricks notebook source
import org.apache.spark.ml.recommendation.ALS

// COMMAND ----------

val ratings = spark.read.option("header", "true").option("inferSchema", "true").format("csv").load("/FileStore/tables/movie_ratings.csv")

// COMMAND ----------

ratings.head()

// COMMAND ----------

ratings.printSchema

// COMMAND ----------

val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
val model = als.fit(training)
val predictions = model.transform(test)

predictions.show()

// COMMAND ----------

import org.apache.spark.sql.functions._
val error = predictions.select(abs($"rating" - $"prediction"))
error.show

// COMMAND ----------

error.na.drop.describe().show
