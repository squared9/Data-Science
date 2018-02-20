/*
  Logistic regression operations in Apache Spark with Scala
*/
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

object LogisticRegressionAdvertising {

  def main(): Unit = {
    // start a Spark session
    val spark = SparkSession.builder().appName("LogisticRegressionAdvertising").getOrCreate

    // training data
    val data = spark.read.option("header", "true").
                          option("inferSchema", "true").
                          format("csv").
                          load("data/advertising.csv")

    data.printSchema

    val columns = data.columns
    val firstRow = data.head(1)(0)
    println("\n")
    println("Data preview")
    for (index <- Range(1, columns.length)) {
      print(columns(index))
      print(": ")
      println(firstRow(index))
    }

    // prepare feature <-> label mapping
    val df = (data.select(data("Clicked on Ad").as("label"),
              $"Daily Time Spent on Site",
              $"Age",
              $"Area Income",
              $"Daily Internet Usage",
              $"Timestamp",
              $"Male"))

    val dfh = df.withColumn("Hour", hour(df("Timestamp")))

    val dfp = dfh.na.drop()

    // SVMlib model
    val assembler = new VectorAssembler().
                            setInputCols(Array("Daily Time Spent on Site",
                                               "Age",
                                               "Area Income",
                                               "Daily Internet Usage",
                                               "Hour",
                                               "Male")).
                                               setOutputCol("features")

    val Array(training, test) = dfp.randomSplit(Array(0.7, 0.3), seed=12345)

    // linear regression
    val logisticRegression = new LogisticRegression()//.setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    val pipeline = new Pipeline().setStages(Array(assembler, logisticRegression))

    // train model
    val model = pipeline.fit(training)

    val testResults = model.transform(test)

    testResults.printSchema

    val predictionAndLabels = testResults.select($"prediction", $"label").as[(Double, Double)].rdd

    val metrics = new MulticlassMetrics(predictionAndLabels)

    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    print("Accuracy: ")
    println(metrics.accuracy)

    // resulting linear regression
    // println(s"Logistic regression: coefficients=${model.coefficients} intercept=${model.intercept}")

    // statistics and metrics
    // val summary = model.summary
    // println("Training summary:")
    // println(s"Number of iterations: ${summary.totalIterations}")
    // println(s"Objective history: ${summary.objectiveHistory.toList}")

    spark.stop()
  }
}

LogisticRegressionAdvertising.main()
