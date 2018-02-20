/*
  Logistic regression operations in Apache Spark with Scala
*/
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

object LogisticRegressionWithElasticNet {

  def main(): Unit = {
    // start a Spark session
    val spark = SparkSession.builder().appName("LinearRegressionWithElasticNet").getOrCreate

    // training data
    val trainingData = spark.read.format("libsvm").load("data/logistic_regression_data.txt")

    // linear regression
    val logisticRegression = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    // train model
    val model = logisticRegression.fit(trainingData)

    // resulting linear regression
    println(s"Logistic regression: coefficients=${model.coefficients} intercept=${model.intercept}")

    // statistics and metrics
    val summary = model.summary
    println("Training summary:")
    println(s"Number of iterations: ${summary.totalIterations}")
    println(s"Objective history: ${summary.objectiveHistory.toList}")

    spark.stop()
  }
}

LogisticRegressionWithElasticNet.main()
