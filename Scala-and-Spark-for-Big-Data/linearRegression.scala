/*
  Linear regression operations in Apache Spark with Scala
*/
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

def main(): Unit = {
  // start a Spark session
  val spark = SparkSession.builder().appName("LinearRegression").getOrCreate

  // training data
  val fileName = "data/linear_regression_data.txt"
  val trainingData = spark.read.format("libsvm").load(fileName)
  trainingData.printSchema

  // linear regression
  val linearRegression = new LinearRegression().setMaxIter(100).setRegParam(0.3).setElasticNetParam(0.8)

  // train model
  val model = linearRegression.fit(trainingData)

  // resulting linear regression
  println(s"Linear regression: coefficients=${model.coefficients} intercept=${model.intercept}")

  // statistics and metrics
  val summary = model.summary
  println("Training summary:")
  println(s"Number of iterations: ${summary.totalIterations}")
  println(s"Objective history: ${summary.objectiveHistory.toList}")
  summary.residuals.show
  println(s"RMSE: ${summary.rootMeanSquaredError}")
  println(s"r2: ${summary.r2}")

  spark.stop()
}

main()
