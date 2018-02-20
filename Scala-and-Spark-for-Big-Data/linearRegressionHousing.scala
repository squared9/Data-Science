/*
  Linear regression operations in Apache Spark with Scala
*/
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

def main(): Unit = {
  // start a Spark session
  val spark = SparkSession.builder().appName("LinearRegressionFromData").getOrCreate

  // training data
  val data = spark.read.option("header", "true").
                        option("inferSchema", "true").
                        format("csv").
                        load("data/Clean-USA-Housing.csv")

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
  val df = (data.select(data("Price").as("label"),
            $"Avg Area Income",
            $"Avg Area House Age",
            $"Avg Area Number of Rooms",
            $"Avg Area Number of Bedrooms",
            $"Area Population"))

  // SVMlib model
  val assembler = new VectorAssembler().
                          setInputCols(Array("Avg Area Income",
                                             "Avg Area House Age",
                                             "Avg Area Number of Rooms",
                                             "Avg Area Number of Bedrooms",
                                             "Area Population")).
                                             setOutputCol("features")

  val trainingData = assembler.transform(df).select($"label", $"features")

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
