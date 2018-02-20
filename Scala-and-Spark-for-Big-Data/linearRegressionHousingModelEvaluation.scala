/*
  Linear regression operations in Apache Spark with Scala
*/
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

def main(): Unit = {
  // start a Spark session
  val spark = SparkSession.builder().appName("LinearRegressionModelEvaluationHousing").getOrCreate

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

  // training data
  val assembledData = assembler.transform(df).select($"label", $"features")

  // train/test data
  val Array(trainingData, testData) = assembledData.select("label", "features").randomSplit(Array(0.7, 0.3), seed=12345)

  // linear regression
  val linearRegression = new LinearRegression().setMaxIter(100).setRegParam(0.3).setElasticNetParam(0.8)

  // grid of parameters to search optimal hyperparameters; testing all combinations
  val parameterGrid = new ParamGridBuilder().addGrid(linearRegression.regParam, Array(10000, 1000, 100, 10, 1, 0.1, 0.01)).
                                             addGrid(linearRegression.fitIntercept).
                                             addGrid(linearRegression.elasticNetParam, Array(0.0, 0.5, 1.0)).
                                             build() // fitIntercept is boolean, no need to specify values

  // train/validation split
  val trainValidationSplit = new TrainValidationSplit().setEstimator(linearRegression).
                                                        setEvaluator(new RegressionEvaluator().setMetricName("r2")).
                                                        setEstimatorParamMaps(parameterGrid).
                                                        setTrainRatio(0.8)

  // find best hyperparameters
  val model = trainValidationSplit.fit(trainingData)

  // validation metrics
  // println(model.validationMetrics)

  // best model
  // model.bestModel

  model.transform(testData).select("features", "label", "prediction").show()

  spark.stop()
}

main()
