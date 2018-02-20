/*
  Linear regression model evaluation in Apache Spark with Scala
*/
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

def main(): Unit = {
  // start a Spark session
  val spark = SparkSession.builder().appName("LinearRegressionModelEvaluation").getOrCreate

  // training data
  val data = spark.read.format("libsvm").load("data/linear_regression_data.txt")
  data.printSchema

  // train/test data
  val Array(trainingData, testData) = data.randomSplit(Array(0.9, 0.1), seed=12345)

  // linear regression
  val linearRegression = new LinearRegression()

  // grid of parameters to search optimal hyperparameters; testing all combinations
  val parameterGrid = new ParamGridBuilder().addGrid(linearRegression.regParam, Array(0.1, 0.01)).
                                             addGrid(linearRegression.fitIntercept).
                                             addGrid(linearRegression.elasticNetParam, Array(0.0, 0.5, 1.0)).
                                             build() // fitIntercept is boolean, no need to specify values

  // train/validation split
  val trainValidationSplit = new TrainValidationSplit().setEstimator(linearRegression).
                                                        setEvaluator(new RegressionEvaluator).
                                                        setEstimatorParamMaps(parameterGrid).
                                                        setTrainRatio(0.8)

  // find best hyperparameters
  val model = trainValidationSplit.fit(trainingData)

  model.transform(testData).select("features", "label", "prediction").show()

  // model.bestModel

  spark.stop()
}

main()
