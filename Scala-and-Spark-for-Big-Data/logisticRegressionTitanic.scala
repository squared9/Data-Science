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

object LogisticRegressionTitanic {

  def main(): Unit = {
    // start a Spark session
    val spark = SparkSession.builder().appName("LogisticRegressionTitanic").getOrCreate

    // training data
    val data = spark.read.option("header", "true").
                          option("inferSchema", "true").
                          format("csv").
                          load("data/titanic.csv")

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
    val df = (data.select(data("Survived").as("label"),
              $"Pclass",
              $"Sex",
              $"Age",
              $"SibSp",
              $"Parch",
              $"Fare",
              $"Embarked"))

    val dfp = df.na.drop()

    // converting strings to numbers as indexes
    val genderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
    val embarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkedIndex")

    // One-hot encode indices
    val genderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
    val embarkEncoder = new OneHotEncoder().setInputCol("EmbarkedIndex").setOutputCol("EmbarkedVec")

    // SVMlib model
    val assembler = new VectorAssembler().
                            setInputCols(Array("Pclass",
                                               "SexVec",
                                               "Age",
                                               "SibSp",
                                               "Parch",
                                               "Fare",
                                               "EmbarkedVec")).
                                               setOutputCol("features")


    val Array(training, test) = dfp.randomSplit(Array(0.7, 0.3), seed=12345)

    // linear regression
    val logisticRegression = new LogisticRegression()//.setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

    val pipeline = new Pipeline().setStages(Array(genderIndexer, embarkIndexer, genderEncoder, embarkEncoder, assembler, logisticRegression))

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

LogisticRegressionTitanic.main()
