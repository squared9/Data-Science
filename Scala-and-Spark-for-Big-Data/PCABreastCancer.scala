/*
  K-Means operations in Apache Spark with Scala
*/
import org.apache.spark.ml.feature.{PCA, VectorAssembler, StandardScaler}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

object PCABreastCancer {

  def main(): Unit = {
    // start a Spark session
    val spark = SparkSession.builder().appName("PCABreastCancer").getOrCreate

    // training data
    val data = spark.read.option("header", "true").
                          option("inferSchema", "true").
                          format("csv").
                          load("data/Cancer_Data.csv")

    val columns = data.columns
    val firstRow = data.head(1)(0)
    println("\n")
    println("Data preview")
    for (index <- Range(1, columns.length)) {
      print(columns(index))
      print(": ")
      println(firstRow(index))
    }

    // column names
    val columnNames = (Array("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
      "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
      "radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error",
      "concavity error", "concave points error", "symmetry error", "fractal dimension error", "worst radius",
      "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity",
      "worst concave points", "worst symmetry", "worst fractal dimension"))

    // SVMlib model
    val assembler = new VectorAssembler().setInputCols(columnNames).setOutputCol("features")

    val trainingData = assembler.transform(data).select($"features")

    var scaler = new StandardScaler().setInputCol("features").
                                      setOutputCol("scaledFeatures").
                                      setWithStd(true).
                                      setWithMean(false)

    val scaledModel = scaler.fit(trainingData)

    val scaledData = scaledModel.transform(trainingData)

    val pca = new PCA().setInputCol("scaledFeatures").
                        setOutputCol("pcaFeatures").
                        setK(4).
                        fit(scaledData)

    val pcaDf = pca.transform(scaledData)
    val result = pcaDf.select("pcaFeatures")
    result.show

    spark.stop()
  }
}

PCABreastCancer.main()
