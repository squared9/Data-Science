/*
  K-Means operations in Apache Spark with Scala
*/
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

object KMeansWholesale {

  def main(): Unit = {
    // start a Spark session
    val spark = SparkSession.builder().appName("KMeansWholesale").getOrCreate

    // training data
    val data = spark.read.option("header", "true").
                          option("inferSchema", "true").
                          format("csv").
                          load("data/Wholesale_customers_data.csv")

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
    val df = (data.select(
              $"Fresh",
              $"Milk",
              $"Grocery",
              $"Detergents_Paper",
              $"Delicassen"))

    // SVMlib model
    val assembler = new VectorAssembler().
                            setInputCols(Array("Fresh",
                                               "Milk",
                                               "Grocery",
                                               "Detergents_Paper",
                                               "Delicassen")).
                                               setOutputCol("features")

    val trainingData = assembler.transform(df).select($"features")

    val kMeans = new KMeans().setK(16).setSeed(1L)
    val model = kMeans.fit(trainingData)

    val WSSSE = model.computeCost(trainingData)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    println("Cluster centers:")
    model.clusterCenters.foreach(println)

    spark.stop()
  }
}

KMeansWholesale.main()
