/*
  K-Means operations in Apache Spark with Scala
*/
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

object KMeansApp {

  def main(): Unit = {
    // start a Spark session
    val spark = SparkSession.builder().appName("KMeans").getOrCreate

    // training data
    val dataset = spark.read.format("libsvm").load("data/clustering_data.txt")

    val kMeans = new KMeans().setK(2).setSeed(1L)
    val model = kMeans.fit(dataset)

    val WSSSE = model.computeCost(dataset)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    println("Cluster centers:")
    model.clusterCenters.foreach(println)

    spark.stop()
  }
}

KMeansApp.main()
