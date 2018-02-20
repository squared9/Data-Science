/*
  K-Means operations in Apache Spark with Scala
*/
import org.apache.spark.ml.feature.PCA
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import spark.implicits._ // tp use Scala syntax for Spark

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

object PCAApp {

  def main(): Unit = {
    // start a Spark session
    val spark = SparkSession.builder().appName("PCA").getOrCreate

    // training data
    val data = Array (
      Vectors.sparse(5, Seq((1, 1.0), (3, 7.0))),
      Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
      Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
    )

    // do PCA
    val df = spark.createDataFrame(data.map(Tuple1.apply)).toDF("features")

    val pca = new PCA().setInputCol("features").
                        setOutputCol("pcaFeatures").
                        setK(3).
                        fit(df)

    val pcaDf = pca.transform(df)
    val result = pcaDf.select("pcaFeatures")
    result.show

    spark.stop()
  }
}

PCAApp.main()
