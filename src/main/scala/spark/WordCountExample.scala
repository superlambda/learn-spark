package spark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object WordCountExample {
  def main2(args: Array[String]): Unit = {
    // Initialize Spark session
    val spark = SparkSession.builder()
      .appName("WordCountExample")
      .master("local[*]")  // Use local mode for testing in IDEA
      .config("spark.executor.extraJavaOptions", "--add-exports java.base/sun.nio.ch=ALL-UNNAMED")
      .config("spark.driver.extraJavaOptions", "--add-exports java.base/sun.nio.ch=ALL-UNNAMED")
      .getOrCreate()

    // Root directory for the data file
    val rootPath: String = "/Users/liuyingjie/git/learn-spark-with-scala/data/"
    val file: String = s"${rootPath}/wikiOfSpark.txt"

    // Read the file into an RDD
    val lineRDD: RDD[String] = spark.sparkContext.textFile(file)

    // Tokenize the text by splitting each line into words
    val wordRDD: RDD[String] = lineRDD.flatMap(line => line.split(" "))

    // Clean the words by removing empty strings
    val cleanWordRDD: RDD[String] = wordRDD.filter(word => !word.equals(""))

    // Map words to key-value pairs (word, 1)
    val kvRDD: RDD[(String, Int)] = cleanWordRDD.map(word => (word, 1))

    // Count words by reducing by key
    val wordCounts: RDD[(String, Int)] = kvRDD.reduceByKey((x, y) => x + y)

    // Sort word counts by frequency and get the top 5 words
    val topWords = wordCounts.map { case (k, v) => (v, k) }
      .sortByKey(false)
      .take(5)

    // Print the top 5 words
    topWords.foreach(println)

    // Stop the Spark session
    spark.stop()
  }
}
