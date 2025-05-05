package spark

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object WordCountCollect {
  def main2(args: Array[String]): Unit = {
    // Initialize Spark session
    val spark = SparkSession.builder()
      .appName("WordCountCollect")
      .master("local[*]")  // Use local mode for testing in IDEA
      .config("spark.executor.extraJavaOptions", "--add-exports java.base/sun.nio.ch=ALL-UNNAMED")
      .config("spark.driver.extraJavaOptions", "--add-exports java.base/sun.nio.ch=ALL-UNNAMED")
      .getOrCreate()

    // Root directory for the data file
    val rootPath: String = "/Users/liuyingjie/git/learn-spark-with-scala/data/"
    val file: String = s"${rootPath}/wikiOfSpark.txt"

    // 读取文件内容
    val lineRDD: RDD[String] = spark.sparkContext.textFile(file)
    print("lineRDD.first")

    print(lineRDD.first)
    // res1: String = Apache Spark

    // 以行为单位做分词
    val wordRDD: RDD[String] = lineRDD.flatMap(line => line.split(" "))
    val cleanWordRDD: RDD[String] = wordRDD.filter(word => !word.equals(""))

    cleanWordRDD.take(3)
    // res2: Array[String] = Array(Apache, Spark, From)
    // 把RDD元素转换为（Key，Value）的形式
    val kvRDD: RDD[(String, Int)] = cleanWordRDD.map(word => (word, 1))
    // 按照单词做分组计数
    val wordCounts: RDD[(String, Int)] = kvRDD.reduceByKey((x, y) => x + y)

    wordCounts.collect
    // res3: Array[(String, Int)] = Array((Because,1), (Open,1), (impl…

    // Stop the Spark session
    spark.stop()
  }
}
