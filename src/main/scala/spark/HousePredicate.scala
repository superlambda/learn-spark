package spark

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.feature.ChiSqSelectorModel
import org.apache.spark.ml.feature.MinMaxScaler



import scala.collection.mutable.ArrayBuffer


object HousePredicate {
  def main(args: Array[String]): Unit = {
//    val rootPath: String = "/Users/liuyingjie/spark_workspace/house-prices-advanced-regression-techniques"
    val rootPath: String = "/home/ec2-user/spark_workspace/house-prices-advanced-regression-techniques"

    val spark = SparkSession.builder()
      .appName("HousePredicate")
      .master("spark://host0:7077")
      .getOrCreate()

    val filePath: String = s"${rootPath}/train.csv"
    val sourceDataDF: DataFrame = spark.read.format("csv").option("header", true).load(filePath)
    // 所有非数值型字段，也即StringIndexer所需的“输入列”
    val categoricalFields: Array[String] = Array("MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition")

    // 非数值字段对应的目标索引字段，也即StringIndexer所需的“输出列”
    val indexFields: Array[String] = categoricalFields.map(_ + "Index").toArray

    // 将engineeringDF定义为var变量，后续所有的特征工程都作用在这个DataFrame之上
    var engineeringDF: DataFrame = sourceDataDF

    // 核心代码：循环遍历所有非数值字段，依次定义StringIndexer，完成字符串到数值索引的转换
    for ((field, indexField) <- categoricalFields.zip(indexFields)) {

      // 定义StringIndexer，指定输入列名、输出列名
      val indexer = new StringIndexer()
        .setInputCol(field)
        .setOutputCol(indexField)

      // 使用StringIndexer对原始数据做转换
      engineeringDF = indexer.fit(engineeringDF).transform(engineeringDF)
    }
    // 删除掉原始的非数值字段列, call drop once
    engineeringDF = engineeringDF.drop(categoricalFields: _*)
    engineeringDF.select("GarageTypeIndex") .distinct() .show(5)

    // 所有数值型字段，共有27个
    val numericFields: Array[String] = Array("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea")

    // 预测标的字段
    val labelFields: Array[String] = Array("SalePrice")

    import org.apache.spark.sql.types.IntegerType

    // 将所有数值型字段，转换为整型Int
    for (field <- (numericFields ++ labelFields)) {
      engineeringDF = engineeringDF.withColumn(s"${field}Int",col(field).cast(IntegerType))
    }

    engineeringDF = engineeringDF.drop(numericFields: _*)

    import org.apache.spark.ml.feature.VectorAssembler

    // 所有类型为Int的数值型字段
    val numericFeatures: Array[String] = numericFields.map(_ + "Int").toArray

    // 定义并初始化VectorAssembler
    val assembler = new VectorAssembler()
      .setInputCols(numericFeatures)
      .setOutputCol("features")
      .setHandleInvalid("skip")
      .setMaxBins(128)

    // 在DataFrame应用VectorAssembler，生成特征向量字段"features"
    engineeringDF = assembler.transform(engineeringDF)
    engineeringDF.printSchema()
    // 定义并初始化ChiSqSelector
    val selector = new ChiSqSelector()
      .setFeaturesCol("features")
      .setLabelCol("SalePriceInt")
      .setNumTopFeatures(20)

    // 调用fit函数，在DataFrame之上完成卡方检验
    val chiSquareModel = selector.fit(engineeringDF)

    // 获取ChiSqSelector选取出来的入选特征集合(索引)
    val indexs: Array[Int] = chiSquareModel.selectedFeatures

    val selectedFeatures: ArrayBuffer[String] = ArrayBuffer[String]()

    // 根据特征索引值，查找数据列的原始字段名
    for (index <- indexs) {
      selectedFeatures += numericFields(index)
    }
    // 所有类型为Int的数值型字段
    // val numericFeatures: Array[String] = numericFields.map(_ + "Int").toArray

    // 遍历每一个数值型字段
    for (field <- numericFeatures) {

      // 定义并初始化VectorAssembler
      val assembler = new VectorAssembler()
        .setInputCols(Array(field))
        .setOutputCol(s"${field}Vector")
        .setHandleInvalid("skip")

      // 调用transform把每个字段由Int转换为Vector类型
      engineeringDF = assembler.transform(engineeringDF)
    }

    // 锁定所有Vector数据列
    val vectorFields: Array[String] = numericFeatures.map(_ + "Vector").toArray

    // 归一化后的数据列
    val scaledFields: Array[String] = vectorFields.map(_ + "Scaled").toArray

    // 循环遍历所有Vector数据列
    for (vector <- vectorFields) {

      // 定义并初始化MinMaxScaler
      val minMaxScaler = new MinMaxScaler()
        .setInputCol(vector)
        .setOutputCol(s"${vector}Scaled")
      // 使用MinMaxScaler，完成Vector数据列的归一化
      engineeringDF = minMaxScaler.fit(engineeringDF).transform(engineeringDF)
    }
    spark.stop()

  }

}
