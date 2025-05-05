import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.{Pipeline, PipelineStage}

val savePath: String = "/Users/liuyingjie/spark_workspace/house-prices-advanced-regression-techniques"

// 加载之前保存到磁盘的Pipeline
val unfitPipeline = Pipeline.load(s"${savePath}/unfit-gbdt-pipeline")

// 获取Pipeline中的每一个Stage（Transformer或Estimator）
val formerStages = unfitPipeline.getStages

// 去掉Pipeline中最后一个组件，也即Estimator：GBTRegressor
val formerStagesWithoutModel = formerStages.dropRight(1)

import org.apache.spark.ml.regression.RandomForestRegressor

// 定义新的Estimator：RandomForestRegressor
val rf = new RandomForestRegressor().setLabelCol("SalePriceInt").setFeaturesCol("indexedFeatures").setNumTrees(30).setMaxDepth(5).setMaxBins(128)

// 将老的Stages与新的Estimator拼接在一起
val stages = formerStagesWithoutModel ++ Array(rf)

// 重新定义新的Pipeline
val newPipeline = new Pipeline().setStages(stages)

// 读取CSV文件
val rootPath: String = "/Users/liuyingjie/spark_workspace/house-prices-advanced-regression-techniques"
val filePath: String = s"${rootPath}/train.csv"
var engineeringDF: DataFrame = spark.read.format("csv").option("header", true).load(filePath)

// 所有数值型字段
val numericFields: Array[String] = Array(
  "LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
  "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
  "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars",
  "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea"
)

val labelFields: Array[String] = Array("SalePrice")

// 将所有数值型字段和标签字段转为Int，并移除原字段
for (field <- numericFields ++ labelFields) {
  engineeringDF = engineeringDF
    .withColumn(s"${field}Int", col(field).cast(IntegerType))
    .drop(field)
}

val Array(trainingData, testData) = engineeringDF.randomSplit(Array(0.7, 0.3))
 // 调用fit方法，触发Pipeline运转，拟合新模型
val pipelineModel = newPipeline.fit(trainingData)


