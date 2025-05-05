import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.ml.feature._
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.ml.{Pipeline, PipelineStage}

// rootPath为房价预测数据集根目录
val rootPath: String = "/Users/liuyingjie/spark_workspace/house-prices-advanced-regression-techniques"
val filePath: String = s"${rootPath}/train.csv"

// 读取CSV文件
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

// 所有非数值型字段
val categoricalFields: Array[String] = Array(
  "MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig",
  "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual",
  "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd",
  "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure",
  "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual",
  "Functional", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual",
  "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold",
  "SaleType", "SaleCondition"
)

val indexFields: Array[String] = categoricalFields.map(_ + "Index")

// StringIndexer
val stringIndexer = new StringIndexer().setInputCols(categoricalFields).setOutputCols(indexFields).setHandleInvalid("keep")

// 转换后用于特征的字段名
val numericFeatures: Array[String] = numericFields.map(_ + "Int")

// 缺失值填补（只对存在缺失值的字段进行）
val imputer = new Imputer().setInputCols(Array("LotFrontageInt", "MasVnrAreaInt")).setOutputCols(Array("LotFrontageInt", "MasVnrAreaInt")).setStrategy("mean")

// 特征向量组合器
val vectorAssembler = new VectorAssembler().setInputCols(numericFeatures ++ indexFields).setOutputCol("features").setHandleInvalid("keep")

// 特征离散化
val vectorIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(30).setHandleInvalid("keep")

// 回归器
val gbtRegressor = new GBTRegressor().setLabelCol("SalePriceInt").setFeaturesCol("indexedFeatures").setMaxIter(30).setMaxDepth(5).setMaxBins(128)

// 构建Pipeline
val pipeline = new Pipeline().setStages(Array(
  imputer,
  stringIndexer,
  vectorAssembler,
  vectorIndexer,
  gbtRegressor
))

// 保存未拟合的Pipeline
val savePath: String = "/Users/liuyingjie/spark_workspace/house-prices-advanced-regression-techniques"
pipeline.write.overwrite().save(s"${savePath}/unfit-gbdt-pipeline")

// 划分训练/验证集
val Array(trainingData, validationData) = engineeringDF.randomSplit(Array(0.7, 0.3))

// 拟合模型
val pipelineModel = pipeline.fit(trainingData)

import org.apache.spark.sql.DataFrame

val rootPath: String = "/Users/liuyingjie/spark_workspace/house-prices-advanced-regression-techniques"

val filePath: String = s"${rootPath}/test.csv"

// 加载test.csv，并创建DataFrame
var testData: DataFrame = spark.read.format("csv").option("header", true).load(filePath)

// 所有数值型字段
val numericFields: Array[String] = Array("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea")

// 所有非数值型字段
val categoricalFields: Array[String] = Array("MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType", "SaleCondition")

import org.apache.spark.sql.types.IntegerType

// 注意，test.csv没有SalePrice字段，也即没有Label
for (field <- (numericFields)) {
testData = testData.withColumn(s"${field}Int",col(field).cast(IntegerType)).drop(field)
}

val predictions = pipelineModel.transform(testData)

// 验证模型
import org.apache.spark.ml.evaluation.RegressionEvaluator

// 提取提交需要的字段（Id + prediction）
val submission = predictions.select(col("Id"), col("prediction").alias("SalePrice"))
submission.show(10)

