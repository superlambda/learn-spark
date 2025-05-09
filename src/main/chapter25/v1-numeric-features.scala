import org.apache.spark.sql.DataFrame

val rootPath: String = "/Users/liuyingjie/spark_workspace/house-prices-advanced-regression-techniques"
val filePath: String = s"${rootPath}/train.csv"

val trainDF: DataFrame = spark.read.format("csv").option("header", true).load(filePath)

// 所有数值型字段
val numericFields: Array[String] = Array("LotFrontage", "LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea")
val labelFields: Array[String] = Array("SalePrice")

// 抽取所有数值型字段
val selectedFields: DataFrame = trainDF.selectExpr((numericFields ++ labelFields): _*)

import org.apache.spark.sql.types.IntegerType

var typedFields = selectedFields

for (field <- (numericFields ++ labelFields)) {
  typedFields = typedFields.withColumn(s"${field}Int",col(field).cast(IntegerType)).drop(field)
}

import org.apache.spark.ml.feature.VectorAssembler

val features: Array[String] = numericFields.map(_ + "Int").toArray
val assembler = new VectorAssembler().setInputCols(features).setOutputCol("features").setHandleInvalid("skip")

var featuresAdded: DataFrame = assembler.transform(typedFields)
for (feature <- features) {
  featuresAdded = featuresAdded.drop(feature)
}

val Array(trainSet, testSet) = featuresAdded.randomSplit(Array(0.7, 0.3))

import org.apache.spark.ml.regression.LinearRegression

val lr = new LinearRegression().setLabelCol("SalePriceInt").setFeaturesCol("features").setMaxIter(1000)
val lrModel = lr.fit(trainSet)

val trainingSummary = lrModel.summary
println(s"Root Mean Squared Error (RMSE) on train data: ${trainingSummary.rootMeanSquaredError}")
// RMSE: 38288.77947156114

import org.apache.spark.ml.evaluation.RegressionEvaluator

val predictions: DataFrame = lrModel.transform(testSet).select("SalePriceInt", "prediction")
val evaluator = new RegressionEvaluator().setLabelCol("SalePriceInt").setPredictionCol("prediction").setMetricName("rmse")
val rmse = evaluator.evaluate(predictions)
println("Root Mean Squared Error (RMSE) on test data = " + rmse)

