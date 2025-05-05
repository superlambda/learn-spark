import org.apache.spark.sql.DataFrame

val rootPath: String = "/Users/liuyingjie/spark_workspace/house-prices-advanced-regression-techniques"

val filePath: String = s"${rootPath}/train.csv"
val sourceDataDF: DataFrame = spark.read.format("csv").option("header", true).load(filePath)
// 导入StringIndexer
import org.apache.spark.ml.feature.StringIndexer

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
