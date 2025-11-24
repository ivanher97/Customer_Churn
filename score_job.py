from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import when


if __name__ == "__main__":
    spark = SparkSession.builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    
    pipe_loaded = PipelineModel.load("./models/random_forest_pipeline_v1")

    telco_data = spark.read.csv('./data/Telco-Customer-Churn.csv', header = True, inferSchema = True)
    telco_data = telco_data.withColumn('TotalCharges', col("TotalCharges").cast(DoubleType()))

    telco_clean = telco_data.na.drop(subset=["TotalCharges"])
    telco_clean = telco_clean.withColumn("label", when(col("Churn") == 'Yes', 1).otherwise(0))

    res = pipe_loaded.transform(telco_clean)

    res.select('customerID','prediction').write.mode('overwrite').option("header", "true").csv('./data/prediction')

    spark.stop()
