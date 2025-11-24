from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

if __name__ == "__main__":
    
    spark = SparkSession.builder.getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    
    print("\n" + "="*50)
    print("INICIANDO PROCESO DE ENTRENAMIENTO")
    print("="*50 + "\n")
    
    df = spark.read.parquet('./data/processed')
    train_data, test_data = df.randomSplit([0.8, 0.2], 42)
    
    ignore_cols = ['customerID', 'Churn', 'label']
    cat_col = [ nombre for (nombre, tipo) in df.dtypes if tipo == 'string'
              and nombre not in ignore_cols]
    
    num_col = [ nombre for (nombre, tipo) in df.dtypes if nombre not in cat_col
              and nombre not in ignore_cols ]
    
    cat_col_indx = [nombre + '_indx' for nombre in cat_col]
    indexer = StringIndexer(inputCols = cat_col,  
                            outputCols = cat_col_indx, handleInvalid = 'keep')
    
    cat_col_vect = [nombre + '_vect' for nombre in cat_col]
    encoder = OneHotEncoder(inputCols = cat_col_indx, outputCols = cat_col_vect)
    
    assembler_inputs = num_col +  cat_col_vect
    assembler = VectorAssembler(inputCols = assembler_inputs, outputCol = 'features')
    
    rfc = RandomForestClassifier(featuresCol = 'features', labelCol = 'label', seed = 42)
    
    stages = [indexer, encoder, assembler, rfc]
    
    pipeline = Pipeline(stages = stages)
    
    pgb = ParamGridBuilder().addGrid(rfc.numTrees, [20, 50]).addGrid(rfc.maxDepth, [5, 10]).build()
    
    cv = CrossValidator(estimator = pipeline,
                        estimatorParamMaps = pgb,
                        evaluator = BinaryClassificationEvaluator(),
                        numFolds = 3)
    
    cv_model = cv.fit(train_data)
    print("Entrenamiento finalizado. Ya tenemos al campeón.")
    
    predictions = cv_model.transform(test_data)
    
    bce = BinaryClassificationEvaluator()
    eva = bce.evaluate(predictions)
    print(f"Model AUC: {eva}")
    
    rf_model = cv_model.bestModel.stages[-1]
    
    def ExtractFeatureImp(featureImp, dataset, featuresCol):
        list_extract = []
        
        # Extraemos la metadata de la columna features
        meta = dataset.schema[featuresCol].metadata.get('ml_attr')
        if meta is None:
            print("No hay metadatos. Asegúrate de usar el DataFrame transformado.")
            return
            
        # Los metadatos pueden estar divididos en tipos (numéricos, binarios, nominales)
        attrs = meta.get('attrs')
        
        # Recorremos todos los atributos para reconstruir la lista completa ordenada por índice
        for attr_type, attr_list in attrs.items():
            for attr in attr_list:
                # Guardamos (Nombre, Score)
                list_extract.append((attr['name'], featureImp[attr['idx']]))
        
        # Ordenamos por importancia descendente (de mayor a menor)
        list_extract.sort(key=lambda x: x[1], reverse=True)
        
        # Imprimimos el Top 10
        print("=== TOP 10 FACTORES DE FUGA ===")
        for i, (name, score) in enumerate(list_extract[:10]):
            print(f"{i+1}. {name}: {score:.4f}")
    
    # Ejecutar la función usando tu modelo y tus predicciones
    # rf_model: El objeto Random Forest que sacaste antes (bestModel.stages[-1])
    # predictions: El dataframe resultante del transform
    ExtractFeatureImp(rf_model.featureImportances, predictions, "features")
    
    final_pipeline = cv_model.bestModel
    final_pipeline.write().overwrite().save("./models/random_forest_pipeline_v1")

    print('HABEMUS TERMINADO')

    spark.stop()