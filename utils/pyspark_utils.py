from pyspark.ml.feature import VectorAssembler

def combine_columns(columns, df, out_col):
    assembler = VectorAssembler(inputCols=columns,
                                outputCol=out_col)
    return assembler.transform(df)