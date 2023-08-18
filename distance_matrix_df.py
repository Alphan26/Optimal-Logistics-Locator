import pandas as pd

distance_matrix_df = pd.read_csv("dataset/Distance_Matrix.csv")

# investigate the dataset

print(distance_matrix_df.head(20))
print(distance_matrix_df.info())
print(distance_matrix_df.describe())
print(distance_matrix_df.isna().sum())

def detect_outliers(df, columns, z_score_threshold):
    outliers = pd.DataFrame()

    for column in columns:
        mean = df[column].mean()
        std_dev = df[column].std()
        df[column + '_z_score'] = (df[column] - mean) / std_dev

        column_outliers = df[abs(df[column + '_z_score']) > z_score_threshold]
        outliers = pd.concat([outliers, column_outliers], axis=0)

    return outliers


columns_to_check = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]
z_score_threshold = 3
#outliers = detect_outliers(distance_matrix_df, columns_to_check, z_score_threshold)