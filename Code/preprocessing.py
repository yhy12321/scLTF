import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def preprocessingCSV(df, expressionFilename=None):
    if isinstance(df, pd.DataFrame):
        data = df
    else:
        data = pd.read_csv(expressionFilename, index_col=0, header=0)

    # 1. Remove genes
    data = data[(data[data.columns[1:]] > 0).sum(axis=1) >= 5]
    print('After preprocessing, {} genes remaining'.format(data.shape[0] - 1))


    # 3. Perform standardization
    # Log normalization
    data = np.log1p(data)

    # 3. Select the 2000 genes that are farthest from the gene mean
    data = data.loc[data.var(axis=1, numeric_only=True).nlargest(2000).index]

    print('After normalization, data shape:', data.shape)
    return data



# Unified management of data paths
# Raw gene expression matrix
raw_expression_path = ('path/to/raw_counts.csv')
# Preprocessed gene expression matrix (without PCA)
data_preprocessed_no_pca_path = ('path/to/preprocessed_data_no_pca.csv')
# Preprocessed gene expression matrix (with PCA)
data_pca_path = ('path/to/preprocessed_data_pca.csv')


# 读取文件
data = pd.read_csv(raw_expression_path, index_col=0)

print(f"Number of rows in the original data: {data.shape[0]}")
print(f"Number of columns in the original data: {data.shape[1]}")

print("\nNames of the first few cells and genes in the original data:")
print(f"Row names: {data.index[:5]}")
print(f"Column names: {data.columns[:5]}")

# Call the preprocessingCSV function to perform data preprocessing
data = preprocessingCSV(data)

# Save as a CSV file
data.to_csv(data_preprocessed_no_pca_path)

# Print the number of rows and columns of the saved CSV file
print(f"\nNumber of rows in the saved CSV file: {data.shape[0]}")
print(f"Number of columns in the saved CSV file: {data.shape[1]}")

# Perform dimensionality reduction using PCA
pca = PCA(n_components=50)
pca_data = pca.fit_transform(data.T)
pca_df = pd.DataFrame(data=pca_data, index=data.columns, columns=[f'PC{i+1}' for i in range(50)])

# Save as a CSV file
pca_df.to_csv(data_pca_path)

# Print the number of rows and columns of the saved CSV file after PCA
print(f"\nNumber of rows in the saved CSV file after PCA: {pca_df.shape[0]}")
print(f"Number of columns in the saved CSV file after PCA: {pca_df.shape[1]}")