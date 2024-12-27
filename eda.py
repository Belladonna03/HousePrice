import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def read_csv_file():
    filename = 'data/house_price_regression_dataset.csv'
    df = pd.read_csv(filename)
    return df

def check_dir(dir_name):
    try:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    except OSError as e:
        print(f"Error occurred while creating the directory: {e}")

def plot_corr_matrix(df, title, dir_name):
    corr_matrix = df.corr()

    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)

    check_dir(dir_name)
    filename = os.path.join(dir_name, f'{title}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_histogram(column, df, dir_name, bins=30, kde=True):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=bins, kde=kde, color='blue')
    title = f'Histogram for {column}'
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')

    check_dir(dir_name)
    filepath = os.path.join(dir_name, f'histogram_{column}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_boxplot(column, df, dir_name):
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column], color='blue')
    title = f'Boxplot for {column}'
    plt.title(title)
    plt.ylabel(column)

    check_dir(dir_name)
    filepath = os.path.join(dir_name, f'boxplot_{column}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.show()

def drop_duplicates(df):
    cnt_duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {cnt_duplicates}")

    if cnt_duplicates > 0:
        df_cleaned = df.drop_duplicates().reset_index(drop=True)
        return df_cleaned
    else:
        return df

def eda(df):
    print('Dataframe info:')
    print(df.info())
    print('Summary statistics of the dataframe:')
    print(df.describe())
    print('Column names of the dataset:')
    print(df.columns)

    plot_corr_matrix(df, title='Correlation Matrix Before Preprocessing', dir_name='eda')

    #The correlation with target column is 0.99
    df = df.drop('Square_Footage', axis=1)

    df = drop_duplicates(df)

    for column in df.columns:
        plot_histogram(column, df, dir_name='eda')
        plot_boxplot(column, df, dir_name='eda')

    plot_corr_matrix(df, title='Correlation Matrix After Preprocessing', dir_name='eda')

    return df


if __name__ == "__main__":
    df = read_csv_file()

    cleaned_df = eda(df)

    cleaned_df.to_csv('data/cleaned_dataset.csv', index=False)