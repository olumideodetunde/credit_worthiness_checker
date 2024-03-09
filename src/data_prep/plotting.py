'''This module contains the Plot class that holds methods for visualising the data'''
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class Plot:
    ''' This class contains methods for visualising the data'''
    def __init__(self, file_path:str, output_dir:str):
        self.file_path = file_path
        self.output_dir = output_dir
        self.df = None

    def _convert_to_date(self, date_cols:list) -> pd.DataFrame:
        '''This method converts the date columns to datetime format'''
        self.df[date_cols] = self.df[date_cols].apply(
            pd.to_datetime, format="%Y-%m-%d", errors="coerce"
        )
        return self.df

    def load_data(self) -> pd.DataFrame:
        '''This method loads the data from the parquet filepath'''
        self.df = pd.read_parquet(self.file_path)
        return self.df

    def viz_dtypes(self) -> None:
        '''This method shows the datatypes of the columns in the dataframe'''
        for col in self.df.columns:
            print(f"{col}: {self.df[col].dtype}")

    def viz_completeness(self) -> None:
        '''This methods checks the completeness of the dataframe 
           and visualises it using a barplot'''
        x = (self.df.isnull().sum())
        report = pd.DataFrame({
            "column": x.index,
            "missing": x.values,
            "percentage": (x.values / len(self.df)) * 100,
        }).sort_values(by="percentage", ascending=True)
        plt.figure(figsize=(5, 3))
        sns.barplot(x="column", y="percentage", data=report)
        plt.title("Percentage of missing data")
        plt.xticks(rotation=90)
        plt.savefig(f"{self.output_dir}/eda_data_completeness.png", bbox_inches="tight")

    def viz_numeric_summary(self, numeric_cols:list) -> pd.DataFrame:
        '''This method returns the summary statistics of the numerical columns'''
        num_df = self.df[numeric_cols]
        x = num_df.describe().T
        return x

    def viz_numeric_distribution(self, numeric_cols:list) -> None:
        '''This method visualises the distribution of the numerical columns 
           using histograms'''
        num_df = self.df[numeric_cols]
        figs, axes = plt.subplots(3,1)
        for i, column in enumerate(num_df.columns):
            num_df[column].plot(
                kind ="hist",
                ax = axes.flatten()[i],
                fontsize = "large"
            ).set_title(column)
        figs.set_size_inches(7, 15)
        plt.tight_layout()
        plt.show()

    def viz_categorical_barplot(self, cat_cols:list) -> None:
        '''This method visualises counts of each category in the 
           categorical columns using barplots'''
        cat_df = self.df[cat_cols]
        figs, axes = plt.subplots(3,1)
        for i, column in enumerate(cat_df.columns):
            counts = cat_df[column].value_counts()
            counts.plot(
                kind ="bar",
                ax = axes.flatten()[i],
                fontsize = "large"
            ).set_title(column)
        figs.set_size_inches(7, 15)
        plt.tight_layout()
        plt.show()

    def viz_bivariate_relationship(self, x:str, y:str) -> None:
        '''This method visualises the bivariate relationship between two 
           numeric columns using a scatter plot'''
        plt.figure(figsize=(6, 6))
        plt.scatter(self.df[x], self.df[y])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

    def viz_linear_correlation(self, subset_col) -> None:
        '''This method visualises the linear correlation between
           the subset columns/features'''
        corr = self.df[subset_col].corr(method='pearson')
        sns.heatmap(corr, annot=True)
        plt.show()

#%%
if __name__ == "__main__":
    example = Plot(file_path="data/output/train.parquet", output_dir="artifacts/plots")
    example.load_data()
    example._convert_to_date(date_cols=["date_decision", "dateofbirth"])
    example.viz_completeness()

