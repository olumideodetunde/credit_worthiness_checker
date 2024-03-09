'''This module contains the Plot class that holds methods for visualising the data'''
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(PROJECT_ROOT)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_prep import logger

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
        plt.ylabel("Percentage (%)")
        plt.xlabel("Columns")
        plt.xticks(rotation=90)
        plt.savefig(f"{self.output_dir}/eda_data_completeness.png", bbox_inches="tight")

    def viz_distribution(self, feature_type:str, cols:list, plot_type:str) -> None:
        '''This method visualises the distribution of the columns 
           using a defined plot type (histogram, boxplot, etc)'''
        num_df = self.df[cols]
        figs, axes = plt.subplots(len(cols),1)
        for i, column in enumerate(num_df.columns):
            num_df[column].plot(
                kind = plot_type,
                ax = axes.flatten()[i],
                fontsize = "large"
            ).set_title(column)
        figs.set_size_inches(7, 8)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/eda_{feature_type}_distribution.png", bbox_inches="tight")

    def viz_bivariate_relationship(self, x:str, y:str) -> None:
        '''This method visualises the bivariate relationship between two 
           numeric columns using a scatter plot'''
        plt.figure(figsize=(6, 6))
        plt.scatter(self.df[x], self.df[y])
        plt.xlabel(x.upper())
        plt.ylabel(y.upper())
        plt.title(f"{x} vs {y}")
        plt.savefig(f"{self.output_dir}/eda_bivariate_relationship.png", bbox_inches="tight")

    def viz_linear_correlation(self, subset_col) -> None:
        '''This method visualises the linear correlation between
           the numeric subset columns/features'''
        corr = self.df[subset_col].corr(method='pearson')
        sns.heatmap(corr, annot=True)
        plt.savefig(f"{self.output_dir}/eda_linear_correlation.png", bbox_inches="tight")

    def viz_monthly_trend(self, date_cols:list):
        '''This method visualises the monthly trend of the data'''
        self._convert_to_date(date_cols)
        self.df["month"] = self.df[date_cols[0]].dt.month
        self.df.groupby('month')['case_id'].count().plot(kind='line')
        plt.title("Monthly trend")
        plt.xlabel("Months")
        plt.ylabel("Number of cases")
        plt.savefig(f"{self.output_dir}/eda_monthly_trend.png", bbox_inches="tight")

def main(file:str, output:str) -> None:
    '''This is  a main function that generates the defined plots for data visualisation'''
    example = Plot(file_path=file, output_dir=output)
    example.load_data()
    example.viz_completeness()
    example.viz_distribution(cols=["debt", "income"], feature_type="numeric",
                                     plot_type="hist")
    example.viz_bivariate_relationship(x="debt", y="income")
    example.viz_linear_correlation(subset_col=["debt", "income", "no_of_children"])
    example.viz_monthly_trend(date_cols=['date_decision', 'dateofbirth'])

#%%
if __name__ == "__main__":
    PROJECT_STAGE = "Data Visualisation"
    try:
        logger.info(">>>>> Generating plots for data visualisation")
        main(file="data/output/train.parquet", output="artifacts/plots")
        logger.info(">>>> Plots generated successfully <<<<<<<")
        logger.info(">>>>> Ended process for data visualisation <<<<< %s", PROJECT_STAGE)

    except Exception as e:
        logger.exception(e)
        raise e

#End of file