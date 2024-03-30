import pandas as pd
from ydata_profiling import ProfileReport
from IPython.core.display import display, HTML, Javascript
from highlight_text import fig_text
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


class myEDA:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(self.path)

    def Data_profiling(self):
        depict = ProfileReport(
            df=self.df,
            title="Data profiling",
            minimal=True,
            progress_bar=False,
            samples=None,
            correlations=None,
            interactions=None,
            explorative=True,
            dark_mode=True,
            notebook={"iframe": {"height": "600px"}},
            html={"style": {"primary_color": "red"}},
            missing_diagrams={"heatmap": False, "dendrogram": False},
        ).to_notebook_iframe()
        display(HTML(depict))

    def heatmap(self):
        self.df2 = self.df.copy()
        string_columns = self.df2.select_dtypes(include="object").columns
        # Drop columns with string data type
        self.df2 = self.df2.drop(columns=string_columns)
        mask = np.triu(np.ones_like(self.df2.corr(), dtype=bool))
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            self.df2.corr(),
            mask=mask,
            annot=True,
            linewidths=0.2,
            cbar=False,
            annot_kws={"size": 7},
            rasterized=True,
        )
        xy_label = dict(size=6)
        yticks, ylabels = plt.yticks()
        xticks, xlabels = plt.xticks()
        ax.set_xticklabels(xlabels, rotation=0, **xy_label)
        ax.set_yticklabels(ylabels, **xy_label)
        ax.grid(False)
        # fig_text(s="Variables Correlation Map")
        plt.tight_layout(rect=[0, 0.04, 1, 1.01])
        plt.gcf()
        plt.show()


if __name__ == "__main__":
    dataset_path = ".//Iris_Data.csv"
    seedata = myEDA(dataset_path)
    # print(seedata.df)
    # seedata.Data_profiling()
    seedata.heatmap()
