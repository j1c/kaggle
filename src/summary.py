import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def summarize(df):
    """Generates a summary of each column of input dataframe.

    From Kaggle.

    Parameters
    ----------
    df : pd.DataFrame

    Notes
    -----
    summarize(df).style.background_gradient(cmap='YlOrBr')
    """
    # pd.options.display.float_format = "{:,.2f}".format
    # print(f"data shape: {df.shape}")

    summ = pd.DataFrame(df.dtypes, columns=["data type"])

    summ["#missing"] = df.isnull().sum().values
    summ["%missing"] = df.isnull().sum().values / len(df) * 100
    summ["#unique"] = df.nunique().values

    desc = pd.DataFrame(df.describe(include="all").transpose())
    summ["min"] = desc["min"].values
    summ["max"] = desc["max"].values
    summ["average"] = desc["mean"].values
    summ["standard_deviation"] = desc["std"].values
    summ["first value"] = df.loc[0].values
    summ["second value"] = df.loc[1].values
    summ["third value"] = df.loc[2].values

    return summ


def plot_count(df: pd.DataFrame, col_list: list, title_name: str = "Train") -> None:
    """Draws the pie and count plots for categorical variables.

    Args:
        df (pd.core.frame.DataFrame): A pandas dataframe representing the data to be analyzed.
            This could be a training set, test set, etc.
        col_list (list): A list of categorical variable column names from 'df' to be analyzed.
        title_name (str): The title of the graph. Default is 'Train'.

    Returns:
        None. This function produces pie and count plots of the input data and displays them using matplotlib.
    """

    # Creating subplots with 2 columns for pie and count plots for each variable in col_list
    f, ax = plt.subplots(len(col_list), 2, figsize=(12, 5))
    plt.subplots_adjust(wspace=0.3)

    for col in col_list:
        # Computing value counts for each category in the column
        s1 = df[col].value_counts()
        N = len(s1)

        outer_sizes = s1
        inner_sizes = s1 / N

        # Colors for the outer and inner parts of the pie chart
        outer_colors = ["#FF6347", "#20B2AA"]
        inner_colors = ["#FFA07A", "#40E0D0"]

        # Creating outer pie chart
        ax[0].pie(
            outer_sizes,
            colors=outer_colors,
            labels=s1.index.tolist(),
            startangle=90,
            frame=True,
            radius=1.2,
            explode=([0.05] * (N - 1) + [0.2]),
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
            textprops={"fontsize": 14, "weight": "bold"},
            shadow=True,
        )

        # Creating inner pie chart
        ax[0].pie(
            inner_sizes,
            colors=inner_colors,
            radius=0.8,
            startangle=90,
            autopct="%1.f%%",
            explode=([0.1] * (N - 1) + [0.2]),
            pctdistance=0.8,
            textprops={"size": 13, "weight": "bold", "color": "black"},
            shadow=True,
        )

        # Creating a white circle at the center
        center_circle = plt.Circle((0, 0), 0.5, color="black", fc="white", linewidth=0)
        ax[0].add_artist(center_circle)

        # Barplot for the count of each category in the column
        sns.barplot(x=s1, y=s1.index, ax=ax[1], palette="coolwarm", orient="horizontal")

        # Customizing the bar plot
        ax[1].spines["top"].set_visible(False)
        ax[1].spines["right"].set_visible(False)
        ax[1].tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax[1].set_ylabel("")  # Remove y label

        # Adding count values at the end of each bar
        for i, v in enumerate(s1):
            ax[1].text(
                v, i + 0.1, str(v), color="black", fontweight="bold", fontsize=14
            )

        # Adding labels and title
        plt.setp(ax[1].get_yticklabels(), fontweight="bold")
        plt.setp(ax[1].get_xticklabels(), fontweight="bold")
        ax[1].set_xlabel(col, fontweight="bold", color="black", fontsize=14)

    # Setting a global title for all subplots
    f.suptitle(
        f"{title_name} Dataset Distribution of {col}",
        fontsize=20,
        fontweight="bold",
        y=1.05,
    )

    # Adjusting the spacing between the plots
    plt.tight_layout()
    plt.show()
