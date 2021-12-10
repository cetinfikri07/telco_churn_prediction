import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
#############################################
# GENERAL
#############################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)

    print("##################### Types #####################")
    print(dataframe.dtypes)

    print("##################### Head #####################")
    print(dataframe.head(head))

    print("##################### Tail #####################")
    print(dataframe.tail(head))

    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.

    """


    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car




#############################################
# CATEGORICAL
#############################################
os.makedirs("figures/cat_summary_plots")
os.path.exists("figures/cat_summary_plots")

if not os.path.exists("figures/cat_summary_plots"):
    print("dsfgh")


def cat_summary(dataframe, col_name, plot=False,directory = "figures"):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        plt.figure(figsize=(10, 8))

        ax = sns.countplot(x=col_name, data=dataframe, palette="rocket")

        plt.xlabel(col_name, fontsize=12)
        plt.ylabel("# of Clients", fontsize=12)
        plt.ylim(0, 7500)
        plt.xticks(list(range(0,len(dataframe[col_name].unique()))), dataframe[col_name].unique(), fontsize=11)

        # create folder and save figures

        if not os.path.exists("figures/cat_summary_plots"):
            os.makedirs("figures/cat_summary_plots")

        plt.savefig(directory + "/cat_summary_plots" + "/" + col_name + ".png")

        for p in ax.patches:
            ax.annotate((p.get_height()), (p.get_x() + 0.30, p.get_height() + 300), fontsize=14)



#############################################
# NUMERICAL
#############################################

def num_summary(dataframe, numerical_col, plot=False, directory = "figures",kde = "False"):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(10, 8))
        sns.histplot(data = dataframe,x = numerical_col,kde = kde)

        if not os.path.exists("figures/num_summary_plots"):
            os.makedirs("figures/num_summary_plots")

        plt.savefig(directory + "/num_summary_plots" + "/" + numerical_col + ".png")


#############################################
# TARGET
#############################################

def target_summary_with_cat(dataframe, target, categorical_col,plot = False,directory = "figures"):
    df = pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "FREQUENCY" : dataframe[categorical_col].value_counts(),
                        "FREQUENCY_PER" : 100 * dataframe[categorical_col].value_counts() / len(dataframe)
                        })
    df.index.name = categorical_col
    df = df.reset_index()
    print(df,end = "\n\n\n")

    if plot:
        plt.figure(figsize=(10, 8))
        sns.catplot(x=categorical_col, y="TARGET_MEAN", kind="bar", data=df)

        if not os.path.exists("figures/target_summary_with_cat"):
            os.makedirs("figures/target_summary_with_cat")

        plt.savefig(directory + "/target_summary_with_cat" + "/" + categorical_col + ".png")



def target_summary_with_num(dataframe, target, numerical_col,plot = [False,"mean"], directory = "figures" ):
    df = dataframe.groupby(target).agg({numerical_col: ["mean","median","std"]})
    print(df, end="\n\n\n")

    if plot[0]:

        df = df[numerical_col][plot[1]].to_frame()

        sns.catplot(x = df.index, y = plot[1],kind = "bar", data = df)
        plt.ylabel("{} of ".format(plot[1]) + numerical_col)

        if not os.path.exists("figures/target_summary_with_num"):
            os.makedirs("figures/target_summary_with_num")

        plt.savefig(directory + "/target_summary_with_num" + "/" + numerical_col + ".png")



def box_plot(dataframe,target,numerical_col,directory = "figures"):
    plt.figure(figsize=(10, 8))
    sns.boxplot(x=dataframe[target], y=dataframe[numerical_col])

    if not os.path.exists("figures/box_plot"):
        os.makedirs("figures/box_plot")

    plt.savefig(directory + "/box_plot" + "/" + numerical_col + ".png")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)













