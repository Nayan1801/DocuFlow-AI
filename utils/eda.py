def describe_data(df):
    return df.describe(include='all')

def plot_missing_values(df):
    import matplotlib.pyplot as plt
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing.plot(kind='bar')
    plt.title("Missing Values")
    plt.show()

def correlation_heatmap(df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()
