import pandas as pd
import matplotlib.pyplot as plt


def draw_avg_influence_per_class(influences, labels, labels_name, colors=None):
    if colors is None:
        colors = ["#0d0385", "#00c0d3"]
    complete_datas = influences.copy()
    complete_datas["labels"] = labels
    avg_infs_per_class = pd.DataFrame(columns=influences.columns)
    for c in range(labels[labels.columns[0]].nunique()):
        temp_avg = (
            abs(complete_datas[complete_datas.labels == c])
            .mean()
            .drop(["labels"])
            .rename(labels_name[c])
        )
        avg_infs_per_class = avg_infs_per_class.append(temp_avg)
    avg_infs_per_class.T.plot.barh(
        color=colors, rot=0, title="Average attribute influence by class"
    )
    plt.xlabel("Mean absolute influence", fontweight="bold")
    plt.show()


def draw_influence_instance(influences, label, labels_name, id_instance):
    infs_instance = influences.iloc[id_instance].sort_values(ascending=True)
    colors = ["green" if x > 0 else "red" for x in infs_instance]
    plt.title("Patient : {} ; Class : {}".format(id_instance, labels_name[label]))
    plt.xlabel("Influences")
    influences.iloc[id_instance].plot.barh(color=colors)
    plt.show()
