"""
drawing.py
Copyright (C) 2020 Elodie Escriva, Kaduceo <elodie.escriva@kaduceo.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
import matplotlib.pyplot as plt


def draw_avg_influence_per_class(influences, labels, labels_name):
    complete_datas = influences.copy()
    complete_datas["labels"] = labels
    avg_infs_per_class = pd.DataFrame(columns=influences.columns)
    for c in range(labels.nunique()):
        temp_avg = (
            abs(complete_datas[complete_datas.labels == c])
            .mean()
            .drop(["labels"])
            .rename(labels_name[c])
        )
        avg_infs_per_class = avg_infs_per_class.append(temp_avg)
    avg_infs_per_class.T.plot.barh(rot=0, title="Average attribute influence by class")
    plt.xlabel("Mean absolute influence", fontweight="bold")
    plt.show()


def draw_influence_instance(influences, label, labels_name, id_instance):
    infs_instance = influences.iloc[id_instance].sort_values(ascending=True)
    print(influences.iloc[id_instance])
    print(infs_instance)
    colors = ["green" if x > 0 else "red" for x in infs_instance]
    plt.title(
        "Patient : {} ; Class : {}".format(id_instance, labels_name[label[id_instance]])
    )
    plt.xlabel("Influences")
    infs_instance.plot.barh(color=colors)
    plt.show()
