import coalitional_methods as coal
import drawing
import random

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

dataset = load_iris(as_frame=True)
X, y = dataset.data, dataset.target

model = RandomForestClassifier()
model.fit(X.values, y.values.flatten())

spearman_infs_comp25 = coal.coalitional_method(
    X, y, model, 0.25, method="spearman", complexity=False
)

abs(spearman_infs_comp25).mean().sort_values(ascending=True).plot(
    kind="barh", color="dimgray"
)

drawing.draw_avg_influence_per_class(abs(spearman_infs_comp25), y, dataset.target_names)


rand_int = random.randrange(X.shape[0])
drawing.draw_influence_instance(spearman_infs_comp25, y, dataset.target_names, rand_int)
