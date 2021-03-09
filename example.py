import coalitional_methods as coal
import drawing
import random
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor






### CLASSIFICATION

problem_type = "Classification"

dataset = load_iris()
X, y = pd.DataFrame(dataset.data, columns=dataset.feature_names), pd.Series(dataset.target)



model = RandomForestClassifier()
model.fit(X.values, y.values.flatten())

spearman_infs_comp25 = coal.coalitional_method(
    X, y, model, 0.25, problem_type=problem_type, fvoid=None, look_at=None, method="spearman", complexity=False
)

abs(spearman_infs_comp25).mean().sort_values(ascending=True).plot(
    kind="barh", color="dimgray"
)
plt.show()


drawing.draw_avg_influence_per_class(abs(spearman_infs_comp25), y, dataset.target_names)


rand_int = random.randrange(X.shape[0])
drawing.draw_influence_instance(spearman_infs_comp25, y, dataset.target_names, rand_int, problem_type=problem_type)







### REGRESSION

problem_type = "Regression"

dataset = load_diabetes()
X, y = pd.DataFrame(dataset.data, columns=dataset.feature_names), pd.Series(dataset.target)


model = RandomForestRegressor()
model.fit(X.values, y.values.flatten())




spearman_infs_comp25 = coal.coalitional_method(
    X, y, model, 0.25, problem_type=problem_type, fvoid=None, look_at=None, method="spearman", complexity=False
)


spearman_infs_comp25.abs().mean().sort_values(ascending=True).plot(
    kind="barh", color="dimgray"
)
plt.show()



rand_int = random.randrange(X.shape[0])
drawing.draw_influence_instance(spearman_infs_comp25, y, None, rand_int, problem_type=problem_type)