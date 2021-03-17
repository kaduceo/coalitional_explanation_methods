"""
Coalitional explanation method (https://hal.archives-ouvertes.fr/hal-03138314)
Copyright (C) 2020 Gabriel Ferrettini <gabriel.ferrettini@irit.fr>
Copyright (C) 2020 Julien Aligon <julien.aligon@irit.fr>
Copyright (C) 2020 Chantal Soulé-Dupuy <chantal.soule-dupuy@irit.fr>

coalitional_methods.py
Copyright (C) 2020 Elodie Escriva, Kaduceo <elodie.escriva@kaduceo.com>
Copyright (C) 2020 Jean-Baptiste Excoffier, Kaduceo <jeanbaptiste.excoffier@kaduceo.com>

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

import sys
import pandas as pd
import numpy as np
import itertools

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import train_models, explain_groups_w_retrain, influence_calcul
from utils import check_all_attributs_groups, compute_subgroups_correlation
from utils import remove_inclusions, generate_subgroups_group, coal_penalisation


def compute_vifs(datas):
    """
    Compute VIF for each attribut in the dataset.

    Parameters
    ----------
    datas : pandas.DataFrame
        Dataframe of the input datas.

    Returns
    -------
    pandas.Series
        VIF for each attributs.

    """

    return pd.Series(
        [
            variance_inflation_factor(datas.assign(const=1).values, i)
            for i in range(datas.shape[1])
        ],
        index=datas.columns,
    )


def vif_grouping(datas, threshold, reverse=False):
    """
    Generate groups of attributs based on VIF ir reverse VIF methods

    Parameters
    ----------
    datas : pandas.DataFrame
         Dataframe of the input datas.
    threshold : float
        Correlation threshold between two attributes.
    reverse : boolean
        Define the method to use (Reverse or not). Default False.

    Returns
    -------
    groups : two-dimensional list
        List of groups of uncorrelated attributs based on the reverse VIF methods.

    """

    true_j = 0
    groups = []
    vifs = compute_vifs(datas)

    for i in range(datas.shape[1]):
        group = [i]
        datas_clone = datas.drop(datas.columns[i], axis=1)
        new_vifs = compute_vifs(datas_clone)

        for j in range(datas_clone.shape[1]):
            true_j = j + 1 if j >= 1 else j
            vif_formula = (
                (new_vifs[j] > vifs[true_j] * (1 - threshold * 0.05))
                if reverse
                else (new_vifs[j] < vifs[true_j] * (0.4 + threshold))
            )

            if vif_formula:
                group.append(true_j)

        group.sort()

        if not group in groups:
            groups.append(group)

    groups = check_all_attributs_groups(groups, datas.shape[1])

    return groups


def spearman_grouping(datas, threshold, reverse=False):
    """
    Generate groups of attributs based on Spearman or reverse Spearman methods

    Parameters
    ----------
    datas : pandas.DataFrame
        Dataframe of the input datas.
    threshold : float
        Correlation threshold between two attributes.
    reverse : boolean
        Define the method to use (Reverse or not). Default False.

    Returns
    -------
    groups : two-dimensional list
        List of groups of uncorrelated attributs based on the reverse Spearman methods.

    """

    groups = []
    spearman_matrix = datas.corr(method="spearman")

    for i in range(datas.shape[1]):
        group = [i]
        max_ = max(abs(spearman_matrix).iloc[i].drop([spearman_matrix.columns[i]]))
        min_ = min(abs(spearman_matrix).iloc[i].drop([spearman_matrix.columns[i]]))

        for j in range(spearman_matrix.shape[1]):
            if reverse:
                group_condition = (
                    min_ < 0.5
                    and j != i
                    and abs(spearman_matrix.iloc[i, j]) < min_ + max_ * threshold
                )
            else:
                group_condition = (
                    max_ > 0.1
                    and j != i
                    and np.abs(spearman_matrix.iloc[i, j]) > max_ * (1 - threshold)
                )

            if group_condition:
                group.append(j)

        group.sort()

        if not group in groups:
            groups.append(group)

    groups = check_all_attributs_groups(groups, datas.shape[1])

    return groups


def pca_grouping(datas, threshold):
    """
    Generate groups of attributs based on PCA methods

    Parameters
    ----------
    datas : pandas.DataFrame
         Dataframe of the input datas..
    threshold : float
        Correlation threshold between two attributes.
    scaler: boolean
        Flag for apply a scaler to the datas.

    Returns
    -------
    groups : two-dimensional list
        List of groups of correlated attributs based on the PCA methods.
    """

    groups = []
    pca = PCA().fit(datas)
    eigenvectors = pca.components_

    for vector in eigenvectors:
        group = []
        max_vect = max(abs(vector))

        for k in range(len(vector)):
            if abs(vector[k]) == max_vect:
                group.append(k)
            elif abs(vector[k]) > max_vect * (1 - threshold):
                group.append(k)

        group.sort()

        if not group in groups:
            groups.append(group)

    groups = check_all_attributs_groups(groups, datas.shape[1])

    return groups


def compute_number_distinct_subgroups(groups):
    subgroups_list = compute_subgroups_correlation(groups)
    subgroups_list.sort()

    return sum(1 for x in itertools.groupby(subgroups_list))


def find_alpha_rate(coal_function, n_rate, X, max_iterations=100):
    alpha = 0.5
    subgroups = coal_function(X, threshold=alpha)
    nb_subgroups = compute_number_distinct_subgroups(subgroups)

    (alpha_best, subgroups_best, nb_subgroups_best) = (alpha, subgroups, nb_subgroups)

    if nb_subgroups == n_rate:
        return subgroups, alpha

    i = 0

    while i < max_iterations:
        alpha = alpha + alpha / 2 if nb_subgroups < n_rate else alpha - alpha / 2
        subgroups = coal_function(X, threshold=alpha)
        nb_subgroups = compute_number_distinct_subgroups(subgroups)

        if nb_subgroups == n_rate:
            return subgroups, alpha

        if abs(nb_subgroups - n_rate) < abs(nb_subgroups_best - n_rate):
            (alpha_best, subgroups_best, nb_subgroups_best) = (
                alpha,
                subgroups,
                nb_subgroups,
            )

        i += 1

    return subgroups_best, alpha_best


def complexity_coal_groups(X, rate, grouping_function):
    n_total = 2 ** X.shape[1] - 1
    n_rate = int(np.round(n_total * rate, 0))
    coal_groups, alpha = find_alpha_rate(
        coal_function=lambda X_, threshold: remove_inclusions(
            grouping_function(X_, threshold)
        ),
        n_rate=n_rate,
        X=X,
        max_iterations=50,
    )

    return coal_groups


def compute_instance_coal_inf(raw_instance_inf, columns, relevant_groups):
    """ Coalitional method for one instance, when attributs overlap in groups (Ferrettini et al. 2020)"""

    influences = dict([(c, 0) for c in columns])
    denoms_shap = dict([(c, 0) for c in columns])

    for group in relevant_groups:
        subgroups = generate_subgroups_group(group)

        for subgroup in subgroups:
            for i in subgroup:
                pena = coal_penalisation(len(subgroup) - 1, len(group))
                denoms_shap[columns[i]] += pena
                influences[columns[i]] += influence_calcul(
                    pena, raw_instance_inf, subgroup, i
                )

    for i in columns:
        influences[i] /= denoms_shap[i]

    return influences


def compute_coalitional_influences(raw_groups_influences, X, relevant_groups):
    """Coalitional method for all instances, when attributs overlap in groups
    (Ferrettini et al. 2020)"""

    coalitional_influences = pd.DataFrame(columns=X.columns)

    for instance in X.index:
        raw_infs = raw_groups_influences[instance]
        influences = compute_instance_coal_inf(raw_infs, X.columns, relevant_groups)
        coalitional_influences = coalitional_influences.append(
            pd.Series(influences, name=instance)
        )

    return coalitional_influences


def coalitional_method(
    X,
    y,
    model,
    rate,
    problem_type,
    fvoid=None,
    look_at=None,
    method="spearman",
    reverse=False,
    complexity=False,
    scaler=False,
):
    methods = {"pca": pca_grouping, "spearman": spearman_grouping, "vif": vif_grouping}

    if method not in methods.keys():
        sys.stderr.write("ERROR: Invalid method.\n")
        return

    if X.shape[1] == 1:
        groups = [[0]]
    else:
        if method == "pca" and scaler:
            X = StandardScaler().fit_transform(X)
        if complexity:
            groups = complexity_coal_groups(X, rate, methods[method])
        else:
            groups = methods[method](X, rate)

    groups = compute_subgroups_correlation(groups) + [[]]

    pretrained_models = train_models(model, X, y, groups, problem_type, fvoid)
    raw_groups_influences = explain_groups_w_retrain(
        pretrained_models, X, problem_type, look_at
    )
    coalition_influences = compute_coalitional_influences(
        raw_groups_influences, X, groups
    )

    return coalition_influences
