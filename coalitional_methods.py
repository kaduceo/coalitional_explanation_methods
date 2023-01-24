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
from tqdm import tqdm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import train_models, explain_groups_w_retrain, influence_calcul
from utils import check_all_attributs_groups, compute_subgroups_correlation
from utils import remove_inclusions, generate_subgroups_group, coal_penalisation


def compute_vifs(datas):
    """
    Compute Variance Inflation Factor for each attribut in the dataset.

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
    Generate groups of attributs based on VIF or reverse VIF method.

    Parameters
    ----------
    datas : pandas.DataFrame
        Dataframe of the input datas.
    threshold : float
        Correlation threshold between two attributes.
    reverse : boolean, default=False
        Define the method to use : reverse or not.

    Returns
    -------
    groups : two-dimensional list
        Groups of (un)correlated attributs based on the (reverse) VIF method.

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
    reverse : boolean, default=false
        Define the method to use : reverse or not.

    Returns
    -------
    groups : two-dimensional list
        Groups of (un)correlated attributs based on the (reverse) Spearman method.

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


def pca_grouping(datas, threshold, reverse=False):
    """
    Generate groups of attributs based on PCA method.

    Parameters
    ----------
    datas : pandas.DataFrame
         Dataframe of the input datas.
    threshold : float
        Correlation threshold between two attributes.

    Returns
    -------
    groups : two-dimensional list
        Groups of correlated attributs based on the PCA method.
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
    """
    Compute the alpha to achieve the wanted complexity rate.
    Bisection method is used to find the appropriate alpha-threshold.

    Parameters
    ----------
    coal_function : String
        Name of the coalitional method to use.
    n_rate : int
        Number of subgroups needed to achieve complexity rate.
    X : pandas.DataFrame
        Dataframe of the input datas.
    max_iterations : int, default=100
        Maximum number of iteration for the bisection.

    Returns
    -------
    subgroups_best : two-dimensional list
        Groups of attributs compute with the best alpha and the coalition method.
    alpha_best : float
        Best alpha-threshold find by the bisection method
    """

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


def complexity_coal_groups(X, rate, grouping_function, reverse):
    """
    Compute attributs groups based on the method and the complexity rate in parameter.

    Parameters
    ----------
    X : pandas.DataFrame
        Dataframe of the input datas.
    rate : float
        Complexity percentage.
    grouping_function : string
        Name of the coalitional method to use.
    reverse : bool
        Type of method to use for Spearman and VIF coalition method.

    Returns
    -------
    coal_groups : two-dimensional list
        Groups of correlated attributs compute with the selected method.
    """
    n_total = 2 ** X.shape[1] - 1
    n_rate = int(np.round(n_total * rate, 0))
    coal_groups, alpha = find_alpha_rate(
        coal_function=lambda X_, threshold: remove_inclusions(
            grouping_function(X_, threshold, reverse)
        ),
        n_rate=n_rate,
        X=X,
    )

    return coal_groups


def compute_instance_coal_inf(raw_instance_inf, columns, relevant_groups):
    """
    Compute the influence of each attribut for one instance, based on the coalitional method.
    Attributs can overlap in groups.

    Parameters
    ----------
    raw_instance_inf : dict {tuple : float}
        Influence of each group of attributs for one instance.
    columns : list
        Names of attributs in the dataset.
    relevant_groups : list
        Groups of attributs defined by the coalition method.

    Returns
    -------
    influences : dict {string : float}
        Influence of each attribut for the instance. Key is the name of the attributs, value is the numeric influence.
    """

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


def compute_coalitional_influences(
    raw_influences, X, relevant_groups, progression_bar=True
):
    """Coalitional method for all instances, when attributs overlap in groups.

    Parameters
    ----------
    raw_influences : dict {int : dict {tuple : float}}
        Influence of each group of attributs for all instances.
    X : pandas.DataFrame
        The training input samples.
    relevant_groups : list
        Groups of attributs defined by the coalition method.
    progression_bar : boolean, default=True
        If True, progression bar are shown during computing explanations

    Returns
    -------
    coalitional_influences : pandas.DataFrame
        Influences for each attributs and each instances in the dataset.
    """

    coalitional_influences = pd.DataFrame(columns=X.columns)

    for instance in tqdm(
        X.index, desc="Coalitional influences", disable=not progression_bar
    ):
        raw_infs = raw_influences[instance]
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
    progression_bar=True,
):

    """
    Compute the influences based on the method in parameters.

    Parameters
    ----------
    X : pandas.DataFrame
        The training input samples.
    y : pandas.DataFrame
        The target values (class labels in classification, real numbers in regression).
    model : pandas.DataFrame
        Model to train and explain.
    rate : float
        Number to use for computing coalitional groups.
    problem_type : {"classification", "regression"}
        Type of machine learning problem.
    fvoid : float, default=None
        Prediction when all attributs are unknown. If None, the default value is used (expected value for each class for classification, mean label for regression).
    look_at : int, default=None
        Class to look at when computing influences in case of classification problem.
        If None, prediction is used.
    method : {"pca", "spearman", "vif"}, default="spearman"
        Name of the coalition method to compute attributs groups.
    reverse : boolean, default=False
        Type of method to use for Spearman and VIF coalition method.
    complexity : boolean, default=False
        Approach to calculating the threshold for coalition methods.
        If False, rate parameter is use as alpha-threshold.
        If True, rate is use as complexity rate to compute the alpha-threshold.
    scaler : boolean, default=False
        If True, a Standard Scaler is apply to data before compute PCA coalitional method.
    progression_bar : boolean, default=True
        If True, progression bar are shown during computing explanations

    Returns
    -------
    coalition_influences : two-dimensional list
        Influences for each attributs and each instances in the dataset.
    """
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
            groups = complexity_coal_groups(X, rate, methods[method], reverse)
        else:
            groups = methods[method](X, rate, reverse)

    subgroups = compute_subgroups_correlation(groups) + [[]]

    pretrained_models = train_models(
        model, X, y, subgroups, problem_type, fvoid, progression_bar
    )
    raw_groups_influences = explain_groups_w_retrain(
        pretrained_models, X, problem_type, look_at, progression_bar
    )

    coalition_influences = compute_coalitional_influences(
        raw_groups_influences, X, groups, progression_bar
    )

    return coalition_influences, pretrained_models, groups


def compute_influences(
    X,
    pretrained_models,
    problem_type,
    groups,
    look_at=None,
    progression_bar=True,
):
    """
    Compute the influences based on the coalitional method for the instances in parameter.


    Parameters
    ----------
    X : pandas.DatFrame
        The training input samples.
    pretrained_models : dictionary {tuple : pickle object}
        Models trained to compute explanations.
    problem_type :{"classification", "regression"}
        Type of machine learning problem.
    groups : list
        Groups of attributs used to compute explanations
    look_at : int, default=None
        Class to look at when computing influences in case of classification problem.
        If None, prediction is used.

    Returns
    -------
    complete_influences : two-dimensional list
        Influences for each attributs and each instances in the dataset.
    """

    raw_groups_influences = explain_groups_w_retrain(
        pretrained_models, X, problem_type, look_at, progression_bar
    )

    influences = compute_coalitional_influences(
        raw_groups_influences, X, groups, progression_bar
    )

    return influences
