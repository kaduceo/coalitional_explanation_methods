"""
complete_method.py
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

import pandas as pd
from tqdm import tqdm

from utils import standard_penalisation, generate_groups_wo_label
from utils import train_models, explain_groups_w_retrain, influence_calcul


def compute_instance_complete_inf(raw_instance_inf, columns):
    """
    Computes influence of each attributs for one instance with the complete method.
    Shapley value approximation (Strumbelj et al. 2010)

    Parameters
    ----------
    raw_instance_inf : dict {tuple : float}
        Influence of each group of attributs of a instance.
    columns : list
        Names of attributs in the dataset.

    Returns
    -------
    influences : dict {string : float}
        Influence of each attributs for the instance. Key is the attribut name, value in the numeric influence.

    """

    influences = dict([(c, 0) for c in columns])

    for i in range(len(columns)):
        for group in raw_instance_inf.keys():
            if i in group:
                pena = standard_penalisation(len(group) - 1, len(columns))
                influences[columns[i]] += influence_calcul(
                    pena, raw_instance_inf, group, i
                )

    return influences


def compute_complete_influences(raw_influences, X, progression_bar):
    """
    Complete method, for all instances
    Shapley value approximation (Strumbelj et al. 2010)

    Parameters
    ----------
    raw_influences : dict {int : dict {tuple : float}}
        Influence of each group of attributs for all instances.
    X : pandas.DatFrame
        The training input samples.

    Returns
    -------
    influences : dict {string : float}
        Influences for each attributs and each instances in the dataset.
    """

    complete_influences = pd.DataFrame(columns=X.columns)

    for instance in tqdm(
        X.index, desc="Complete influences", disable=not progression_bar
    ):
        raw_infs = raw_influences[instance]
        influences = compute_instance_complete_inf(raw_infs, X.columns)
        complete_influences = complete_influences.append(
            pd.Series(influences, name=instance)
        )

    return complete_influences


def complete_method(
    X, y, model, problem_type, fvoid=None, look_at=None, progression_bar=True
):
    """
    Initialise the explanation method and compute the influences based on the complete method for the instances used to train the model.


    Parameters
    ----------
    X : pandas.DatFrame
        The training input samples.
    y : pandas.DataFrame
        The target values (class labels in classification, real numbers in regression).
    model
        Trained model to explain
    problem_type :{"classification", "regression"}
        Type of machine learning problem.
    fvoid : float, default=None
        Prediction when all attributs are unknown. If None, the default value is used (expected value for each class for classification, mean label for regression).
    look_at : int, default=None
        Class to look at when computing influences in case of classification problem.
        If None, prediction is used.

    Returns
    -------
    complete_influences : two-dimensional list
        Influences for each attributs and each instances in the dataset.
    pretrained_models : dictionary {tuple : pickle object}
        Models trained to compute explanations.
    """

    groups = generate_groups_wo_label(X.shape[1])

    pretrained_models = train_models(
        model, X, y, groups, problem_type, fvoid, progression_bar
    )

    raw_influences = explain_groups_w_retrain(
        pretrained_models, X, problem_type, look_at, progression_bar
    )

    complete_influences = compute_complete_influences(
        raw_influences, X, progression_bar
    )

    return complete_influences, pretrained_models


def compute_influences(
    X,
    pretrained_models,
    problem_type,
    look_at=None,
    progression_bar=True,
):
    """
    Compute the influences based on the complete method for the instances in parameter.


    Parameters
    ----------
    X : pandas.DatFrame
        The training input samples.
    pretrained_models : dictionary {tuple : pickle object}
        Models trained to compute explanations.
    problem_type :{"classification", "regression"}
        Type of machine learning problem.
    look_at : int, default=None
        Class to look at when computing influences in case of classification problem.
        If None, prediction is used.

    Returns
    -------
    complete_influences : two-dimensional list
        Influences for each attributs and each instances in the dataset.
    """

    raw_influences = explain_groups_w_retrain(
        pretrained_models, X, problem_type, look_at, progression_bar
    )

    influences = compute_complete_influences(raw_influences, X, progression_bar)

    return influences
