"""
Coalitional explanation method
Copyright (C) 2020 Gabriel Ferrettini <gabriel.ferrettini@irit.fr>

utils.py
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

import numpy as np
import pickle
from sklearn.base import clone
from tqdm import tqdm


def generate_groups_wo_label(nb_attributs):
    """
    Generates all possible combinations of attributes.

    Parameters
    ----------
    nb_attributs : int
        Number of attributs in the dataset

    Returns
    -------
    two-dimensional list
        List of groups of index, representing all the combinations for the
        number of attributs
    """

    return [
        [j for j in range(nb_attributs) if ((i & (1 << j)) > 0)]
        for i in range(2 ** nb_attributs)
    ]


def generate_subgroups_group(group):
    """
    Generates all possible subgroups from a group of attributes.

    Parameters
    ----------
    group : list
        list of attributs.

    Returns
    -------
    two-dimensional list
        list of all the subgroups from the attributs list.
    """

    sub_ids = generate_groups_wo_label(len(group))
    sub_ids = [i for i in sub_ids if len(i) > 0]

    return [[group[i] for i in index] for index in sub_ids]


def compute_subgroups_correlation(groups):
    subgroups_list = []

    for group in groups:
        subgroups_list.extend(
            [
                subgroup
                for subgroup in generate_subgroups_group(group)
                if subgroup not in subgroups_list
            ]
        )

    return subgroups_list


def sorted_groups(groups):
    return [sorted(group) for group in groups]


def check_all_attributs_groups(groups, nb_attributs):
    """
    Check if all attributs are in at least one group.

    Parameters
    ----------
    groups : two-dimension list
        List of groups defined by a coalitional method.
    nb_attributs : int
        Number of attribut in the dataset.

    Returns
    -------
    groups : two-dimension list
        Checked list of groups, with all attributs.

    """

    for i in range(nb_attributs):
        flag = False

        for group in groups:
            flag = flag | (i in group)

        if not flag:
            groups.append([i])

    return sorted_groups(groups)


def remove_inclusions(groups):
    return [
        groups[i]
        for i in range(len(groups))
        if not any([set(groups[i]).issubset(x) for x in groups[:i] + groups[(i + 1) :]])
    ]



def train_models(model, X, y, groups, problem_type, fvoid):
    """
    Trains the model with all the attributs, compute the
    and an array of model, each one wo one group of attribut

    Parameters
    ----------
    model : Scikit-learn model
        Model to train.
    X : Pandas.Dataframe
        Dataframe of the input datas.
    y : Pandas.Dataframe
        dataframe of the expected prediction from the model.
    groups : two-dimensional list
        List of all possible attributs subgroups.

    Returns
    -------
    pretrained_models : dictionary {tuple : pickle object}
        Dictionary of the Pre-trained models (serialized with pickle). The key
        is the tuple of attributs the model has been train on. The item is the pre-trained model
    """

    n_variables = X.shape[1]
    pretrained_models = {}
    complete_group = [i for i in range(X.shape[1])]
    pretrained_models[tuple(complete_group)] = pickle.dumps(model)

    for group in tqdm(groups, desc="Train"):
        model_clone = clone(model)

        if len(group) == 0:
            if fvoid is None :
                if problem_type == "Classification" :
                    fvoid = y.value_counts(normalize=True).sort_index().values                  # A MODIFIER!
                elif problem_type == "Regression" :
                    fvoid = y.mean()
            pretrained_models[tuple(group)] = fvoid
        elif len(group) < n_variables:
            model_clone.fit(X[X.columns[group]].values, y.values.flatten())
            pickle_model = pickle.dumps(model_clone)
            group.sort()
            pretrained_models[tuple(group)] = pickle_model

    return pretrained_models


def explain_groups_w_retrain(pretrained_models, X, problem_type, look_at):
    """
    Compute the influence of each atributs or group of attributs for the instance
    in parameter.

    Parameters
    ----------
    pretrained_models : dictionary {tuple : pickle object}
        Dictionary of all the Pre-trained models (serialized with pickle).
    X : Pandas.Dataframe
        Dataframe of the input datas.

    Returns
    -------
    explanations_groups : dictionary {tuple : float}
        Influence of each attributs/group of attributs. The key is a tuple with
        the indexes of the attributs, the value the influence of the group.
    """

    explanations_groups = {}

    for i in X.index:
        explanations_groups[i] = {}

    all_attributes = tuple([i for i in range(X.shape[1])])
    preds = pickle.loads(pretrained_models.get(all_attributes)).predict(X)
    fvoid = pretrained_models.get(())

    for group in tqdm(pretrained_models.keys(), desc="Raw impact"):
        if len(group) == 0:
            for i in X.index:
                explanations_groups[i][group] = 0.0
        else:
            model = pickle.loads(pretrained_models.get(group))
            X_groups = X[X.columns[list(group)]]
            
            if problem_type == "Classification" :
                preds_proba = model.predict_proba(X_groups)

                for i in X.index:
                    look_at_i = look_at
                    if look_at == None :
                        look_at_i = preds[i]
                    explanations_groups[i][group] = (
                    preds_proba[i][look_at_i] - fvoid[look_at_i]
                    )
                    
            elif problem_type == "Regression" :
                preds_ = model.predict(X_groups)

                for i in X.index:
                    explanations_groups[i][group] = (
                    preds_[i] - fvoid
                    )

    return explanations_groups


"""
Compute the penalisation values for complete, coalitional 
and k-depth methods

Parameters
----------
n : int
    length of the subgroup.
s : int
    length of the group.
k : int
    max cardinal of the subgroups for k-depth.

Returns
-------
Float
    the penalisation for the subgroup.

"""


def standard_penalisation(s, n):
    return (np.math.factorial(s) * np.math.factorial(n - s - 1)) / (
        np.math.factorial(n)
    )


def kdepth_penalisation(s, n, k):
    return (np.math.factorial(s) * np.math.factorial(n - s - 1)) / (
        k * np.math.factorial(n - 1)
    )


def coal_penalisation(s, n):
    return np.math.factorial(s) * np.math.factorial(n - s - 1)


def influence_calcul(pena, raw_infs, group, i):
    """
    Compute the influence of a attribut for one group, regardless the method.

    Parameters
    ----------
    pena : float
        Penalisation of the group, regarding the method.
    raw_infs : dictionary
        Influence of each group of attributs for one instance.
    group : list
        Attributs in the group to study.
    i : int
        Attribut to study.

    Returns
    -------
    float
        Influence of the attribut i in the group.

    """

    group_clone = list(group).copy()
    group_clone.remove(i)

    return (
        pena * (raw_infs.get(tuple(group)))
        if len(group_clone) == 0
        else pena * (raw_infs.get(tuple(group)) - raw_infs.get(tuple(group_clone)))
    )
