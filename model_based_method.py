"""
model_based_method.py
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

import random
import pandas as pd

from utils import train_models, explain_groups_w_retrain, influence_calcul
from utils import check_all_attributs_groups, compute_subgroups_correlation
from utils import generate_subgroups_group, coal_penalisation


def random_permutation(groups, nb_att, preds):
    nb_inst = len(preds)
    nb_class = len(set(preds))
    perm = {}

    class_indices = [
        [j for j in range(nb_inst) if preds[j] == i] for i in range(nb_class)
    ]
    indices = list(range(nb_inst))

    for group in groups:
        permutation = [-1] * nb_inst

        # if group is not a singleton, permutation within class
        if len(group) > 1:
            for class_i in class_indices:
                temp_class = class_i.copy()

                for i in class_i:
                    rand_int = random.randrange(len(temp_class))
                    permutation[i] = temp_class[rand_int]
                    temp_class.pop(rand_int)

        # if the group is a singleton, fully random permutation
        else:
            temp_id = indices.copy()

            for i in indices:
                rand_int = random.randrange(len(temp_id))
                permutation[i] = temp_id[rand_int]
                temp_id.pop(rand_int)

        for i in group:
            perm[i] = permutation

    return perm


def fidelity(datas, model, groups):
    nb_att = datas.shape[1]
    nb_inst = datas.shape[0]
    nb_min = 2500

    # Data augmentation if nb_inst < nb_min
    new_dataset = datas.copy()

    if nb_inst < nb_min:
        for i in range(nb_min - nb_inst):
            new_inst = {}

            for j in range(nb_att):
                rand_int = random.randrange(nb_inst)
                new_inst[datas.columns[j]] = datas.iloc[rand_int, j]

            new_dataset = new_dataset.append(new_inst, ignore_index=True)

    old_preds = model.predict(new_dataset)
    permutation = random_permutation(groups, nb_att, old_preds)

    temp_dataset = pd.DataFrame(columns=new_dataset.columns)

    for i in range(nb_inst):
        temp_inst = {}

        for j in range(nb_att):
            temp_inst[new_dataset.columns[j]] = new_dataset.iloc[permutation[j][i], j]

        temp_dataset = temp_dataset.append(temp_inst, ignore_index=True)

    new_preds = model.predict(temp_dataset)

    count_diff = sum(1 for i in range(nb_inst) if new_preds[i] != old_preds[i])
    fid = (nb_inst - count_diff) / nb_inst

    return fid


def max_fidelity(S, A, R, datas, model):
    i_max = 0
    fid_max = 0

    for i in R:
        R_temp = R.copy()
        R_temp.remove(i)
        groups_temp = []
        temp = [i]

        groups_temp += [R_temp] if len(R_temp) else []
        groups_temp += [temp] if len(temp) else []
        groups_temp += [A] if len(A) else []
        groups_temp += [[i] for group in S for i in group] if len(S) else []

        fid = fidelity(datas, model, groups_temp)

        if fid_max < fid:
            i_max = i
            fid_max = fid

    return i_max, fid_max


def model_grouping(datas, model, threshold):
    nb_att = datas.shape[1]
    R = [i for i in range(nb_att)]
    L_m = [[i] for i in range(nb_att)]
    A = []
    S = []  # relevant_groups
    delta = fidelity(datas, model, L_m) + threshold

    while len(R) != 0 or len(A) != 0:
        S_temp = [[]]

        if len(S) > 0:
            S_temp = [[i] for group in S for i in group]

        if len(R) > 0:
            S_temp.append(R)

        if len(A) == 0 and fidelity(datas, model, S_temp) < delta:
            S_temp = [[i] for i in R]

            for att in S_temp:
                if att not in S:
                    S.append(att)

            R = []
            A = []
        else:
            indice_max, fid_max = max_fidelity(S, A, R, datas, model)

            if len(R) == 1 or fid_max < delta:
                S.append(R)
                R = A.copy()
                A = []
            else:
                R.remove(indice_max)
                A.append(indice_max)

    S = check_all_attributs_groups(S, nb_att)

    return S


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


def modelbased_method(X, y, model, threshold):
    groups = model_grouping(X, model, threshold) if X.shape[1] != 1 else [[0]]

    groups = compute_subgroups_correlation(groups) + [[]]

    pretrained_models = train_models(model, X, y, groups)
    raw_groups_influences = explain_groups_w_retrain(pretrained_models, X)
    coalition_influences = compute_coalitional_influences(
        raw_groups_influences, X, groups
    )

    return coalition_influences
