"""
K-depth explanation method (https://hal.archives-ouvertes.fr/hal-02942310)
Copyright (C) 2020 Gabriel Ferrettini <gabriel.ferrettini@irit.fr>
Copyright (C) 2020 Julien Aligon <julien.aligoni@irit.fr>
Copyright (C) 2020 Chantal Soulé-Dupuy <chantal.soule-dupuy@irit.fr>

kdepth_method.py
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

from utils import kdepth_penalisation, generate_groups_wo_label
from utils import train_models, explain_groups_w_retrain, influence_calcul


def compute_instance_kdepth_inf(raw_instance_inf, columns, relevant_groups, k):
    """ K-depth method, per instance (Ferrettini et al. 2020)"""

    influences = dict([(c, 0) for c in columns])

    for i in range(len(columns)):
        for group in relevant_groups:
            if i in group:
                pena = kdepth_penalisation(len(group) - 1, len(columns), k)
                influences[columns[i]] += influence_calcul(
                    pena, raw_instance_inf, group, i
                )

    return influences


def compute_kdepth_influences(raw_groups_influences, X, relevant_groups, k):
    """ K-depth method, for all instances (Ferrettini et al. 2020)"""

    kdepth_influences = pd.DataFrame(columns=X.columns)

    for instance in X.index:
        raw_infs = raw_groups_influences[instance]
        influences = compute_instance_kdepth_inf(
            raw_infs, X.columns, relevant_groups, k
        )
        kdepth_influences = kdepth_influences.append(
            pd.Series(influences, name=instance)
        )

    return kdepth_influences


def kdepth_method(X, y, model, k, problem_type, fvoid=None, look_at=None):
    groups = generate_groups_wo_label(X.shape[1])
    groups_kdepth = [groups[i] for i in range(len(groups)) if len(groups[i]) <= k]

    pretrained_models = train_models(model, X, y, groups_kdepth, problem_type, fvoid)
    raw_groups_influences = explain_groups_w_retrain(
        pretrained_models, X, problem_type, look_at
    )
    kdepth_influences = compute_kdepth_influences(
        raw_groups_influences, X, groups_kdepth, k
    )

    return kdepth_influences


def compute_linear_influences(raw_groups_influences, X):
    """ Linear method for all instances """

    linear_influences = pd.DataFrame(columns=X.columns)

    for instance in X.index:
        influences = {}

        for i in range(X.shape[1]):
            influences[X.columns[i]] = raw_groups_influences[instance].get(tuple([i]))

        linear_influences = linear_influences.append(
            pd.Series(influences, name=instance)
        )

    return linear_influences


def linear_method(X, y, model, problem_type, fvoid=None, look_at=None):
    groups = [[i] for i in range(X.shape[1])]
    groups += [[]]

    pretrained_models = train_models(model, X, y, groups, problem_type, fvoid)
    raw_groups_influences = explain_groups_w_retrain(
        pretrained_models, X, problem_type, look_at
    )

    linear_influences = compute_linear_influences(raw_groups_influences, X)

    return linear_influences
