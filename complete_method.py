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

from utils import standard_penalisation, generate_groups_wo_label
from utils import train_models, explain_groups_w_retrain, influence_calcul


def compute_instance_complete_inf(raw_instance_inf, columns):
    """Complete method, per instance
    Shapley value approximation (Strumbelj et al. 2010)"""

    influences = dict([(c, 0) for c in columns])

    for i in range(len(columns)):
        for group in raw_instance_inf.keys():
            if i in group:
                pena = standard_penalisation(len(group) - 1, len(columns))
                influences[columns[i]] += influence_calcul(
                    pena, raw_instance_inf, group, i
                )

    return influences


def compute_complete_influences(raw_groups_influences, X):
    """Complete method, for all instances
    Shapley value approximation (Strumbelj et al. 2010)"""

    complete_influences = pd.DataFrame(columns=X.columns)

    for instance in X.index:
        raw_infs = raw_groups_influences[instance]
        influences = compute_instance_complete_inf(raw_infs, X.columns)
        complete_influences = complete_influences.append(
            pd.Series(influences, name=instance)
        )

    return complete_influences


def complete_method(X, y, model, problem_type, fvoid=None, look_at=None):
    groups = generate_groups_wo_label(X.shape[1])

    pretrained_models = train_models(model, X, y, groups, problem_type, fvoid)
    raw_groups_influences = explain_groups_w_retrain(
        pretrained_models, X, problem_type, look_at
    )

    complete_influences = compute_complete_influences(raw_groups_influences, X)

    return complete_influences
