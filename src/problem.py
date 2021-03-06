import logging
import os
import pickle
from typing import Dict

import arff
import functools
import operator
import json
import pandas as pd

from surrogates import train_save_surrogates, load_surrogates



class Problem:
    """ Object for organizing all data of a problem. """

    def __init__(self, name: str):
        self.name = name
        self._data = None
        self._surrogates = None
        self._metadata = None

        with open(f'problems/{name}.json', 'r') as fh:
            self._json = json.load(fh)

    @property
    def hyperparameters(self):
        return list(self._json['hyperparameters'].keys())

    @property
    def hyperparameter_types(self):
        return self._json['hyperparameters']

    @property
    def benchmarks(self) -> Dict[str, str]:
        return self._json.get('benchmark', {})

    @property
    def data(self) -> pd.DataFrame:
        """ Experiment data with task, hyperparameter configuration and score. """
        if self._data is None:
            logging.info("Loading experiment data.")
            with open(self._json['experiment_data'], 'r') as fh:
                data = arff.load(fh)
            unfiltered_data = pd.DataFrame(
                data['data'], columns=[name for name, type_ in data['attributes']]
            )

            # The problem may specify filters on hyperparameter values
            if len(self._json.get('filters', [])) > 0:
                filters = [unfiltered_data[hp] == default
                           for (hp, default) in self._json['filters'].items()]
                combined_filter = functools.reduce(operator.iand, filters)
                experiments = unfiltered_data[combined_filter]
            else:
                experiments = unfiltered_data

            if len(self._json.get('ignore', [])) > 0:
                experiments = experiments.drop(self._json['ignore'], axis=1)

            experiments = experiments.rename(columns={self._json['metric']: 'target'})

            # Drop tasks with constant performance
            counts = experiments.groupby("task_id")['target'].nunique()
            if len(counts[counts==1]):
                experiments.drop(experiments[experiments.task_id == counts[counts==1].index.values[0]].index, axis=0, inplace=True)

            tasks_to_ignore = self._json.get("exclude", [])
            if len(tasks_to_ignore) > 1:
                experiments.drop(experiments[experiments.task_id.isin(tasks_to_ignore)].index, axis=0, inplace=True)

            if self._json.get("how") == "minimize":
                experiments.target = -experiments.target
            self._data = experiments

        return self._data

    @property
    def metadata(self) -> pd.DataFrame:
        """ The meta-data of datasets for which a surrogate model has been created. """
        if self._metadata is None:
            metadataset = pd.read_csv(self._json['metadata'], index_col=0)
            self._metadata = metadataset[metadataset.index.isin(self.surrogates)]
        return self._metadata

    @property
    def surrogates(self) -> Dict[int, object]:
        """ Surrogate models for each task with experiment data. """
        if self._surrogates is None:
            surrogate_file = self._json["surrogates"]
            if os.path.exists(surrogate_file):
                self._surrogates = load_surrogates(surrogate_file)
            else:
                self._surrogates = train_save_surrogates(
                    self.data, self.hyperparameters, surrogate_file
                )
        return self._surrogates

    @property
    def fixed(self) -> Dict[str, float]:
        return self._json.get('fixed', {})

    @property
    def valid_tasks(self):
        return self.data.task_id.unique()


