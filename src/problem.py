from typing import Dict

import arff
import functools
import operator
import json
import pandas as pd

from surrogates import load_or_train_surrogates


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
        return self._json['hyperparameters']

    @property
    def performance_column(self):
        return self._json['performance_column']

    @property
    def benchmarks(self) -> Dict[str, str]:
        return self._json.get('benchmark', {})

    @property
    def data(self) -> pd.DataFrame:
        """ Experiment data with task, hyperparameter configuration and score. """
        if self._data is None:
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
            self._surrogates = load_or_train_surrogates(self._json)
        return self._surrogates
