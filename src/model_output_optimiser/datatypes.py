from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Protocol

import numpy as np

type FeatureCalculator[T: int] = Callable[
    [np.ndarray[tuple[T, int], np.dtype[np.number]]],
    np.ndarray[tuple[T], np.dtype[np.number]],
]

type NumpyInt = int | np.integer
type NumpyFloat = float | np.floating
type NumpyNumber = NumpyInt | NumpyFloat
type SumConstraintSpecifier = tuple[Sequence[str], NumpyNumber, NumpyNumber]
type NonzeroConstraintSpecifier = tuple[Sequence[str], NumpyInt]


class EstimatorProtocol[T: int](Protocol):

    """
    Protocol for estimators with a predict method.

    Attributes
    ----------
    feature_names_in_ : np.ndarray[tuple[T], np.dtype[np.str_]]
        Array of input feature names, shape (T,).

    Methods
    -------
    predict(X, *args, **kwargs)
        Predict output values for input data.

        Parameters
        ----------
        X : np.ndarray[tuple[int, T], np.dtype[np.number]]
            Input data array of shape (n_samples, T).
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        y_pred : np.ndarray[tuple[T], np.dtype[np.number]]
            Predicted output values, shape (T,).

    Notes
    -----
    This protocol enforces the presence of a predict method and feature_names_in_ 
    attribute.
    """

    feature_names_in_: np.ndarray[tuple[T], np.dtype[np.str_]]

    def predict(
        self,
        X: np.ndarray[tuple[int, T], np.dtype[np.number]],
        *args,
        **kwargs,
    ) -> np.ndarray[tuple[T], np.dtype[np.number]]:

        ...


class DifferentialEvolutionStrategy(StrEnum):
   BEST1BIN = "best1bin"
   BEST1EXP = "best1exp"
   RAND1BIN = "rand1bin"
   RAND1EXP = "rand1exp"
   RAND2BIN = "rand2bin"
   RAND2EXP = "rand2exp"
   RANDTOBEST1BIN = "randtobest1bin"
   RANDTOBEST1EXP = "randtobest1exp"
   CURRENTTOBEST1BIN = "currenttobest1bin"
   CURRENTTOBEST1EXP = "currenttobest1exp"
   BEST2EXP = "best2exp"
   BEST2BIN = "best2bin"


class DifferentialEvolutionInitialConfiguration(StrEnum):
   LATINHYPERCUBE = "latinhypercube"
   SOBOL = "sobol"
   HALTON = "halton"
   RANDOM = "random"