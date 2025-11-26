from collections.abc import Iterable, KeysView, Mapping
import operator as op
from typing import Literal, overload

import numpy as np
from scipy import optimize
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from .datatypes import (
    DifferentialEvolutionInitialConfiguration,
    DifferentialEvolutionStrategy,
    EstimatorProtocol, 
    FeatureCalculator, 
    NonzeroConstraintSpecifier, 
    NumpyNumber, 
    SumConstraintSpecifier,
)
from .exceptions import (
    DuplicateFeatureError, 
    FeatureMismatchError, 
    NoFeatureNamesError, 
    ShapeError,
)
from .utilities import (
    generate_bounds_combinations, 
    get_data_bounds, 
    get_element_positions, 
    get_pairwise_distance_metric,
    scale_data_to_bounds, 
    validate_bounds,
)


class ConstrainedEstimatorOptimiser[T: int, U: int]:

    """
    Optimises variables of a constrained estimator model subject to feature, sum, and 
    distance constraints.

    This class provides a flexible interface for setting up and solving constrained 
    optimisation problems involving machine learning estimators, with support for custom 
    feature calculators, variable bounds, sum constraints, nonzero constraints, and 
    distance constraints. It is designed to work with models following the 
    EstimatorProtocol interface.

    Parameters
    ----------
    None

    Attributes
    ----------
    feature_names : np.ndarray
        Array of feature names as recognised by the model.
    n_features : int
        Number of features in the model.
    variable_names_array : np.ndarray
        Array of variable names subject to optimisation.
    variable_bounds_array : np.ndarray
        Array of bounds for each variable.
    _constraints : list
        List of constraints to be passed to the optimiser.
    _distance_metric : str
        The distance metric used for pairwise distance calculations.
    _training_data_scaled : np.ndarray
        Scaled version of the training data.
    _bounds_combinations : np.ndarray
        All generated bounds combinations for the optimisation.

    Notes
    -----
    This class is intended for advanced users who need fine-grained control over 
    constrained optimisation in the context of machine learning models. It is compatible 
    with strong NumPy typing and supports integration with SciPy's differential evolution
    optimiser.

    Examples
    --------
    >>> optimiser = ConstrainedEstimatorOptimiser()
    >>> optimiser.set_up_optimisation(
    ...     model=my_model,
    ...     training_data=X_train,
    ...     feature_calculators=feature_calcs,
    ...     user_defined_features=user_feats,
    ...     variable_bounds=var_bounds,
    ...     variable_prices=var_prices,
    ...     minimise_output=True,
    ... )
    >>> result = optimiser.perform_optimisation()
    """

    def __init__(
        self,
    ) -> None:

        """
        Initialises the ConstrainedEstimatorOptimiser.

        Notes
        -----
        All configuration is performed via the `set_up_optimisation` method.
        """

        pass


    def _verify_and_register_feature_names(
        self,
        *,
        model: EstimatorProtocol,
        training_data: np.ndarray[tuple[int, U], np.dtype[np.number]],
        variables_keys: KeysView[str],
        user_defined_features_keys: KeysView[str],
        feature_calculators_keys: KeysView[str],
    ) -> None:

        """
        Validates and registers feature names from the model and user input.

        Parameters
        ----------
        model : EstimatorProtocol
            The estimator model, which must have a `feature_names_in_` attribute.
        training_data : np.ndarray
            Training data array of shape (n_samples, n_features).
        variables_keys : KeysView[str]
            Keys of variables to be optimised.
        user_defined_features_keys : KeysView[str]
            Keys of user-defined features.
        feature_calculators_keys : KeysView[str]
            Keys of features calculated via feature calculators.

        Raises
        ------
        NoFeatureNamesError
            If the model does not have feature names.
        FeatureMismatchError
            If the feature names in the model do not match the union of user-defined 
            features, variables, and feature calculators.

        Notes
        -----
        This method sets up internal mappings and masks for feature handling throughout 
        the optimisation.
        """

        if not hasattr(model, 'feature_names_in_'):
            raise NoFeatureNamesError('Model must have attribute feature_names_in_.')
        elif (feature_names := model.feature_names_in_) is None:
            raise NoFeatureNamesError(
                'Model must be trained with feature names.'
            )
        else:
            self.feature_names = feature_names

        self.n_features = len(self.feature_names)
        if self.n_features != training_data.shape[1]:
            raise FeatureMismatchError(
                f'number of features in model.feature names {self.n_features} does ' \
                f'not match {training_data.shape[1] = }.'
            )

        true_feature_set = {*self.feature_names}
        self._variables_set = variables_keys
        feature_set = (
            user_defined_features_keys
            |
            self._variables_set
            |
            feature_calculators_keys
        )

        if (feature_mismatch := true_feature_set ^ feature_set):
            raise FeatureMismatchError(
                'features in trained model do not match union of ' \
                'user_defined_features, variable_bounds and feature_calculators: ' \
                f'{", ".join(feature_mismatch)}).'
            )

        self.feature_names_array = np.array(self.feature_names, dtype = np.str_)
        self._feature_names_sorter = self.feature_names_array.argsort()
        self._user_defined_features_mask = np.isin(
            self.feature_names_array, 
            [*user_defined_features_keys],
        )
        self._variables_mask = np.isin(
            self.feature_names_array, 
            [*variables_keys],
        )
        self.variable_names_array = self.feature_names_array[self._variables_mask]
        self._n_variables = self.variable_names_array.shape[0]
        self._calculated_features_indices = get_element_positions(
            sequence = self.feature_names_array,
            elements = [*feature_calculators_keys],
            sorter = self._feature_names_sorter,
        )


    def _register_problem_bounds(
        self,
        variable_bounds: Mapping[str, tuple[NumpyNumber, NumpyNumber]],
        training_data_bounds: np.ndarray[tuple[U, Literal[2]], np.dtype[np.float32]],
    ) -> np.ndarray[tuple[U, Literal[2]], np.dtype[np.float32]]:

        """
        Registers and combines variable bounds with training data bounds.

        Parameters
        ----------
        variable_bounds : mapping
            Mapping from variable names to (lower, upper) bounds.
        training_data_bounds : np.ndarray
            Array of shape (n_features, 2) with bounds from the training data.

        Returns
        -------
        bounds_array : np.ndarray
            Combined bounds array of shape (n_features, 2).

        Notes
        -----
        Variable bounds override the corresponding training data bounds for variables.
        """

        insertion_positions = get_element_positions(
            sequence=self.feature_names_array, 
            elements=[*self._variables_set],
            sorter = self._feature_names_sorter,
        )
        self._bounds_array = training_data_bounds.copy()
        self._bounds_array[insertion_positions] = np.array(
            [*variable_bounds.values()], 
            dtype = np.float32,
        )
        self.variable_bounds_array = self._bounds_array[self._variables_mask]
        return self._bounds_array


    def _create_price_array(
        self,
        prices: Mapping[str, NumpyNumber]
    ) -> np.ndarray[tuple[int], np.dtype[np.float32]]:

        """
        Creates an array of prices for the variables in the correct order.

        Parameters
        ----------
        prices : mapping
            Mapping from variable names to their prices.

        Returns
        -------
        price_array : np.ndarray
            Array of prices aligned with the variable order.

        Notes
        -----
        The order of prices matches the order of variables in `variable_names_array`.
        """

        self._price_array = np.empty(
            shape = self._variables_mask.sum(),
            dtype = np.float32,
        )
        insertion_order = np.searchsorted(
            self.variable_names_array,
            [*prices.keys()],
            side = 'left',
            sorter = self.variable_names_array.argsort(),
        )
        self._price_array[insertion_order] = [*prices.values()]
        return self._price_array


    def _create_base_row(
        self,
        user_defined_features: Mapping[str, NumpyNumber],
    ) -> np.ndarray[tuple[U], np.dtype[np.float32]]:

        """
        Creates a base row for prediction, initialised with user-defined feature values.

        Parameters
        ----------
        user_defined_features : mapping
            Mapping from feature names to their fixed values.

        Returns
        -------
        base_row : np.ndarray
            Array of feature values, with user-defined features set.

        Notes
        -----
        This base row is used as a template for constructing prediction arrays during 
        optimisation.
        """

        base_row = np.empty(self.feature_names_array.shape, dtype = np.float32)
        insertion_positions = get_element_positions(
            sequence=self.feature_names_array, 
            elements=[*user_defined_features.keys()],
            sorter = self._feature_names_sorter,
        )
        base_row[insertion_positions] = [*user_defined_features.values()]
        self.base_row = base_row
        return base_row


    @overload
    def _reshape_optimiser_variables_input[V: int, W: np.dtype[np.number]](
        self,
        variable_values: np.ndarray[tuple[T, V], W],
    ) -> np.ndarray[tuple[V, T], W]:

        ...


    @overload
    def _reshape_optimiser_variables_input[W: np.dtype[np.number]](
        self,
        variable_values: np.ndarray[tuple[T], W],
    ) -> np.ndarray[tuple[Literal[1], T], W]:

        ...


    def _reshape_optimiser_variables_input(
        self,
        variable_values,
    ):

        """
        Reshapes the input array of variable values for the optimiser.

        Parameters
        ----------
        variable_values : np.ndarray
            Array of variable values, either 1D or 2D.

        Returns
        -------
        reshaped : np.ndarray
            2D array of shape (n_samples, n_variables).

        Raises
        ------
        ShapeError
            If the input array has more than 2 dimensions.

        Notes
        -----
        Ensures that variable values are always in a consistent shape for downstream 
        processing.
        """

        match variable_values.ndim:
            case 1:
                return variable_values[None]
            case 2:
                return variable_values.T
            case _:
                raise ShapeError(
                    f'unexpected number of input dimensions {variable_values.ndim}.'
                )


    @overload
    def _prepare_prediction_array[V: int](
        self,
        variable_values: np.ndarray[tuple[T, V], np.dtype[np.number]],
    ) -> np.ndarray[tuple[V, U], np.dtype[np.float32]]:

        ...


    @overload
    def _prepare_prediction_array(
        self,
        variable_values: np.ndarray[tuple[T], np.dtype[np.number]],
    ) -> np.ndarray[tuple[Literal[1], U], np.dtype[np.floating]]:

        ...


    def _prepare_prediction_array(
        self,
        variable_values,
    ):

        """
        Prepares a full feature array for prediction by self.model, inserting variable 
        values and calculating derived features.

        Parameters
        ----------
        variable_values : np.ndarray
            Array of variable values, either 1D or 2D.

        Returns
        -------
        prediction_array : np.ndarray
            Array of shape (n_samples, n_features) ready for model prediction.

        Notes
        -----
        This method fills in variable values and computes any calculated features using 
        the provided feature calculators.
        """

        variable_values = self._reshape_optimiser_variables_input(variable_values)
        n_rows_out = variable_values.shape[0]
        base_array = self.base_row[None].repeat(n_rows_out, axis = 0)
        base_array[:,self._variables_mask] = variable_values
        for feature_calculator, feature_index in zip(
            self.feature_calculators.values(), 
            self._calculated_features_indices,
        ):
            base_array[:,feature_index] = feature_calculator(base_array)
        return base_array


    @overload
    def loss_function[V: int](
        self,
        variable_values: np.ndarray[tuple[T, V], np.dtype[np.number]],
    ) -> np.ndarray[tuple[V], np.dtype[np.number]]:

        ...


    @overload
    def loss_function(
        self,
        variable_values: np.ndarray[tuple[T], np.dtype[np.number]],
    ) -> np.ndarray[tuple[Literal[1]], np.dtype[np.number]]:

        ...


    def loss_function(
        self,
        variable_values,
    ):

        """
        Computes the loss for the given variable values.

        Parameters
        ----------
        variable_values : np.ndarray
            Array of variable values, either 1D or 2D.

        Returns
        -------
        loss : np.ndarray
            Array of loss values for each set of variable values.

        Notes
        -----
        If `minimise_output` is True, the model's prediction is returned; otherwise, the 
        negative prediction is returned. This allows for both minimisation and 
        maximisation objectives.
        """

        self.history.append(variable_values)
        array_to_predict = self._prepare_prediction_array(
            variable_values = variable_values,
        )
        if self.minimise_output:
            return self.model.predict(array_to_predict)
        else:
            return -self.model.predict(array_to_predict)


    @overload
    def distance_function[V: int](
        self,
        variable_values: np.ndarray[tuple[T, V], np.dtype[np.number]],
    ) -> np.ndarray[tuple[V], np.dtype[np.number]]:

        ...


    @overload
    def distance_function(
        self,
        variable_values: np.ndarray[tuple[T], np.dtype[np.number]],
    ) -> np.ndarray[tuple[Literal[1]], np.dtype[np.number]]:

        ...


    def distance_function(
        self,
        variable_values,
    ):

        """
        Computes the minimum normalised distance from the training data for the given 
        variable values.

        Parameters
        ----------
        variable_values : np.ndarray
            Array of variable values, either 1D or 2D.

        Returns
        -------
        distances : np.ndarray
            Array of minimum normalised distances for each set of variable values.

        Notes
        -----
        The distance is normalised by the number of features and uses the selected 
        distance metric.
        """

        array_to_predict = self._prepare_prediction_array(
            variable_values = variable_values,
        )
        array_to_predict_scaled = scale_data_to_bounds(
            data = array_to_predict, 
            bounds = self._bounds_array,
            copy = True,
            dtype = np.float32,
        )
        distances = pairwise_distances(
            X = self._training_data_scaled,
            Y = array_to_predict_scaled,
            metric = self._distance_metric,
        )
        return (distances.min(axis = 0) / self.n_features)[None]


    def _generate_linear_constraint(
        self,
        sum_constraints: Iterable[SumConstraintSpecifier] | None,
        price_constraint: NumpyNumber | None,
    ) -> optimize.LinearConstraint:

        """
        Generates a linear constraint object for the optimiser based on sum and price 
        constraints.

        Parameters
        ----------
        sum_constraints : iterable of tuple or None
            Iterable of (variable_names, lower_bound, upper_bound) specifying sum 
            constraints.
        price_constraint : NumpyNumber or None
            Upper bound on the total price of variables.

        Returns
        -------
        linear_constraint : scipy.optimize.LinearConstraint or None
            The constructed linear constraint, or None if no constraints are specified.

        Raises
        ------
        FeatureMismatchError
            If unknown variables are referenced in the constraints.
        DuplicateFeatureError
            If duplicate variables are found in a constraint.

        Notes
        -----
        This method validates all constraints and ensures they are compatible with the 
        variable set.
        """

        linear_constraints = []
        lower_bounds = []
        upper_bounds = []
        for variable_subset, lb, ub in (sum_constraints or []):
            variable_subset_set = {*variable_subset}
            if (unknown_features := variable_subset_set - self._variables_set):
                raise FeatureMismatchError(
                    f'Unknown variables {", ".join(unknown_features)} in sum_constraints.'
                )
            elif len(variable_subset_set) != len(variable_subset):
                raise DuplicateFeatureError(
                    f'Duplicated variables in sum_constraints {", ".join(variable_subset)}.'
                )
            feature_mask = np.isin(
                self.variable_names_array, 
                variable_subset,
                assume_unique = True,
            )
            validate_bounds(
                lower_bound = lb,
                upper_bound = ub,
                lower_bound_min = self.variable_bounds_array[feature_mask, 0].sum(),
                lower_bound_max = ub,
                upper_bound_min = lb,
                upper_bound_max = self.variable_bounds_array[feature_mask, 1].sum(),
                name = f'sum constraint {variable_subset}',
            )
            linear_constraints.append(feature_mask)
            lower_bounds.append(lb)
            upper_bounds.append(ub)

        if price_constraint:
            linear_constraints.append(self._price_array)
            lower_bounds.append(0)
            upper_bounds.append(price_constraint)

        if linear_constraints:
            self._linear_constraint = optimize.LinearConstraint(
                np.stack(linear_constraints),
                lb = np.array(lower_bounds, dtype = np.float32),
                ub = np.array(upper_bounds, dtype = np.float32),
            )
        else:
            self._linear_constraint = None
        
        return self._linear_constraint


    def _generate_bounds_combinations(
        self,
        nonzero_constraints: Iterable[NonzeroConstraintSpecifier] | None = None,
    ) -> np.ndarray[tuple[int, U, Literal[2]], np.dtype[np.float32]]:

        """
        Generates all possible bounds combinations based on nonzero constraints.

        Parameters
        ----------
        nonzero_constraints : iterable of tuple or None
            Iterable of (feature_names, min_nonzero) specifying nonzero constraints.

        Returns
        -------
        bounds_combinations : np.ndarray
            Array of shape (n_combinations, n_variables, 2) containing all bounds 
            combinations.

        Notes
        -----
        If no nonzero constraints are provided, a single combination is returned.
        """

        if nonzero_constraints:
            self._bounds_combinations = generate_bounds_combinations(
                bounds = self.variable_bounds_array,
                nonzero_constraints = nonzero_constraints,
                feature_names = self.variable_names_array,
                dtype = np.float32,
            )
        else:
            self._bounds_combinations = self.variable_bounds_array[None]
        return self._bounds_combinations


    def _generate_distance_constraint(
        self,
        max_distance_constraint: NumpyNumber | None,
    ) -> optimize.NonlinearConstraint | None:

        """
        Generates a nonlinear distance constraint for the optimiser.

        Parameters
        ----------
        max_distance_constraint : NumpyNumber or None
            Maximum allowed normalised distance from the training data.

        Returns
        -------
        distance_constraint : scipy.optimize.NonlinearConstraint or None
            The constructed nonlinear constraint, or None if not specified.

        Notes
        -----
        The constraint ensures that solutions are not too far from the training data in 
        feature space.
        """

        if max_distance_constraint:
            self._distance_constraint = optimize.NonlinearConstraint(
                fun = self.distance_function,
                lb = 0.,
                ub = max_distance_constraint,
            )
        else:
            self._distance_constraint = None
        return self._distance_constraint


    def set_up_optimisation(
        self,
        *,
        model: EstimatorProtocol,
        training_data: np.ndarray[tuple[int, int], np.dtype[np.number]],
        feature_calculators: dict[str, FeatureCalculator],
        user_defined_features: Mapping[str, NumpyNumber],
        variable_bounds: Mapping[str, tuple[NumpyNumber, NumpyNumber]],
        variable_prices: Mapping[str, NumpyNumber],
        minimise_output: bool,
        sum_constraints: Iterable[SumConstraintSpecifier] | None = None,
        nonzero_constraints: Iterable[NonzeroConstraintSpecifier] | None = None,
        price_constraint: NumpyNumber | None = None,
        max_distance_constraint: NumpyNumber | None = None,
    ) -> None:

        """
        Configures the optimiser with all required data, constraints, and model 
        information.

        Parameters
        ----------
        model : EstimatorProtocol
            The estimator model to be optimised.
        training_data : np.ndarray
            Training data array of shape (n_samples, n_features).
        feature_calculators : dict
            Dictionary mapping feature names to feature calculator callables.
        user_defined_features : mapping
            Mapping from feature names to their fixed values.
        variable_bounds : mapping
            Mapping from variable names to (lower, upper) bounds.
        variable_prices : mapping
            Mapping from variable names to their prices.
        minimise_output : bool
            Whether to minimise (True) or maximise (False) the model output.
        sum_constraints : iterable of tuple, optional
            Iterable of (variable_names, lower_bound, upper_bound) specifying sum 
            constraints.
        nonzero_constraints : iterable of tuple, optional
            Iterable of (feature_names, min_nonzero) specifying nonzero constraints.
        price_constraint : NumpyNumber, optional
            Upper bound on the total price of variables.
        max_distance_constraint : NumpyNumber, optional
            Maximum allowed normalised distance from the training data.

        Raises
        ------
        FeatureMismatchError
            If there is a mismatch between variable prices and variable bounds.

        Notes
        -----
        This method must be called before `perform_optimisation`.
        """

        if (feature_mismatch := variable_prices.keys() ^ variable_bounds.keys()):
            raise FeatureMismatchError(
                f"Feature mismatch between variable_prices and declared variables: " \
                f"{', '.join(feature_mismatch)}."
            )

        self.model = model
        self.training_data = training_data
        self.feature_calculators = feature_calculators
        self.user_defined_features = user_defined_features
        self.feature_calculators = feature_calculators
        self.variable_bounds = variable_bounds
        self.variable_prices = variable_prices
        self.minimise_output = minimise_output
        self.price_constraint = price_constraint
        self.sum_constraints = sum_constraints
        self.nonzero_constraints = nonzero_constraints
        self.max_distance_constraint = max_distance_constraint


        self._verify_and_register_feature_names(
            model = self.model,
            training_data = self.training_data,
            variables_keys = self.variable_bounds.keys(),
            user_defined_features_keys = self.user_defined_features.keys(),
            feature_calculators_keys = self.feature_calculators.keys(),
        )
        self._distance_metric = get_pairwise_distance_metric(
            data = self.training_data,
        )
        self._training_data_bounds = get_data_bounds(
            data = self.training_data,
            dtype = np.float32,
        )

        self._register_problem_bounds(
            variable_bounds = self.variable_bounds,
            training_data_bounds = self._training_data_bounds,
        )

        self._training_data_scaled = scale_data_to_bounds(
            data = training_data,
            bounds = self._bounds_array,
            copy = True,
            dtype = np.float32,
        )

        self._create_price_array(prices=variable_prices)
        self._create_base_row(user_defined_features = user_defined_features)

        self._constraints = []
        if (linear_constraint := self._generate_linear_constraint(
            sum_constraints = sum_constraints,
            price_constraint = price_constraint,
        )):
            self._constraints.append(linear_constraint)

        if (distance_constraint := self._generate_distance_constraint(
            max_distance_constraint = max_distance_constraint
        )):
            self._constraints.append(distance_constraint)

        self._generate_bounds_combinations(nonzero_constraints=nonzero_constraints)


    @overload
    def perform_optimisation(
        self,
        *,
        iterations: int = 1000,
        display: bool = False,
        return_all_results: Literal[False] = False,
        population_size: int = 15,
        mutation: tuple[float, float] | float = (0.5, 1),
        recombination: float = 0.7,
        strategy: DifferentialEvolutionStrategy = DifferentialEvolutionStrategy.BEST1BIN,
        initial_configuration: DifferentialEvolutionInitialConfiguration = DifferentialEvolutionInitialConfiguration.SOBOL,
    ) -> optimize.OptimizeResult:
        ...


    @overload
    def perform_optimisation(
        self,
        *,
        iterations: int = 1000,
        display: bool = False,
        return_all_results: Literal[True] = True,
        population_size: int = 15,
        mutation: tuple[float, float] | float = (0.5, 1),
        recombination: float = 0.7,
        strategy: DifferentialEvolutionStrategy = DifferentialEvolutionStrategy.BEST1BIN,
        initial_configuration: DifferentialEvolutionInitialConfiguration = DifferentialEvolutionInitialConfiguration.SOBOL,
    ) -> list[optimize.OptimizeResult]:
        ...


    def perform_optimisation(
        self,
        *,
        iterations: int = 1000,
        display: bool = False,
        return_all_results: bool = False,
        population_size: int = 15,
        mutation: tuple[float, float] | float = (0.5, 1),
        recombination: float = 0.7,
        strategy: DifferentialEvolutionStrategy = DifferentialEvolutionStrategy.BEST1BIN,
        initial_configuration: DifferentialEvolutionInitialConfiguration = DifferentialEvolutionInitialConfiguration.SOBOL,
    ) -> list[optimize.OptimizeResult] | optimize.OptimizeResult:

        """
        Performs the constrained optimisation using differential evolution.

        Parameters
        ----------
        iterations : int, optional
            Maximum number of generations over which the entire population is evolved 
            (default: 1000).
        display : bool, optional
            If True, progress and bounds combinations are printed to the console.
        return_all_results : bool, optional
            If True, returns a list of results for all bounds combinations; otherwise, 
            returns the best result.
        population_size : int, optional
            Number of individuals in the population (default: 15).
        mutation : float or tuple of float, optional
            Mutation constant or range (default: (0.5, 1)).
        recombination : float, optional
            Recombination constant (default: 0.7).
        strategy : str, optional
            Differential evolution strategy to use (default: 'best1bin').
        initial_configuration : str, optional
            Method for initial population generation (default: 'sobol').

        Returns
        -------
        result : scipy.optimize.OptimizeResult or list of OptimizeResult
            The best optimisation result, or a list of results if `return_all_results` is
            True.

        Notes
        -----
        This method iterates over all generated bounds combinations and runs the 
        optimiser for each.
        The best result is selected based on the objective function value.

        Examples
        --------
        >>> result = optimiser.perform_optimisation(iterations=500, display=True)
        """

        self.history = []
        results = []
        for n, bounds_combination in tqdm(
            enumerate(self._bounds_combinations),
            desc = 'iterating through bounds combinations',
        ):
            if display and n:
                print('\n\nNEXT BOUNDS COMBINATION')

            results.append(optimize.differential_evolution(
                func = self.loss_function,
                bounds = bounds_combination,
                constraints = self._constraints,
                maxiter = iterations,
                polish = False,
                updating = 'deferred',
                vectorized = True,
                integrality = None,
                workers = 1,
                strategy = strategy,
                init = initial_configuration,
                popsize = population_size,
                mutation = mutation,
                recombination = recombination,
                disp = display,
            ))

        if return_all_results:
            return results
        else:
            return min(results, key = op.attrgetter('fun'))