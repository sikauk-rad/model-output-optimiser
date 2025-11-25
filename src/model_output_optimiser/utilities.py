from collections.abc import Iterable, Mapping, Sequence
import itertools
from math import comb
from typing import Literal
from warnings import warn

import numpy as np

from app_utilities.numbers import get_optimal_uintype

from .datatypes import NumpyNumber
from .exceptions import BoundError, FeatureMismatchError, ShapeError


def validate_bounds(
    lower_bound: NumpyNumber,
    upper_bound: NumpyNumber,
    lower_bound_min: NumpyNumber,
    lower_bound_max: NumpyNumber,
    upper_bound_min: NumpyNumber,
    upper_bound_max: NumpyNumber,
    name: str | None = None,
    limit_name: str | None = None,#
) -> None:

    """
    Validates that the provided bounds are within specified limits.

    Parameters
    ----------
    lower_bound : NumpyNumber
        The lower bound value to validate.
    upper_bound : NumpyNumber
        The upper bound value to validate.
    lower_bound_min : NumpyNumber
        Minimum allowed value for the lower bound.
    lower_bound_max : NumpyNumber
        Maximum allowed value for the lower bound.
    upper_bound_min : NumpyNumber
        Minimum allowed value for the upper bound.
    upper_bound_max : NumpyNumber
        Maximum allowed value for the upper bound.
    name : str or None, optional
        Name of the variable being validated, used in error messages.
    limit_name : str or None, optional
        Name of the limit being validated against, used in error messages.

    Raises
    ------
    BoundError
        If any of the bounds are outside the specified limits.

    Notes
    -----
    This function is useful for validating parameter ranges and ensuring logical 
    consistency.
    """

    name = f'of {name} ' if name else ''
    limit_name = f'{limit_name} ' if limit_name else 'limit of '
    if lower_bound > upper_bound:
        raise BoundError(f'lower bound {name}exceeds upper bound.')
    elif lower_bound < lower_bound_min:
        raise BoundError(f'lower bound {name}deceeds {limit_name}{lower_bound_min}.')
    elif lower_bound > lower_bound_max:
        raise BoundError(f'lower bound {name}exceeds {limit_name}{lower_bound_max}.')
    elif upper_bound < upper_bound_min:
        raise BoundError(f'upper bound {name}deceeds {limit_name}{upper_bound_min}.')
    elif upper_bound > upper_bound_max:
        raise BoundError(f'upper bound {name}exceeds {limit_name}{upper_bound_max}.')


def get_data_bounds[T: int, U: np.number](
    data: np.ndarray[tuple[T, int], np.dtype[np.number]],
    dtype: type[U] = np.float32
) -> np.ndarray[tuple[T, Literal[2]], np.dtype[U]]:

    """
    Compute the minimum and maximum values for each feature (column) in the input data.

    For each feature in the input array, this function calculates the minimum and maximum
    values, ignoring NaNs, and returns them as a (n_features, 2) array, where the first
    column contains the minima and the second column contains the maxima.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, n_features)
        Input data array. Each row corresponds to a sample and each column to a feature.
    dtype : type, default=np.float32
        Data type of the output array.

    Returns
    -------
    bounds : np.ndarray of shape (n_features, 2), dtype=dtype
        Array containing the minimum and maximum value for each feature. The first column
        contains the minima, the second column contains the maxima.

    Raises
    ------
    ValueError
        If the input array is not two-dimensional.
    ValueError
        If the input array has one or no rows.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
    >>> get_data_bounds(data)
    array([[1., 7.],
           [2., 8.],
           [3., 9.]], dtype=float32)
    """

    if data.ndim != 2:
        raise ShapeError(f'Data must be two-dimensional; {data.ndim = }.')
    elif data.shape[0] < 2:
        raise ShapeError(f'Data must contain more than one row; {data.shape[0] = }.')
    bounds = np.empty(shape = (data.shape[1], 2), dtype = dtype)
    np.nanmin(data, axis = 0, out = bounds[:,0])
    np.nanmax(data, axis = 0, out = bounds[:,1])
    return bounds


def scale_data_to_bounds[T: int, U: int, V: np.floating](
    data: np.ndarray[tuple[T, U], np.dtype[np.number]],
    bounds: np.ndarray[tuple[U, Literal[2]], np.dtype[np.number]],
    copy: bool = True,
    dtype: type[V] = np.float32,
) -> np.ndarray[tuple[T, U], np.dtype[V]]:

    """
    Scales each feature of the input data to the [0, 1] range according to the provided 
    bounds.

    Each column (feature) in `data` is scaled independently using the corresponding min 
    and max values from `bounds`. The transformation applied is:
        scaled = (data - min) / (max - min)

    Parameters
    ----------
    data : np.ndarray[tuple[T, U], np.dtype[np.number]]
        Input data array of shape (n_samples, n_features).
    bounds : np.ndarray[tuple[U, Literal[2]], np.dtype[np.number]]
        Array of shape (n_features, 2) containing lower and upper bounds for each 
        feature.
    copy : bool, optional
        If True, the input data is copied before scaling (default is True).
    dtype : type, optional
        Desired output data type (default is np.float32).

    Returns
    -------
    scaled_data : np.ndarray[tuple[T, U], np.dtype[V]]
        Scaled data array of shape (n_samples, n_features).

    Raises
    ------
    ShapeError
        If the number of features in data and bounds do not match.
    ZeroDivisionError
        If any feature has identical lower and upper bounds.

    Notes
    -----
    Each feature is scaled to the interval [0, 1] based on its bounds.
    """

    if copy:
        data = data.copy()
    bounds = bounds.astype(dtype)
    data = data.astype(dtype)
    if data.shape[1] != bounds.shape[0]:
        raise ShapeError(
            f'{data.shape[1] = } is not broadcastable to {bounds.shape[0] = }.'
        )
    data -= bounds[:,0]
    bounds_diff = bounds[:,1] - bounds[:,0]
    bounds_diff_0_mask = bounds_diff == 0
    if bounds_diff_0_mask.any():
        raise ZeroDivisionError(
            'Upper and lower bounds are the same at positions ' \
            f'{", ".join(bounds_diff_0_mask.nonzero()[0])}.'
        )
    else:
        return data / bounds_diff


def generate_numeric_combinations(
    n: int | np.integer,
    k: int | np.integer,
) -> np.ndarray[tuple[int, int], np.dtype[np.uint]]:

    """
    Generate all possible combinations of `k` distinct elements chosen from the range 
    [0, n).

    Each combination is represented as a sorted tuple of indices. The function returns an
    array where each row is one combination.

    Parameters
    ----------
    n : int or np.integer
        The size of the set to choose elements from. Elements are taken from the range 
        [0, n).
    k : int or np.integer
        The number of elements in each combination.

    Returns
    -------
    combinations : np.ndarray of shape (C, k), dtype=uint
        An array containing all possible combinations, where C = binomial(n, k).
        Each row is a combination of `k` unique indices in increasing order.

    Examples
    --------
    >>> generate_numeric_combinations(4, 2)
    array([[0, 1],
           [0, 2],
           [0, 3],
           [1, 2],
           [1, 3],
           [2, 3]], dtype=uint8)  # dtype may vary depending on n

    >>> generate_numeric_combinations(3, 3)
    array([[0, 1, 2]], dtype=uint8)

    Notes
    -----
    - The output dtype is an unsigned integer type chosen to be large enough to hold 
    values up to `n`.
    - The function uses `itertools.combinations` to generate the combinations.
    - If `k > n`, the result will be an empty array.
    - The function assumes that `get_optimal_uintype` and `comb` are available in the 
    scope.
    - The output array has shape (binomial(n, k), k).
    - No input validation is performed beyond what is described above.

    Warns
    -----
    UserWarning
        If n < k, no combinations are possible.
    """

    dtype = get_optimal_uintype(max(n,k))
    n_combinations = comb(n, k)
    if not n_combinations:
        warn(
            message=f'No combinations possible as {n = } < {k = }.',
            category = UserWarning,
        )
    return np.fromiter(
        itertools.combinations(np.arange(n, dtype = dtype), k),
        count = n_combinations,
        dtype = np.dtype((dtype, k)),
    )


def generate_bounds_combinations[T: int, U: np.floating](
    bounds: Mapping[str, tuple[NumpyNumber, NumpyNumber]] | np.ndarray[tuple[T, Literal[2]], np.dtype[np.number]],
    nonzero_constraints: Iterable[tuple[Sequence[str], int]],
    feature_names: Sequence[str] | None = None,
    dtype: type[U] = np.float32,
) -> np.ndarray[tuple[int, T, Literal[2]], np.dtype[U]]:

    """
    Generate all possible combinations of bounds arrays under nonzero constraints for 
    features.

    This function creates a set of bounds arrays, each corresponding to a unique way of 
    setting exactly `k` features (from a specified group) to have nonzero bounds, while 
    the rest are set to zero. Multiple such constraints can be applied sequentially, and 
    the function returns all possible combinations that satisfy all constraints.

    Parameters
    ----------
    bounds : dict[str, tuple[int | float | np.number, int | float | np.number]] or 
    np.ndarray of shape (T, 2)
        The lower and upper bounds for each feature. If a dictionary is provided, keys 
        are feature names and values are (min, max) tuples. If an array is provided, 
        `feature_names` must also be given.
    nonzero_constraints : Iterable of tuple[Sequence[str], int]
        An iterable of constraints. Each constraint is a tuple of (feature_names, k), 
        where exactly `k` features from the given Sequence must have nonzero bounds in 
        each combination.
    feature_names : Sequence[str] or None, optional
        The names of the features, required if `bounds` is provided as an ndarray.
    dtype : type, default=np.float32
        The data type for the output array.

    Returns
    -------
    bounds_combinations : np.ndarray of shape (N, T, 2), dtype=dtype
        An array containing all possible bounds arrays that satisfy the constraints. N is
        the total number of valid combinations, T is the number of features, and the last 
        dimension contains the (min, max) bounds for each feature.

    Raises
    ------
    FeatureMismatchError
        If any unmatching features are found.

    Examples
    --------
    >>> bounds = {'a': (0, 1), 'b': (0, 2), 'c': (0, 3)}
    >>> nonzero_constraints = [(['a', 'b'], 1)]
    >>> generate_bounds_combinations(bounds, nonzero_constraints)
    array([[[0., 1.], [0., 0.], [0., 3.]],
           [[0., 0.], [0., 2.], [0., 3.]]], dtype=float32)

    Notes
    -----
    - For each constraint, exactly `k` features from the specified group will have 
    nonzero bounds; the rest in the group will be set to zero.
    - Multiple constraints are applied sequentially, and the output contains all 
    combinations that satisfy all constraints.
    - If `bounds` is a dict, feature names are inferred from the keys. If it is an 
    ndarray, `feature_names` must be provided.
    - The function assumes that `generate_numeric_combinations` and `get_optimal_uintype`
    are available.
    - No input validation is performed beyond what is described above.
    - The output dtype is determined by the `dtype` parameter.
    """

    if isinstance(bounds, Mapping):
        feature_names_arr = np.array([*bounds.keys()], dtype = np.str_)
        bounds = np.array([*bounds.values()], dtype = dtype)
    elif feature_names is None:
        raise TypeError('if bounds is NDArray feature_names must be provided.')
    else:
        feature_names_arr = np.array([*feature_names], dtype = np.str_)
        bounds = bounds.astype(dtype)

    n_bounds = bounds.shape[0]
    if not nonzero_constraints:
        return bounds[None]

    for n, (constrained_ings, k) in enumerate(nonzero_constraints):
        # finding mask of features involved in current constraint
        bound_locs = np.isin(feature_names_arr, constrained_ings)
        n_nonzero_bounds = bound_locs.sum()
        # checking all constrained features exist
        if n_nonzero_bounds != len(constrained_ings):
            raise FeatureMismatchError(
                'some features not recognised in nonzero constraints.'
            )
        # finding all possible ways to choose `k` features from all `n_nonzero_bounds`
        # constrained features.
        bool_combos = generate_numeric_combinations(
            n_nonzero_bounds,
            k,
        )
        # creating a mask for nonzero bounds in each combination.
        n_combos = bool_combos.shape[0]
        n_arange = np.arange(n_combos, dtype = get_optimal_uintype(n_combos))
        bound_mask = np.zeros((n_combos, n_nonzero_bounds), dtype = np.bool_)
        bound_mask[n_arange[:,None], bool_combos] = True

        # mapping mask to full feature set
        bound_locs_2d = bound_locs[None].repeat(n_combos, axis = 0)
        bound_locs_2d[:,bound_locs] = ~bound_mask

        if not n:
            # initialising combined_bounds_zero_mask on first iteration.
            combined_bounds_zero_mask = bound_locs_2d
        else:
            # if combined_bounds_zero_mask is already initialised, calculate the 
            # Cartesian product of the already existing bounds zero mask and the new one,
            # then reshape it to be 2-dimensional.
            combined_bounds_zero_mask = (
                np.bitwise_or(combined_bounds_zero_mask[:,None], bound_locs_2d)
                .reshape(-1, n_bounds)
            )

    # after full combined_bounds_zero_mask is generated, use it to create a new array of 
    # bounds wherein for each new combination, a new combination of variables is set to 
    # be nonzero.
    bounds_3d = bounds[None].repeat(combined_bounds_zero_mask.shape[0], axis = 0)
    bounds_3d[combined_bounds_zero_mask] = 0
    return bounds_3d


def get_pairwise_distance_metric(
    data: np.ndarray[tuple[int, int], np.dtype[np.number]],
) -> Literal['euclidean', 'nan_euclidean']:

    """
    Determines the appropriate pairwise distance metric for the given data.

    Parameters
    ----------
    data : np.ndarray[tuple[int, int], np.dtype[np.number]]
        Input data array.

    Returns
    -------
    metric : {'euclidean', 'nan_euclidean'}
        'nan_euclidean' if the data contains NaN values, otherwise 'euclidean'.

    Notes
    -----
    Useful for selecting the correct metric for distance computations.
    """

    return 'nan_euclidean' if np.isnan(data).any() else 'euclidean'


def get_element_positions[T: (str, int, float), U: (np.number, np.str_), V: int, W: int](
    sequence: Sequence[T] | np.ndarray[tuple[W], np.dtype[U]],
    elements: Sequence[T] | np.ndarray[tuple[V], np.dtype[U]],
    sorter: Sequence[int] | np.ndarray[tuple[W], np.dtype[np.integer]] | None = None,
) -> np.ndarray[tuple[V], np.dtype[np.integer]]:

    """
    Find the indices of specified elements within a sequence or array.

    For each element in `elements`, this function returns the index of its position
    within `sequence`. If a `sorter` is provided, it is used to sort `sequence` before
    searching. If not, the function sorts `sequence` automatically.

    .. warning::
        This function does not check whether all elements are present in `sequence`. If 
        an element is missing, the result may be incorrect or untrustworthy.

    Parameters
    ----------
    sequence : Sequence or np.ndarray
        The sequence or array in which to search for elements.
    elements : Sequence or np.ndarray
        The elements whose positions are to be found in `sequence`.
    sorter : Sequence[int] or np.ndarray or None, optional
        Optional indices that sort `sequence` into ascending order. If None, the function
        will sort `sequence` automatically.

    Returns
    -------
    positions : np.ndarray of shape (len(elements),), dtype=np.integer
        Array of indices indicating the positions of each element from `elements` in
        `sequence`.

    Raises
    ------
    None explicitly, but results are unreliable if any element in `elements` is not
    present in `sequence`.

    Examples
    --------
    >>> import numpy as np
    >>> sequence = np.array(['a', 'b', 'c', 'd'])
    >>> elements = np.array(['b', 'd'])
    >>> get_element_positions(sequence, elements)
    array([1, 3])

    >>> sequence = np.array([10, 20, 30, 40])
    >>> elements = np.array([30, 10])
    >>> get_element_positions(sequence, elements)
    array([2, 0])

    >>> sequence = np.array([5, 2, 8, 1])
    >>> elements = np.array([2, 8])
    >>> get_element_positions(sequence, elements)
    array([1, 2])
    """

    if sorter is None:
        sorter_array = np.argsort(sequence)
    elif isinstance(sorter, Sequence):
        sorter_array = np.array(sorter, dtype = get_optimal_uintype(max(sorter)))
    else:
        sorter_array = sorter

    return sorter_array[np.searchsorted(
        a= sequence,
        v = elements,
        side = 'left',
        sorter = sorter_array,
    )]