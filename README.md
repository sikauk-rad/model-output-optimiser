# Constrained Estimator Optimiser
A future-proof, NumPy-typed Python package for constrained optimisation of machine learning estimators, supporting advanced feature handling, custom constraints, and robust integration with SciPy and scikit-learn.

## Overview
This package provides a flexible and extensible framework for optimising variables of machine learning estimators under a variety of constraints, including:

Variable bounds (per-feature lower and upper limits)
Sum constraints (e.g., total allocation or budget)
Nonzero constraints (e.g., minimum number of active features)
Distance constraints (e.g., solutions must remain close to training data)
Custom feature calculators (derived or engineered features)
The package is designed for future compatibility with NumPy 3 typing and leverages modern Python type annotations for clarity and safety.

## Features
Advanced NumPy typing for all arrays and functions
Protocol-based estimator interface for model flexibility
Customisable constraints for real-world optimisation problems
Integration with SciPy's differential evolution optimiser
Support for missing data and robust distance metrics
Extensive error checking and informative exceptions

## Installation
Clone from git and install!
Note: This package requires Python 3.10+ and NumPy 2.3+ for advanced typing support.

Quick Start
```
from model_output_optimiser import ConstrainedEstimatorOptimiser

# Assume you have a trained model, training data, and feature definitions
optimiser = ConstrainedEstimatorOptimiser()
optimiser.set_up_optimisation(
    model=my_model,
    training_data=X_train,
    feature_calculators=feature_calculators,
    user_defined_features=user_defined_features,
    variable_bounds=variable_bounds,
    variable_prices=variable_prices,
    minimise_output=True,
    sum_constraints=sum_constraints,
    nonzero_constraints=nonzero_constraints,
    price_constraint=price_constraint,
    max_distance_constraint=max_distance_constraint,
)

result = optimiser.perform_optimisation(
    iterations=1000,
    display=True,
    return_all_results=False,
)
print(result.x, result.fun)
```

## Key Classes and Functions
ConstrainedEstimatorOptimiser
A class for setting up and performing constrained optimisation on estimator models.

set_up_optimisation: Configures the optimiser with all required data and constraints.
perform_optimisation: Runs the optimisation and returns the best result.
Utilities
validate_bounds: Validates that bounds are logically consistent.
scale_data_to_bounds: Scales data to the specified feature bounds.
generate_bounds_combinations: Generates all possible bounds arrays under nonzero constraints.
Exceptions
FeatureMismatchError: Raised when feature sets do not match.
NoFeatureNamesError: Raised when feature names are missing.
DuplicateFeatureError: Raised when duplicate features are detected.
BoundError: Raised when bounds are violated.
ShapeError: Raised when array shapes are invalid.

## Typing and Protocols
This package uses advanced typing features, including:

NumPy-style array typing (e.g., np.ndarray[tuple[int, int], np.dtype[np.number]])
Protocols for estimator interfaces
Type aliases for numeric types and constraints

## Contributing
Contributions are welcome! Please:

Fork the repository and create a feature branch.
Add tests for new features or bug fixes.
Ensure all code is type-checked and passes linting.
Submit a pull request with a clear description.

## License
MIT License

## Acknowledgements
Built on top of NumPy, SciPy, and scikit-learn.
Inspired by best practices in scientific Python and open-source software.

## Contact
For questions, suggestions, or bug reports, please open an issue or contact the maintainer.

This package is under active development and aims to be fully compatible with future NumPy typing standards.