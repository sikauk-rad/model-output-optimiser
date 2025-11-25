class FeatureMismatchError(KeyError):
    """
    Raised when there is a mismatch between expected and provided feature names.

    Notes
    -----
    This exception is typically raised when the input features do not match the expected 
    set, such as during validation or transformation steps.
    """
    ...


class NoFeatureNamesError(TypeError):
    """
    Raised when feature names are required but not provided.

    Notes
    -----
    This exception is typically raised when an operation depends on feature names, but 
    they are missing or not set.
    """
    ...


class DuplicateFeatureError(ValueError):
    """
    Raised when duplicate feature names are detected.

    Notes
    -----
    This exception is typically raised during feature validation to ensure all feature 
    names are unique.
    """
    ...


class BoundError(ValueError):
    """
    Raised when a value violates specified bounds or constraints.

    Notes
    -----
    This exception is typically raised when input data or computed values fall outside 
    allowed limits.
    """
    ...


class ShapeError(ValueError):
    """
    Raised when an array or input has an unexpected or invalid shape.

    Notes
    -----
    This exception is typically raised during validation of input shapes for arrays or 
    matrices.
    """
    ...