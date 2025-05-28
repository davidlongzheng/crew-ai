def coerce_string(value):
    """
    Coerces a string value to its appropriate primitive type (int, float, bool, or string).

    Args:
        value: The string value to coerce.

    Returns:
        The coerced value, or the original string if coercion is not possible.
    """

    if value.lower() == "none":
        return None

    if value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False

    if value.startswith("("):
        assert value.endswith(")")
        return tuple(
            int(x) for x in value.removeprefix("(").removesuffix(")").split("_")
        )

    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value
