print(
    "Welcome to Jargon! To import the most common data science tools, call jargon.import_stack()"
)


def import_stack():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    print("Imported pandas as pd, numpy as np, pyplot as plt, seaborn as sns")


class NoWildCardError(Exception):
    pass


def nope():
    raise NoWildCardError("This is not Fast.AI. Import your stuff properly!")


__all__ = nope()

