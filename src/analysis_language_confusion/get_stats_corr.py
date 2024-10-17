from scipy.stats import spearmanr
import numpy as np


def get_annotated_corr(df, corr_method="spearman"):
    rho = df.corr(method=corr_method)
    pval =df.corr(method=lambda x, y: spearmanr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x <= t]))
    # Combine rho (correlation coefficients) and p-values (significance stars)
    annotated = rho.round(2).astype(str) + p
    # Create a mask for the upper triangle and diagonal
    mask = np.triu(np.ones_like(rho, dtype=bool))

    # Mask the upper triangle of the annotated DataFrame
    annotated = annotated.mask(mask)
    return annotated