# This file is used to implement the 95% CI Clopper-Pearson method used by Conceptual Network method.
# Originally, it relied on statsmodels/scipy, which is a very large package
from . import *

# copied from https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
def contfractbeta(a,b,x, ITMAX = 200):
    """ contfractbeta() evaluates the continued fraction form of the incomplete Beta function; incompbeta().  
    (Code translated from: Numerical Recipes in C.)
    Parameters
    ----------
    a : float
        Alpha parameter of the beta distribution (a > 0).
    b : float
        Beta parameter of the beta distribution (b > 0).
    x : float
        The evaluation point (0 <= x <= 1).
    ITMAX : int, optional
        Maximum number of iterations (default is 200).

    Returns
    -------
    float
        Continued fraction approximation of the incomplete beta function.
    """ 
    EPS = 3.0e-7
    bm = az = am = 1.0
    qab = a+b
    qap = a+1.0
    qam = a-1.0
    bz = 1.0-qab*x/qap
    for i in range(ITMAX+1):
        em = float(i+1)
        tem = em + em
        d = em*(b-em)*x/((qam+tem)*(a+tem))
        ap = az + d*am
        bp = bz+d*bm
        d = -(a+em)*(qab+em)*x/((qap+tem)*(a+tem))
        app = ap+d*az
        bpp = bp+d*bz
        aold = az
        am = ap/bpp
        bm = bp/bpp
        az = app/bpp
        bz = 1.0
        if (abs(az-aold)<(EPS*abs(az))):
            return az
    print ('a or b too large or given ITMAX too small for computing incomplete beta function.')

# copied from https://malishoaib.wordpress.com/2014/04/15/the-beautiful-beta-functions-in-raw-python/
# same as scipy.special.betainc within rounding
# normalized incomplete beta is same as beta cdf
def incomplete_beta(a, b, x):
    ''' incompbeta(a,b,x) evaluates incomplete beta function, here a, b > 0 and 0 <= x <= 1. This function requires contfractbeta(a,b,x, ITMAX = 200) 
    (Code translated from: Numerical Recipes in C.)
    Parameters
    ----------
    a : float
        Alpha parameter (a > 0).
    b : float
        Beta parameter (b > 0).
    x : float
        Evaluation point (0 <= x <= 1).

    Returns
    -------
    float
        Value of the regularized incomplete beta function at `x`.
    '''
    if (x == 0):
        return 0;
    elif (x == 1):
        return 1;
    else:
        lbeta = math.lgamma(a+b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1-x)
        if (x < (a+1) / (a+b+2)):
            return math.exp(lbeta) * contfractbeta(a, b, x) / a;
        else:
            return 1 - math.exp(lbeta) * contfractbeta(b, a, 1-x) / b;

# implements beta ppf
# same result as stats.beta.ppf(alpha_2, a, b)
def ppf(alpha_2, a, b, lower=0.0, upper=1.0, span=11, maxiter=20):
    """Invert the incomplete beta function to find the quantile (percent point function).

    Uses binary search to find the quantile corresponding to a given cumulative
    probability `alpha_2`. This approximates the inverse of the beta CDF.

    Parameters
    ----------
    alpha_2 : float
        Target probability (between 0 and 1).
    a : float
        Alpha parameter of the beta distribution.
    b : float
        Beta parameter of the beta distribution.
    lower : float, optional
        Lower bound for the search interval (default is 0.0).
    upper : float, optional
        Upper bound for the search interval (default is 1.0).
    span : int, optional
        Number of evenly spaced points to test in the interval (default is 11).
    maxiter : int, optional
        Maximum recursion depth for binary search (default is 20).

    Returns
    -------
    float
        Quantile corresponding to `alpha_2`. """
    if alpha_2 == 1.0:
        return 1.0
    elif alpha_2 == 0.0:
        return 0.0
    nprange = np.linspace(lower, upper, span)
    highlow = [incomplete_beta(a, b, x) > alpha_2 for x in nprange]
    idx_of_true = [idx for idx, x in enumerate(highlow) if x == True]
    if len(idx_of_true) == span:
        return lower
    elif len(idx_of_true) == 0:
        return upper
    else:
        if maxiter == 0:
            return nprange[idx_of_true[0]]
        else:
            return ppf(alpha_2, a, b, lower=nprange[idx_of_true[0]-1], upper=nprange[idx_of_true[0]], maxiter=(maxiter-1))

# same result as stats.beta.ppf(alpha_2, count, nobs - count + 1) (from statsmodels)
def pci_lowerbound(cooccur, total, alpha):
    """
    Compute the lower bound of a confidence interval using the Clopper-Pearson method.

    This is a conservative (exact) method for estimating a binomial proportion's 
    confidence interval using the inverse incomplete beta function.

    Parameters
    ----------
    cooccur : int
        Number of successes (co-occurrences).
    total : int
        Total number of trials.
    alpha : float
        Significance level (e.g., 0.05 for 95% CI).

    Returns
    -------
    float
        Lower bound of the (1 - alpha) confidence interval.
    """
    alpha_2 = alpha * 0.5
    return ppf(alpha_2, cooccur, total - cooccur + 1)
