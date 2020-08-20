"""
SAR flood mapping: open water and inundated vegetation

This module implements algorithms described in:
"Flood mapping under vegetation using single SAR acquisitions",
S. Grimaldi et al, RSE 111582 (2020)
https://doi.org/10.1016/j.rse.2019.111582

Verify tests by executing:  python -m doctest algorithm.py
"""

import numpy as np
import scipy.optimize, scipy.stats

def fuzzy_smf(x, a, b):
    """
    Fuzzy step function
    
    Examples:
    >>> x = np.arange(7)
    >>> fuzzy_smf(x, 3, 5).tolist()
    [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0]
    >>> fuzzy_zmf(x, 3, 5).tolist()
    [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0]
    """
    delta = b - a
    middle = np.mean([a, b])
    
    A = 2 * ((x - a) / delta)**2
    B = 2 * ((x - b) / delta)**2
    
    result = np.where(x < middle, A, 1 - B)
    result[x < a] = 0
    result[x > b] = 1
    return result

def fuzzy_zmf(x, a, b):
    return 1 - fuzzy_smf(x, a, b)

def hist_fixedwidth(population, binwidth=0.1):
    """
    Return (frequency, centres) for a histogram with uniform-width bins
    
    >>> freq, centres = hist_fixedwidth(np.asarray([5, 9.5, 10.5, 6, 6, 6, 6, 6]), binwidth=2)
    >>> centres.tolist()
    [6.0, 8.0, 10.0]
    >>> freq.tolist() # 6/8 per 2; 0 per 2; 2/8 per 2
    [0.375, 0.0, 0.125]
    """
    # Could replace .max() with 0dB to discard non-negative samples?
    edges = np.arange(start=population.min(), stop=population.max() + binwidth, step=binwidth)
    counts, edges = np.histogram(population, bins=edges, density=True)
    centres = (edges[:-1] + edges[1:]) / 2
    return counts, centres

def find_mode(population):
    """
    Returns estimate of distribution mode.
    
    >>> round(find_mode(np.asarray([-10.05, 20.31, 20.29, 17, 20.27, 17, 37])), 1)
    20.3
    """
    counts, values = hist_fixedwidth(population)
    return values[counts.argmax()]

def leftFitNormal(population):
    """
    Obtain mode and standard deviation from the left side of a population.
    
    >>> pop = np.random.normal(loc=-20, scale=3, size=10000)
    >>> mode, sigma = leftFitNormal(pop)
    >>> -22 < mode < -18
    True
    >>> round(sigma)
    3
    
    >>> pop[pop > -18] += 10     # perturb right side
    >>> mode, sigma = leftFitNormal(pop)
    >>> -22 < mode < -18
    True
    >>> round(sigma) == 3
    True
    
    >>> pop[pop < -22] -= 10     # perturb left side
    >>> mode, sigma = leftFitNormal(pop)
    >>> -22 < mode < -18
    True
    >>> round(sigma) == 3
    False
    """

    # Quick alternative robust fit:
    # median = np.nanmedian(population) 
    # MADstd = np.nanmedian(np.abs(population - median)) * 1.4826 
    # Could still modify this estimator to ignore samples > median.
    
    # Note, if the distribution is right-skewed or bimodal (e.g. if there is
    # some land amongst mostly open water) then other relative frequencies
    # will proportionally be depressed, favouring the fit of a broader
    # Gaussian (perhaps also shifted slightly rightward) to the left side
    # of the histogram (compared to if the distribution was normal).
    # Could address this by normalising the interval area.
    #
    # Currently the tests for perturbed distributions bypass this limitation
    # by _conditionally_ replacing existing samples, rather than by mixing
    # additional components into the population i.e. avoiding
    # pop[:5000] = np.linspace(-15, -5, 5000).
    
    std = np.nanstd(population) # naive initial estimate

    Y, X = hist_fixedwidth(population)
    
    # Take left side of distribution
    pos = Y.argmax()
    mode = X[pos]
    X = X[:pos+1]
    Y = Y[:pos+1]
    
    # fit gaussian to (left side of) distribution
    def gaussian(x, mean, sigma):
        return np.exp(-0.5 * ((x - mean)/sigma)**2) / (sigma * (2*np.pi)**0.5)
    (mean, std), cov = scipy.optimize.curve_fit(gaussian, X, Y, p0=[mode, std])
    
    return mode, std

def leftFitNormal2(population):
    """
    Obtain mode, right 1/20 maximum, and covariance of left fit.
    
    The following fits two distributions, the second of which is poorly
    modelled by the Gaussian:

    >>> mode, m20, measure = leftFitNormal2(np.random.normal(loc=-20, scale=3, size=10000))
    >>> -22 < mode < -18
    True
    >>> 0 < (m20 - mode) < 20
    True
    >>> 1 >= measure > 0.9
    True
    
    >>> mode, m20, measure = leftFitNormal2(np.random.normal(loc=-20, scale=3, size=10000)**2)
    >>> mode < m20
    True
    >>> measure > 0.9
    False
    """
    std = np.nanstd(population) # naive initial estimate

    Y, X = hist_fixedwidth(population)
    
    # Find the peak
    pos = Y.argmax()
    mode = X[pos]
    
    # Perform a one-sided fit (for standard deviation and with mode held fixed)
    def gaussian(x, sigma):
        return np.exp(-0.5 * ((x - mode)/sigma)**2) / (sigma * (2*np.pi)**0.5)
    std, cov = scipy.optimize.curve_fit(gaussian, X[:pos+1], Y[:pos+1], p0=std) # histogram fit
    
    # Calculate two-sided covariance of the fit
    double = min(len(X), 2*pos)
    correlation = np.corrcoef(gaussian(X[:double], std), Y[:double])[0, 1] # normalised covariance
    
    # Find the right intercept with 1/20th maximum
    pos20 = (abs(Y[pos:] - Y.max() / 20)).argmin()
    m20 = X[pos + pos20]
    
    return mode, m20, correlation


def leftFitGamma(population, limit, minpoints=20):
    """
    Left fit of Gamma distribution, to extract mode.
    
    Assumes the given limit is within ten units greater than the mode.
    
    >>> g = lambda k, mode, x0, n: np.random.gamma(k, (mode-x0)/(k-1), n) + x0
    >>> -12 < leftFitGamma(g(9, -10, -25, 5000), -5) < -8
    True
    
    >>> pop = g(8, -15, -40, 8000)
    >>> pop[pop > -14] += 30    # perturb right side
    >>> -17 < leftFitGamma(pop, -10) < -13
    True
    
    >>> leftFitGamma(np.linspace(-20, -10, 1000), -25) is None # limit too low
    True
    """
    # Note Gamma(k>=1, theta)+x0 has mode=(k-1)theta+x0.
    # This has curve scipy.stats.gamma(k, loc=x0, scale=theta).pdf(x),
    # and samples np.random.gamma(k, theta, n)+x0.
    
    # Note, must reject the inevitably great (but uninformative) fits that 
    # occur when there are too few datapoints, e.g. if there are too
    # few bins left of (limit-10).
    
    # TODO: Should there be a renormalisation for proper fitting, 
    # after excluding the right component of the sample set?
    
    Y, X = hist_fixedwidth(population)
    
    def fit(mode): # Fit shape parameter and report residual
        def gamma(x, k): # Gamma function with fixed mode and origin
            return scipy.stats.gamma.pdf(x, k, X[0], (mode - X[0])/(k - 1))
        left = (X < mode)
        
        k = scipy.optimize.curve_fit(gamma, X[left], Y[left], bounds=(1, np.inf))[0][0]

        mean_square_error = ((Y[left] - gamma(X[left], k))**2).mean()

        return mean_square_error, mode
    
    # Try different mode candidates (and skip where too few datapoints to fit)
    fits = dict(fit(mode) for mode in np.arange(limit - 10, limit, 0.1) if X[minpoints] <= mode)
    
    # select mode corresponding to candidate with best RMSE (and MSE)
    return fits[min(fits)] if len(fits) else None 
        
    
class Inseparable(Exception): # may be raised by chiSeparate
    pass

def chiSeparate(control, test, nbins=100):
    """
    Perform separability T-test and probability binning.
    
    Given two distributions that may overlap in the middle,
    try to find end-intervals that separate between the two.
    
    This algorithm bins asymmetrically by the control percentiles,
    and marks overlaps using a variant of Chi distance.
    
    >>> pop1 = np.random.normal(loc=-20, scale=3, size=5000)
    >>> pop2 = np.random.normal(loc=-10, scale=3, size=5000) # well-separated
    >>> a, b = chiSeparate(pop1, pop2)
    >>> -17 < a <= b < -13
    True
    >>> a, b = chiSeparate(pop2, pop1)
    >>> -17 < a <= b < -13
    True
    >>> chiSeparate(pop1, pop2-10) # no separation
    Traceback (most recent call last):
        ...
    algorithm.Inseparable: Cannot distinguish populations
    >>> a, b = chiSeparate(pop1 - 20, pop2) # too much separation
    >>> a < -45 and b == 0
    True
    """
    # Could this algorithm be made more robust to unexpected cases
    # such as the final example?
    
    n0 = len(control)
    n1 = len(test)
    minsample = min(n0, n1)
    if (minsample < 200) or (n0 < nbins):
        raise Inseparable
    
    edges = np.nanquantile(control, np.linspace(0, 1, nbins + 1))
    freq0 = 1 / nbins # (constant) relative frequency for control sample
    freq1 = np.histogram(test, edges)[0] / n1 # relative frequencies for test sample
    
    x = (edges[1:] + edges[:-1]) / 2 # the bin centers
    
    z2 = (freq0 - freq1)**2 / (freq0 + freq1)
    zk = (freq0 - freq1) / np.sqrt(freq0/n0 + freq1/n1)
    
    if (minsample * z2.sum() * nbins**-0.5 - nbins**0.5) < 4:
        raise Inseparable('Cannot distinguish populations')
    
    # find where in the domain that zk (interpolated) crosses -1.96
    y = np.poly1d(np.polyfit(x, zk + 1.96, deg=9))
    roots = y.roots.real[y.roots.imag == 0]
    roots = roots[(x[0] < roots) & (roots < x[-1])]
    
    if not len(roots):
        return control.min(), 0
    else:
        return roots.min(), roots.max()

def openwater(backscatter, persistent, historic):
    """
    Adaptively select thresholds for classifying areas of open water (low backscatter)
    
    
    >>> backscatter = np.random.normal(loc=-10, scale=5, size=6000) # land
    >>> backscatter[:1000] = np.random.normal(loc=-18, scale=2, size=1000) # openwater
    >>> persistent = np.ogrid[:6000] < 600 #　boolean: always open water
    >>> historic = np.ogrid[:6000] < 2000 # boolean: sometimes flooded
    >>> a, b = openwater(backscatter, persistent, historic)
    >>> #(backscatter[:1000] < a).mean()
    >>> #backscatter[:1000].mean()
    >>> #backscatter.mean()
    >>> #a,b
    
    """
    # TODO: ensure a lower bound, e.g. -40dB, in places.
    
    # group samples
    dark = backscatter[persistent]
    notdark = backscatter[~persistent]
    unprecedented = backscatter[~historic]
    wettable = backscatter[historic]
    
    # "Methodology 1" - fit left side of normal distribution to persistant waterbodies
    L1, std = leftFitNormal(dark)
    R1 = L1 + std
    
    # "Methodology 2" - find left-separability of distribution by persistance
    try:
        R2 = chiSeparate(notdark, dark)[0]
    except Inseparable:
        R2 = R1
    
    # "Methodology 3" - find left-separability of distribution by precedent
    try:
        R3 = chiSeparate(unprecedented, wettable)[0]
    except Inseparable:
        R3 = R1
    
    mode, std = leftFitNormal(notdark)
    upperbound = mode - 0.5 * std
    
    # Conclusion - decide which thresholds to return
    
    R = min(R1, R2, R3)
    
    if R > upperbound: # abort! (no apparent water)
        return [backscatter.min()] * 2 # TODO: should also require minimum is >-40dB

    L = L1 if R == R1 else np.mean([R, find_mode(dark < R2)])
        
    if L > R:
        L = leftFitGamma(wettable, R3) if R3 < R2 else None # try Matgen method
        if L is None:
            L = np.mean([R, wettable.min()])

    return L, R


def vegetation(backscatter, lowlying, precedent, v4=False):
    """
    Adaptively select thresholds for classifying inundated vegetation (high backscatter)
    
    Assumes open-water (very low backscatter) has already been filtered out.
    """
    
    test = backscatter[lowlying] # mixture wet and dry
    control = backscatter[~lowlying] # presume all dry
    
    S1 = chiSeparate(control, test)[1]
    
    mode, S2, correlation = leftFitNormal2(control)
    
    # Portion of samples within [S1, S2]
    def portion(x):
        return ((x > S1) & (x < S2)).mean()
    HO = portion(backscatter[lowlying & precedent])
    HNO = portion(backscatter[lowlying & ~precedent])
    C = portion(control)
    
    # Produce gamma exponents, to warp or bias the fuzzy interval
    enhance = 4 * (C / HO)**2
    attenuate = 4 * (C / HNO)**2
    
    if v4:
        try:
            # option for more rigorous consistency
            test = backscatter[precedent & lowlying]
            control = backscatter[~precedent & ~lowlying]
            
            S1 = chiSeparate(control, test)[1]
        except Inseparable:
            S1 = 0 
    
    return S1, S2, enhance, attenuate, correlation


def classify(backscatter, wofs, hand, landcover):
    """
    Generate probabilistic flood raster
    """
    
    # define input thresholds
    persistent = wofs > 0.8
    historic = wofs > 0.001
    lowlying = hand < 20
    
    ow1, ow2 = openwater(backscatter, persistent, historic) # find thresholds
    ow = fuzzy_zmf(backscatter, ow1, ow2) # produce open water raster
    
    notopen = (backscatter > ow2) & ~persistent
    
    veg = np.zeros_like(backscatter, dtype=np.float32)
    
    for category in np.unique(landcover):
        try:
            subset = (landcover == category)
            
            s = subset & notopen # initially exclude open-water from vegetation analysis
            
            S1, S2, enhance, attenuate, cov = vegetation(backscatter[s], lowlying[s], historic[s])
        
            veg[subset] = fuzzy_smf(backscatter[subset], S1, S2)
            veg[subset & historic] **= enhance
            veg[subset & ~historic] **= attenuate
            
            # mask strong claims if evidence is uncompelling
            if cov > 0.998:
                veg[subset & ~lowlying & (backscatter > S2)] = 0
                
        except Inseparable:
            pass

    return ow + veg
    
