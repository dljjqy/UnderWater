# Predict one day water parameters by three days

# Main idea: Use as more data as possible.
1. How use the multi multidimensional time series $X_i^j$ besides the time dimension?
(Index i is the time dimensional and j is the j-th multi-dimensional time series,$X_i^j \in \mathbf{R}^{n}$).
    1. The resolutions of the different data on time are not same.
        * Downsample the higher resolution.
        * Upsamle the lower resolution.
        *
        * What else?

    2. After solving the resolution problem.Do we should merge the $X^j$ together?If we should,How?
        * Treat j-th seperatly.
        * Merge together.First, we should decide which $X^j$ are related based on the map and concatenate all the related data together treating them as a time serise belong to $R^{\sum n_{i}}$.
        
2. How treat the time series? 

# TODO LIST:
1. Model part:
    1. SCINet (Working On)
    2. L-Net
    3. Transformers types.

2. Loss function part:
    1. DTW and soft DTW, DIlate

3. Train part
    1. Decide the window size.
    2. Test all the possible model.
