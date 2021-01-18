# ss-nmf-CC
semi-supervised nonnegative matrix factorization with convex coefficients

### Requirements ###
* Python 3.x
* cvxopt >= 1.2.5
* numpy >= 1.19.5
* pandas >= 1.2.0
* scipy >=1.6.0
* six >=1.15.0 



### Required Arguments ###
| <!-- -->    | <!-- -->    |
------------- | -------------
**V**  | Path to .tsv or .csv file for V matrix. Assumes no header exists. Matrix of M features (e.g. CpG sites) over N raw samples.


### Optional arguments ###
| <!-- -->    | <!-- -->    |
------------- | -------------
**-W**  | Path to .tsv or .csv file for W matrix. Can be partial. Assumes no header exists. Atlas matrix of M features over K bases (e.g. tissues).
**-H**  | Path to .tsv or .csv file for H matrix. Assumes no header exists. Coefficients matrix of K bases over N samples such that each column takes the convex combination of a single sample.
**-c, --free_w_cols**  | Number of free columns to add to W. Default: 0
**-iw, --init_w**  | Comma separated string, stating the type of distribution (first argument) and parameters (second and so on) for W initialization. default="normal,0,1". Valid distribution values: normal,beta.
**-ih, --init_h**  | Comma separated string, stating the type of distribution (first argument) and parameters (second and so on) for H initialization. default="beta,70,100". Valid distribution values: normal,beta.
**-t, --iter_num**  | Number of iteration for the algorithm. Default: 10
**-r, --reps**  | Number of Repetitions for the algorithm. Default: 5 
**-o, --tol**  | Tolerance Parameter. Default: 1e-5

### Usage Examples###

`python ssnmfcc.py V.tsv -c 3`\
`python ssnmfcc.py V.tsv -W W.tsv`\
`python ssnmfcc.py V.tsv -W W.tsv -c 3  `\
`python ssnmfcc.py V.tsv -H H.tsv -r 4 --init_w "beta,30,70"`

