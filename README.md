# Code for Shopper


The code in this repo implements Shopper, the model in [the paper](https://arxiv.org/abs/1711.03560 "Arxiv paper"):

+ Francisco J. R. Ruiz, Susan Athey, David M. Blei. *SHOPPER: A Probabilistic Model of Consumer Choice with Substitutes and Complements*. ArXiv 1711.03560. 2017.

Please cite this paper if you use this software.


## Introduction


Shopper is a model of consumer choice that posits a mean utility for each basket `t` and item `c`,
```math
Psi_{tci} = psi_{tc} + rho_c*sum_j[alpha_{y_{tj}}]/(i-1)
```
where the baseline `psi_{tc}` is
```math
psi_{tc} = lambda_c + theta_{u_t}*alpha_c - gamma_{u_t}*beta_c*log(r_{tc}) + delta_{w_t}*mu_c
```

For details on the model, please see the paper.

The **code** is implemented in C++/CUDA and it requires a GPU to run. It also uses multithreading to speed up computations.


## Compilation


You can compile the code from a terminal using this line:
```
nvcc -std=c++11 -o shopper src/emb.cu `gsl-config --cflags --libs`
```

**Requirements.**

+ GCC 5.0 or earlier. (It has been tested with GCC 4.8.5.)

+ [CUDA](https://developer.nvidia.com/cuda-downloads) 8.0.

+ The [GSL library](https://www.gnu.org/software/gsl).

+ The [CTPL](https://github.com/vit-vit/CTPL/blob/master/ctpl_stl.h) thread pool library by Vitaliy Vitsentiy. It is included in this repo, so you do not need to download anything.


## Running the code


You can run the code from a terminal with the line:
```
/your_path/shopper [options]
```

Just replace `/your_path` with the actual path to the compiled file, and replace `[options]` with a list of the options you would like to run the code with.


## Options

Here we describe the full list of options (warning: it is a long list!). For a description of the data format or output files, see the next sections.

**Options to configure the input/output.**

```
-datadir <string>      path to directory with all the data files (see format below)
                       [default: .]

-outdir <string>       path to directory where output files will be written
                       [default: .]

-noTest                do not compute the performance on the test data
                       [default: disabled ---compute the performance on test]

-noVal                 do not compute the performance on the validation data
                       [default: disabled ---compute the performance on validation]

-keepOnly <int>        ignore the less frequent items (keep only the top N items).
                       If <=0, keep all items;
                       if >0, keep only this number of items
                       [default: -1]

-keepAbove <int>       ignore the less frequent items (ignore items with frequency smaller than the specified integer).
                       If <=0, keep all items;
                       if >0, keep only items with frequency lower than the specified integer
                       [default: -1]
```

**Options to configure the model.**

```
-K <int>               number of latent factors for alpha_c, rho_c, theta_u
                       [default: 50]

-userVec <int>         use per-user vectors theta_u?
                       If 0, no per-user vectors are included;
                       if 1, the per-user vectors are added to the context and interact with rho_c;
                       if 3, the per-user vectors interact with alpha_c;
                       other values cause an error
                       [default: 3]

-itemIntercept <int>   use per-item intercepts lambda_c?
                       If 0, do not include per-item intercepts;
                       if 1, include per-item intercepts
                       [default: 1]

-price <int>           length of the price vectors gamma_u and beta_c.
                       If 0, do not include price effects
                       [default: 10]

-days <int>            length of the seasonal effect vectors delta_w and mu_c.
                       If 0, do not include seasonal effects
                       [default: 10]

-lookahead             use the look-ahead procedure
                       [default: disabled ---use the model without look-ahead]
```

**Other options to configure the model.**

```
-normPrice <int>       normalize the prices r_{tc}?
                       If 0, do not normalize the prices;
                       if 1, normalize by their average value;
                       if 2, normalize by their minimum value.
                       Relevant only if '-price' is >=1
                       [default: 1]

-noAvgContext          do not divide by (i-1) in order to average the alpha_c vectors of the items in the context
                       [default: disabled ---the (i-1) term is included]

-noCheckout            do not include a checkout item
                       [default: disabled ---the checkout item is considered]

-likelihood <int>      type of likelihood used.
                       If 0, use a Bernoulli likekihood, similar to Bernoulli embeddings by Rudolph et al.;
                       if 1, use a softmax likelihood and the one-vs-each bound by Titsias;
                       if 3, use a within-group softmax likelihood (i.e., probability of an item conditioned on its group);
                       if 4, use a softmax likelihood
                       [default: 1]

-negsamples <int>      number of negative examples used for the one-vs-each bound.
                       Relevant only if '-likelihood 1'
                       [default: 50]

-zeroFactor <double>   weight applied to the negative examples for the Bernoulli model.
                       Relevant only if '-likelihood 0'
                       [default: 0.1]

-nsFreq <int>          choose the distribution from which to draw negative examples.
                       If -1, use the uniform distribution;
                       if 0, use the unigram distribution (based on the item frequencies);
                       if 1, use a smoothed unigram distribution (unigram raised to the power of 0.75);
                       if >=2, use a distribution that is biased to upweight items in the same group as the target item (in proportion N:1)
                       [default: -1]

-shuffle <int>         shuffle the items in each basket at each iteration?
                       If 0, do not shuffle the items (they're considered in the same order in which they appear in the data files);
                       if 1, randomly shuffle the items at each iteration;
                       if >1, randomly shuffle the items at each iteration and consider that many random permutations
                       [default: 1]

-symmetricRho          set rho_c=alpha_c
                       [default: disabled ---rho_c and alpha_c are different]
```

**Options to configure the optimization algorithm.**

```
-max-iterations <int>  maximum number of iterations of the variational inference procedure
                       [default: 12000]

-batchsize <int>       number of baskets to be considered at each iteration of the SVI algorithm.
                       If <=0, use all baskets instead of minibatches;
                       if >0, set the minibatch size to that value
                       [default: 1000]

-eta <double>          learning rate.
                       IMPORTANT: The optimization can be very sensitive to this parameter
                       [default: 0.01]

-step_schedule <int>   select the stepsize schedule for the SVI algorithm.
                       If 1, use RMSProp;
                       if 2, use Adagrad
                       [default: 1]
```

**Other configuration options.**

```
-threads <int>         number of threads to be used to parallelize some of the computations
                       IMPORTANT: It is strongly recommended to use as many threads as possible to speed-up computations
                       NOTE: You should only use as many threads as available CPU cores in your machine
                       [default: 1]

-seed <int>            random seed used
                       [default: 0]

-rfreq <int>           evaluate test and validation performance every this number of iterations
                       IMPORTANT: Depending on the size of the test/validation files, this may be an expensive operation.
                       Make sure that you set this value high enough to avoid computation bottlenecks
                       [default: 1000]

-saveCycle <int>       saves the variational parameters to a file every this number of iterations
                       IMPORTANT: The files might be big in size. Make sure that you have enough disk space
                       [default: 5000]

-label <string>        append this label to the output folder name
                       [example: '-label crazy-experiment']
                       [default: empty ---do not append any label]
```

**Hyperparameter options.**

```
-s2rho <double>        prior variance over rho_c
                       [default: 1]

-s2alpha <double>      prior variance over alpha_c
                       [default: 1]

-s2theta <double>      prior variance over theta_u
                       [default: 1]

-s2lambda <double>     prior variance over lambda_c
                       [default: 1]

-s2delta <double>      prior variance over delta_w
                       [default: 0.01]
                       
-s2mu <double>         prior variance over mu_c
                       [default: 0.01]

-rtegamma <double>     prior rate over gamma_u
                       [default: 1000]

-shpgamma <double>     prior shape over gamma_u
                       [default: 100]

-rtebeta <double>      prior rate over beta_c
                       [default: 1000]

-shpbeta <double>      prior shape over beta_c
                       [default: 100]
```



## Data Format


The input data must be contained in the folder specified with `-datadir <string>`. It must be a collection of `.tsv` files (tab-separated values). 

IMPORTANT: The column separators **must be tabs**; not spaces.

**Training/Test/Validation.**

The training, test, and validation files have all the same format. They must be named `train.tsv`, `test.tsv`, and `validation.tsv`, respectively. Each of these files must have four columns and no header lines.

The columns indicate (in this order): `user_id`, `item_id`, `session_id`, `quantity`. The ids are all non-negative integers that are unique for each user, item, and session. The quantity must be any positive integer; indeed the integer is ignored and it is recommended to set the fourth column to 1 for all rows.

Here is an example of a valid input file (e.g., `train.tsv`):
```
0    10    20    1
0    11    20    1
0    10    30    1
0    20    30    1
0    21    30    1
1    10    20    1
1    11    20    1
1    20    20    1
```
This file indicates that there are two users (with ids 0 and 1) and four items (with ids 10, 11, 20, 21). User 0 makes two shopping trips: in session 20, she purchases items 10 and 11; in session 30 she purchases items 10, 20, and 21. User 1 makes one single shopping trip, corresponding to session 20.

Sessions allow to indicate multiple shopping trips for each user. In addition, each session corresponds to a specific date and items price configuration. In the example above, users 0 and 1 share session 20, which means that they share the date and see the same item prices in that session. If the session of user 1 were 40 instead, then it could correspond to a different date.

IMPORTANT: The test and validation files are expected to be of different nature. In particular, the shopping trips in `test.tsv` are independent of `train.tsv` even when the session ids are shared. In contrast, the lines of `validation.tsv` are expected to be missing items corresponding to some of the shopping trips in the training set.

**Prices.**

If `-price <int>` is specified with a positive integer, the file `item_sess_price.tsv` is required. It must specify the price of all items for each session.

The columns of `item_sess_price.tsv` are: `item_id`, `session_id`, `price`. Here is an example of a valid file:
```
10    20    1.2
11    20    1.1
20    20    0.9
21    20    1.0
10    30    0.7
11    30    1.5
20    30    1.0
21    30    1.0
```

The total number of lines is equal to the total number of items multiplied by the total number of sessions.

**Seasons.**

If `-days <int>` is specified with a positive integer, the file `sess_days.tsv` is required. It specifies the week id for each session.

The columns of `sess_days.tsv` are: `session_id`, `week_id`, `dow_id`, `hour_id`. The two last columns correspond to day-of-week id and hour id, but they are internally ignored, so they can take any value. The week id is an identifier of the week in which the shopping trip happens. Here's an example:
```
20    1    1    1
30    2    1    1
```

**Item groups.**

The price parameters `beta_c` and seasonal parameters `mu_c` are shared across all items in the same group.

If `-price <int>` or `-days <int>` is specified with a positive integer, the file `itemGroup.tsv` is required. It specifies the group to which each item belongs to.

The columns of `itemGroup.tsv` are: `item_id`, `group_id`. Here's an example:
```
10    10
11    11
20    20
21    21
```
In this example, we have assigned each item to its own group, so no parameter sharing is actually performed.

This other example creates two groups of items (with ids 1 and 2):
```
10    1
11    1
20    2
21    2
```

The file `itemGroup.tsv` is also required if `-likelihood 3` or `-nsFreq <int>` is specified (with an integer >=2).


## Output


The code creates the output files within the directory specified with `-outdir <string>`.

**Variational parameters.**

The files `param_XXX_YYY.tsv` contain the variational parameters corresponding to the latent variables (`alpha_c`, `rho_c`, etc.). For example, `param_alpha_mean.tsv` contains the variational mean and `param_alpha_std.tsv` contains the variational standard deviation corresponding to `alpha_c`.

The columns of these files indicate: `line_number`, `item_id` (or `group_id`), `value1`, `value2`, `value3`, `...`

For the Gaussian factors, there are two files for each variable, with the mean and standard deviation (ending in `_mean.tsv` and `_std.tsv`). For the gamma factors, there will be three files, with the mean, shape, and rate (ending in `_mean.tsv`, `_shp.tsv`, and `_rte.tsv`).

**Test/Validation log-likelihood.**

+ `test.tsv` and `validation.tsv` contain the test and validation log-likelihood, computed every `-rfreq <int>` iterations (and also at the end of the optimization). The columns indicate: `iteration`, `duration`, `average log-lik`, `valid shopping trips`, `valid lines`.

+ `test_all.tsv` and `validation_all.tsv` are created at the end of the optimization. They contain the log-likelihood for each individual line of the input files `test.tsv` and `validation.tsv`. The lines for which the log-likelihood is 0 correspond to *non-valid lines*.

  The log-likelihood is computed excluding the checkout item from the choice set. For each line of the test file, it is assumed that the items that are currently in the shopping cart are the other items corresponding to the same user and session in the *test* data. For each line of the validation file, it is assumed that the items that are currently in the shopping cart are the other items corresponding to the same user and session in the *training* data. 

+ `test_baskets.tsv` and `test_baskets_noChkout.tsv` contain the average log-likelihood of baskets in the test set (with or without the checkout item), assuming the order in which items are specified in the input file `test.tsv`. The log-likelihood is averaged over baskets; *not* over items. The columns indicate: `iteration`, `duration`, `average log-lik`, `valid shopping trips`, `valid lines`.

+ `test_baskets_all.tsv` and `test_baskets_all_noChkout.tsv` contain the log-likelihood of baskets in the test set (with or without the checkout item), assuming the order in which items are specified in the input file `test.tsv`. Each line corresponds to the same line of the input file `test.tsv`. The lines for which the log-likelihood is 0 correspond to *non-valid lines*.

**Other files.**

+ `telapsed.tsv`: Contains the wall-clock time (in seconds) for each iteration.

+ `obj_function.tsv`: Contains the ELBO for each iteration.

+ `mean_prices.tsv` and `min_prices.tsv`: Contain the average and minimum price for each item across all sessions.

+ `vocab.txt`: Contains the frequency (number of occurrences) of each item.

+ `log.txt`: Contains a list with all the configuration parameters.


## Memory Considerations


The code has been specifically designed for the datasets described in the paper. To accelerate computations, it computes the inner products `theta_u*alpha_c` and `rho_c*alpha_{c'}` for all user/item pairs and for all item/item pairs. Therefore, this implementation of Shopper requires storage capacity for the resulting user/item and item/item matrices.

Depending on your system specifications (and the GPU memory), you may run out of memory when the product `number_users x number_items` or `number_items x number_items` is above 10^9 or 10^10. In such case, try reducing the dataset size (e.g., by grouping items according to their category, or by removing low-frequency items or users).
