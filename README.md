This is a fork of the [wrench](https://github.com/JieyuZ2/wrench) repository that focuses on aggregating the provided labeling functions for certain datasets.
Notably, two label models are added, WMRC is our own implemenation and [AMCL\_CC](https://github.com/BatsResearch/amcl/tree/main) is ported, the CVXPY implementation in specific.

This repository also contains code for computing confidences using WMRC's uncertainty set. Code to visualize the confidences has also been provided. 
Also included is a means of computing confidences from EBCC.  Visualization code is also included.

Scripts have been written to automatically run and record the results (0-1 Loss, Brier Score, Log Loss) of multiple methods on select datasets (more than once if the method has randomness).
Further, visualizations of calibration, abstention vs accuracy, and how the performance of WMRC changes as a function of labeled data are provided.

Finally, code that creates a latex table based on the results of the methods on the datasets has also been included.
That code also generates tables showing statistics about the ensemble WMRC used, as well as a comparison of `fit` times between WMRC and AMCL\_CC.

More detailed explanation is given below, along with installation instructions.

## Installation
1. create and activate conda environment WITHOUT using the `environment.yml` file

    `conda create --name wrench python=3.6`

    `conda activate wrench`
2. install wrench

    `pip install ws-benchmark==1.1.2rc0`
3. install other dependencies

    `pip install -r requirements.txt`

## Running the methods
We have provided files `run_(wmrc|amcl_cc|mv|snorkel|ebcc|hyperlm).py` that will automatically run the named method on the following datasets (see `README_wrench.md` for links/sources):

- Animals with Attributes
- Basketball
- Breast Cancer
- Cardiotocography
- DomainNet
- IMDB
- OBS
- SMS
- Yelp
- Youtube

Majority Vote (MV) and Hyper Label Model (HyperLM) only run once because they are deterministic.  The other methods, WMRC, Snorkel, EBCC, AMCL\_CC all run 10 times on each dataset by default.
The 0-1 Loss, Brier Score, and Log Loss are all recorded, along with other information in the `*.mat` file.
A `results` folder is automatically created along with a folder for each dataset.
For each dataset, a folder will be created for each method.
That is where the `*.log` and `*.mat` files can be found.
Below is extra information about changeable settings/other instructions to run the method.

### WMRC

WMRC relies on many hyperparameters, which have already been set in `write_wmrc_settings.py`.
We have also included code that can be used to generate synthetic data essentially in the Dawid-Skene sense.
If wanted, ten synthetic  datasets will be generated -- they are meant to illustrate the confidences one can get from WMRC.
Note that by default, a lot of confidence intervals will be computed and plotted.
This can be time consuming, but can be turned off by changing the `get_confidence` entries of the `unsup_kwargs` and `oracle_kwargs` dictionaries in `write_wmrc_settings.py`.
Also, note that the `fit` method is timed via `perf_counter`, so running multiple methods in parallel can affect the recorded time.

One can switch between running WMRC on synthetic or real datasets by switching the value of `use_synthetic`.
We have also implemented an option to replot the figures from the saved data -- just change the value of `replot_figs`.
Although stuff will be printed as output, none of the original results will be overwritten.

0. (Optional) Generate synthetic data.

    `python3 generate_synthetic.py`
1. Write the settings for all datasets.

    `python3 write_wmrc_settings.py`
2. Run WMRC.

    `python3 run_wmrc.py`

### EBCC
We have also implemented a method of getting confidences from EBCC.  If one does not want that, they can change the default value of the `get_confidences` in the `run_ebcc` function declaration.
Similar to WMRC, there is the option to replot figures by setting `replot_figs`.
Note that this may take some time due to the width of the histogram bars being computed on the fly.
Otherwise,

`python3 run_ebcc.py`.

### AMCL\_CC, MV, Snorkel, HyperLM
Running the respective `python3 run_(amcl_cc|mv|snorkel|hyperlm).py` suffices.

## Result Visualizations and Table Generation
We have also included some code to visualize the predictions of each method (for each dataset in their respective folder) along with code that automatically generates latex tables containing all results (in `results` folder).

- Calibration

    For datasets with two class labels (i.e. everything except DomainNet), run the following to show the calibration graph.
    Note that we take the average prediction from all 10 runs from random methods and then plot the calibration results of that average prediction.

    `python3 plot_calibration.py`

    If one wants the calibration for individual ruls from methods run 10 times, then they either need to change how the calibration results are saved in each run_* file.
    Essentially, one would need to save the result of each run as its own variable.
    Or, the calibration can be computed in `plot_calibration.py` by using the stored predictions.

- Absention vs Accuracy

    This creates a plot that shows how accurate the predictions are on a subset of datapoints.
    The subset is determined by the maximum prediction probability -- for example, we take predictions that have at least 0.9 in a class and see what the accuracy on that is.
    For WMRC, the computed confidence interval (for certain sets of predictions is shown.)
    The default setting shows the accuracy of the predictions, but the cross entropy can be shown by changing the `plot_xent` variable's value.

    `python3 plot_abstention_vs_accuracy.py`

- WMRC Accuracy Trend vs Number of Labeled Data

    A plot to show how the error rate of WMRC and WMRC + MV constraint decreases as more labeled data is used to estimate the accuracies/class frequencies.

    `python3 plot_wmrc_labeled_data_trend.py`

- Result t-test and Aggregation

    Performs a two sided t-test on the error rates (each of the three losses).
    Also aggregates data from results for each dataset. 
    This must be run before making the loss tables (below).

    `python3 result_t_test.py`

- Table Generation

    Lastly, we provide a script to generate latex tables of 0-1 loss, Brier Score/Log Loss (these are combined), and fit times of WMRC and AMCL\_CC.
  Note that you must use latextable version 1.0.0 to get multicolumn tables.
  That is needed to create the Brier Score/Log Loss table. 
  However, Python 3.6 (which is required for the `wrench` package) is too old for that version, so one must use a newer version of Python, 3.9+ should work.

    `conda deactivate wrench`

    `pip install latextable==1.0.0`

    `python3 make_loss_tables.py`


## Miscellaneous Information
We store a simplified version of `wrench` datasets as `.mat` files.
They contain train, validation, and possibly test labels and labeling function predictions.
