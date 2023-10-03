1. create and activate conda environment WITHOUT using the environment.yml file
    conda create --name wrench python=3.6
    conda activate wrench
2. install wrench
    pip install ws-benchmark==1.1.2rc0
3. install other dependencies
    pip install -r requirements.txt


### WMRC
    0. (Optional) generate synthetic data
        python3 generate_synthetic.py
    1. write the settings for all datasets
        python3 write_wmrc_settings.py
    2. run wmrc
        python3 run_wmrc.py

    things to note

### AMCL_CC, MV, Snorkel, EBCC

### plotting misc figures, aggregating results, generating tables

    - calibration
        python3 plot_calibration.py
    - absention vs accuracy (thickness of line is coverage)
        python3 plot_abstention_vs_accuracy.py
    - wmrc accuracy trend vs labeled data
        python3 plot_wmrc_labeled_data_trend.py

    - figure out significance of results via t-test/aggregate results. must
      run this before making tables.
        python3 result_t_test.py

    - generate all tables, brier score/log loss combined, classification error,
      fit times of WMRC vs AMCL_CC.  must use latextable version 1.0.0 to get
      multicolumn tables.  however, python 3.6 is too old for that library

        conda deactivate wrench
        pip install latextable==1.0.0
        python3 make_loss_tables.py


### misc information
    talk about dataset format
    talk about formats of main variables used
    talk about the general control flow
