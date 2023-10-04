import os
import copy
import numpy as np
import scipy.io as sio
import latextable
from texttable import Texttable

def add_bold(val):
    return "\\textbf{{{value:.2f}}}".format(value=val)

if __name__ == "__main__":

    print('Make sure you are out of the `wrench` environment and '
            'latextable=1.0.0 is installed.')

    labeled_datasets = True
    # labeled_datasets = False

    if labeled_datasets:
        prefix = 'labeled_'
    else:
        prefix = 'unlabeled_'

    if labeled_datasets:
        # Cancer is Breast Cancer
        datasets = ['Awa', 'Basketball', 'Cancer', 'Cardio', 'Domain',\
                'IMDB', 'OBS', 'SMS', 'Yelp', 'Youtube']
        methods = ['MV', 'Snorkel', 'EBCC', 'HyperLM', 'AMCL\\_CC', 'WMRC'\
                ,'WMRC+MV', 'WMRC unsup', 'WMRC oracle']
    else:
        # crowdsourcing datasets
        datasets = ['bird', 'rte', 'dog', 'web']
        methods = ['MV', 'Snorkel', 'EBCC', 'HyperLM', 'WMRC unsup', 'WMRC oracle']

    # load all the data from the saved files
    results_folder_path = './results'

    bs_ll_table = Texttable()
    err_table = Texttable()
    bs_ll_table_rows = []
    err_table_rows = []

    for rows in [bs_ll_table_rows, err_table_rows]:
        for method in methods:
            rows.append([method])

    if labeled_datasets:
        fn = 'loss_labeled_ttest.mat'
    else:
        fn = 'loss_unlabeled_ttest.mat'

    mdic = sio.loadmat(os.path.join(results_folder_path, fn))
    bs = mdic['brier_score_train']
    ll = mdic['log_loss_train']
    zo = mdic['err_train']
    bs_bold = mdic['brier_score_train_ttest']
    ll_bold = mdic['log_loss_train_ttest']
    zo_bold = mdic['err_train_ttest']


    ### make classification error table
    # add row for WMRC oracle
    zo = np.vstack((zo, mdic['wmrc_oracle_err_train'].squeeze()))
    zo_bold = np.vstack((zo_bold, np.zeros(zo_bold.shape[1])))

    for d_ind, dataset in enumerate(datasets):
        # make 0-1 loss table
        for i, row in enumerate(err_table_rows):
            val = np.round(zo[i, d_ind] * 100, 2)

            # need $ because for some reason latextable will always print 3
            # digits after the . even though the string is xx.xx
            app_value = add_bold(val) if zo_bold[i, d_ind]\
                    else "${value:.2f}$".format(value=val)
            row.append(app_value)

    # for row in err_table_rows:
    #     print(row)

    err_header = copy.deepcopy(datasets)
    err_header.insert(0, 'Method')
    err_table_rows.insert(0, err_header)
    err_table.set_cols_align(["c"]* (1 + len(datasets)))
    err_table.add_rows(err_table_rows)
    err_caption = 'Classification Error'
    err_label = 'tab:' + prefix + 'wrench_classification_error'
    err_table_latex = latextable.draw_latex(err_table, use_booktabs=True, caption=err_caption, label=err_label)
    with open('results/' + prefix + 'classification_error_table.txt', 'w') as f:
        f.write(err_table_latex)
        f.close()


    ### make brier score/log loss table
    # add rows for WMRC oracle
    bs = np.vstack((bs, mdic['wmrc_oracle_brier_score_train'].squeeze()))
    bs_bold = np.vstack((bs_bold, np.zeros(bs_bold.shape[1])))
    # add row for WMRC oracle
    ll = np.vstack((ll, mdic['wmrc_oracle_log_loss_train'].squeeze()))
    ll_bold = np.vstack((ll_bold, np.zeros(bs_bold.shape[1])))

    for d_ind, dataset in enumerate(datasets):
        # make brier score /log loss table
        for i, row in enumerate(bs_ll_table_rows):
            val_bs = np.round(bs[i, d_ind], 2)
            val_ll = np.round(ll[i, d_ind], 2)

            # need $ because for some reason latextable will always print 3
            # digits after the . even though the string is xx.xx
            app_value = add_bold(val_bs) if bs_bold[i, d_ind]\
                    else "${value:.2f}$".format(value=val_bs)
            row.append(app_value)
            app_value = add_bold(val_ll) if ll_bold[i, d_ind]\
                    else "${value:.2f}$".format(value=val_ll)
            row.append(app_value)

    # make sub header which is just 'LL' and 'BS' alternating
    subheader = [""] + ['BS', 'LL'] * len(datasets)
    bs_ll_table_rows.insert(0, subheader)
    # make header
    header = [("Method", 1)]
    for i, dataset in enumerate(datasets):
        header.append((dataset, 2))

    bs_ll_table.set_cols_align(["c"] * (1 + 2 * len(datasets)))
    bs_ll_table.add_rows(bs_ll_table_rows)
    bs_ll_caption = 'Brier Score, Log Loss'
    bs_ll_label = 'tab:' + prefix + 'wrench_brier_score_log_loss'
    bs_ll_table_latex = latextable.draw_latex(bs_ll_table, use_booktabs=True, caption=bs_ll_caption, label=bs_ll_label, multicolumn_header=header)
    with open('results/' + prefix + 'brier_score_log_loss_table.txt', 'w') as f:
        f.write(bs_ll_table_latex)
        f.close()

    if labeled_datasets:
        ### make a table for avg # datapoints used, avg # of classifiers available
        wmrc_stats_table = Texttable()

        # get data
        wmrc_avg_labeled = mdic['wmrc_avg_labeled_pts_per_classifier'].squeeze().tolist()
        wmrc_avg_avail_lf = mdic['wmrc_avg_num_classifier_available'].squeeze().tolist()
        all_avail_lf = mdic['num_classifier_per_dataset'].squeeze().tolist()

        if not isinstance(wmrc_avg_labeled, list):
            wmrc_avg_labeled = [wmrc_avg_labeled]
        if not isinstance(wmrc_avg_avail_lf, list):
            wmrc_avg_avail_lf = [wmrc_avg_avail_lf]
        if not isinstance(all_avail_lf, list):
            all_avail_lf = [all_avail_lf]

        # make row of labeled points used (with percent)
        lab_pts_row = ['Avg.~Labeled Pts per LF']
        # nl for num labeled
        for i, nl in enumerate(wmrc_avg_labeled):
            app_value = "$100$" if nl == 100\
                    else "${value:.2f}$".format(value=nl)
            lab_pts_row.append(app_value)

        # make row for number of available classifiers (with percent)
        n_lf_row = ['Avg.~LFs Available']
        for i, nlf in enumerate(wmrc_avg_avail_lf):
            n_all_lf = all_avail_lf[i]
            app_value = str(nlf) if nlf == n_all_lf\
                    else "${value:.2f} ({pct:.2f}\\%)$".format(value=nlf, pct = 100 * nlf/n_all_lf)
            n_lf_row.append(app_value)

        # make header
        wmrc_stats_header = copy.deepcopy(datasets)
        wmrc_stats_header.insert(0, '')

        wmrc_stats_table_rows = [wmrc_stats_header, lab_pts_row, n_lf_row]
        wmrc_stats_table.set_cols_align(["c"] * (1 + len(datasets)))
        wmrc_stats_table.add_rows(wmrc_stats_table_rows)
        wmrc_stats_caption = 'WMRC Statistics'
        wmrc_stats_label = 'tab:wmrc_stats'
        wmrc_stats_table_latex = latextable.draw_latex(wmrc_stats_table, use_booktabs=True, caption=wmrc_stats_caption, label=wmrc_stats_label)
        with open('results/' + 'wmrc_stats.txt', 'w') as f:
            f.write(wmrc_stats_table_latex)
            f.close()

        ### make a table for fit times
        fit_time_table = Texttable()

        # get data
        wmrc_avg_time = mdic['wmrc_avg_fit_time'].squeeze().tolist()
        amcl_cc_avg_time = mdic['amcl_cc_avg_fit_time'].squeeze().tolist()

        if not isinstance(wmrc_avg_time, list):
            wmrc_avg_time = [wmrc_avg_time]
        if not isinstance(amcl_cc_avg_time, list):
            amcl_cc_avg_time = [amcl_cc_avg_time]

        # make row of labeled points used (with percent)
        wmrc_time_row = ['WMRC']
        for i, wmrc_time in enumerate(wmrc_avg_time):
            app_value = "${value:.2f}$".format(value=wmrc_time)
            wmrc_time_row.append(app_value)

        # make row for number of available classifiers (with percent)
        amcl_cc_time_row = ['AMCL\\_CC']
        for i, amcl_cc_time in enumerate(amcl_cc_avg_time):
            app_value = "${value:.2f}$".format(value=amcl_cc_time)
            amcl_cc_time_row.append(app_value)

        # make header
        fit_time_header = copy.deepcopy(datasets)
        fit_time_header.insert(0, 'Method')

        fit_time_table_rows = [fit_time_header, wmrc_time_row, amcl_cc_time_row]
        fit_time_table.set_cols_align(["c"] * (1 + len(datasets)))
        fit_time_table.add_rows(fit_time_table_rows)
        fit_time_caption = 'Average fit times (s)'
        fit_time_label = 'tab:fit_time_stats'
        fit_time_table_latex = latextable.draw_latex(fit_time_table, use_booktabs=True, caption=fit_time_caption, label=fit_time_label)
        with open('results/' + 'fit_time.txt', 'w') as f:
            f.write(fit_time_table_latex)
            f.close()
        # make rows

