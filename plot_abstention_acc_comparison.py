import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.io import savemat
from matplotlib.collections import LineCollection

def compute_accs_vs_abstention(preds, labels, xent=False):

    n_pts = len(labels.squeeze())

    # record unique probabilities
    lb_probs = []
    # record accuracy for each probability
    accs = []
    # record coverage of all points (in terms of percent).  This is a function
    # of the lower bound for predicted probabilities.
    coverage = []
    # sort the datapoints so ones with highest probs are first.
    sorted_inds = np.argsort(np.max(preds, axis=1))[::-1]

    # get maximum probability of current largest probability at ind
    prob = np.max(preds[sorted_inds[0], :])
    # find the predicted class
    prob_class = int(np.argmax(preds[sorted_inds[0], :]))

    pt_acc = preds[sorted_inds[0], labels[sorted_inds[0]]]
    if xent:
        pt_acc = - np.log(pt_acc)

    lb_probs.append(prob)
    accs.append(pt_acc)
    coverage.append(1 / n_pts)
    pts_parsed = 1

    for ind in sorted_inds[1:]:
        pts_parsed += 1
        prob = np.max(preds[ind, :])
        prob_class = np.argmax(preds[ind, :])
        pt_acc = preds[ind, labels[ind]]
        if xent:
            pt_acc = - np.log(pt_acc)

        new_acc = accs[-1] * ((pts_parsed - 1) / pts_parsed)\
                    + pt_acc / pts_parsed
        new_cov = coverage[-1] + 1 / n_pts

        if prob == lb_probs[-1]:
            accs[-1] = new_acc
            coverage[-1] = new_cov
        else:
            lb_probs.append(prob)
            accs.append(new_acc)
            coverage.append(new_cov)

    # order them so the lower bound probabilities are increasing
    coverage = np.array(coverage)[::-1]
    lb_probs = np.array(lb_probs)[::-1]
    accs = np.array(accs)[::-1]

    return accs, lb_probs, coverage

def compute_plot(preds, labels, ax, color, method_name, xent=False, legend_name=None):

    # compute accs vs probabilities
    accs, lb_probs, coverage = compute_accs_vs_abstention(preds, labels, xent)

    min_acc = np.min(accs)
    max_acc = np.max(accs)

    if xent:
        measure = " Xent"
    else:
        measure = " Accuracy"

    if legend_name is None:
        legend_name = method_name

    # plot as variable width line
    points = np.array([lb_probs, accs]).T.reshape((-1, 1, 2))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, linewidths=coverage * 8,
            color=color,joinstyle="round", capstyle="round",
            label=legend_name, alpha=0.7)
            # label=legend_name + measure + ' (Coverage)')
    ax.add_collection(lc)

    return min_acc, max_acc

if __name__ == "__main__":

    # whether to graph the cross entropies or not
    plot_xent = False
    # plot_xent = True

    if plot_xent:
        prefix = 'xent'
    else:
        prefix = 'preds'

    labeled_datasets = True
    # labeled_datasets = False

    if labeled_datasets:
        datasets = ['aa2', 'basketball', 'breast_cancer', 'cardio', \
                'imdb', 'obs', 'sms', 'yelp', 'youtube']
        methods = ['MV', 'Snorkel', 'EBCC', 'HyperLM', 'AMCL_CC']
    else:
        # crowdsourcing datasets
        datasets = ['bird', 'rte', 'dog', 'web']
        methods = ['MV', 'Snorkel', 'EBCC', 'HyperLM']

    wmrc_names = ['WMRC', 'WMRC + MV', 'WMRC unsup']
    wmrc_folders = ['WMRC/accuracy/', 'WMRC/accuracy+MV/', 'WMRC/accuracy/']
    wmrc_fn_ends = ['_accuracy_semisup_100validlabels_ineqconst_binomial.mat',
                        '_accuracy_semisup_100validlabels_ineqconst_binomial_add_mv.mat',
                        '_accuracy_unsup_trainlabels_ineqconst.mat']
    if not labeled_datasets:
        wmrc_names = [wmrc_names[2]]
        wmrc_folders = [wmrc_folders[2]]
        wmrc_fn_ends = [wmrc_fn_ends[2]]

    # load all the data from the saved files
    results_folder_path = './results'

    # hard code plot settings
    if labeled_datasets:
        # settings = ['-', '-', '-', '-', '-', '-', ':', '--']
        colors = ['c', 'r', 'y', 'm', 'b', 'g', 'chartreuse', 'olive']
    else:
        # settings = ['-', '-', '-', '-', '--']
        colors = ['c', 'r', 'y', 'm', 'b', 'olive']


    for dataset in datasets:
        fig, ax = plt.subplots()
        # fig.dpi = 1200

        dataset_result_path = os.path.join(results_folder_path, dataset)

        setting_ind = 0
        min_acc = 1
        max_acc = 0
        for method in methods:
            method_result_path = os.path.join(dataset_result_path, method)
            method_fn = method + '_' + dataset + '.mat'
            method_fn_full = os.path.join(method_result_path, method_fn)

            # load data
            mdic = sio.loadmat(method_fn_full)

            preds = np.mean(mdic['pred_train'], axis=0)
            labels = mdic['true_labels_train'].squeeze()
            n_class = np.max(labels) + 1

            # argmax accuracy
            # print(np.mean((np.argmax(preds, axis=1) == labels)))
            # expected accuracy
            # print(np.mean(preds[np.arange(preds.shape[0]), labels]))

            color = colors[setting_ind]
            mthd_min, mthd_max = compute_plot(preds, labels, ax, color, method, xent=plot_xent)
            min_acc = min(min_acc, mthd_min)
            max_acc = max(max_acc, mthd_max)

            setting_ind += 1

        # loading is more complex for WMRC
        for w_i, (wmrc_fn_end, wmrc_folder) in enumerate(zip(wmrc_fn_ends, wmrc_folders)):
            wmrc_fn_full = os.path.join(dataset_result_path, wmrc_folder +\
                    'WMRC_' + dataset + wmrc_fn_end)
            mdic = sio.loadmat(wmrc_fn_full)
            preds = mdic['pred_train'][0]
            labels = mdic['true_labels_train'].squeeze()
            # argmax accuracy
            # print(np.mean((np.argmax(preds, axis=1) == labels)))
            # expected accuracy
            # print(np.mean(preds[np.arange(preds.shape[0]), labels]))

            # manipulate confidences and plot it
            color = colors[setting_ind]
            mthd_min, mthd_max = compute_plot(preds, labels, ax, color, wmrc_folder, xent=plot_xent, legend_name=wmrc_names[w_i])
            min_acc = min(min_acc, mthd_min)
            max_acc = max(max_acc, mthd_max)

            if w_i == 0 and labeled_datasets:
                ### plot the confidences
                cis = mdic[prefix + '_cis_pred_prob_lb_comb_class_train']
                selections = mdic['groupings_pred_prob_lb_comb_class']
                # compute lowest probability for each group
                m=selections.shape[0]
                lb_probs = []
                for i in range(m):
                    s_b = selections[i, :].astype(bool)
                    # sel_preds = preds[s_b, np.arange(n_class)]
                    sel_preds = preds[s_b, :]
                    min_max_pred = np.min(np.max(sel_preds, axis=1))
                    lb_probs.append(min_max_pred)
                lb_probs = np.array(lb_probs)

                # plot it now
                ax.fill_between(lb_probs, cis[0, :], cis[1, :], color='g',\
                        alpha=0.3, label=wmrc_names[w_i] + ' confidence')

            setting_ind += 1

        # plot calibration
        plot_fn = os.path.join(dataset_result_path,\
                dataset + '_' + 'acc_vs_abstention_' + prefix + '.pdf')

        # ax.legend(bbox_to_anchor=(1.55, 1))
        ax.legend(loc='lower right')
        ax.set_xlim(1/n_class, 1)
        min_acc = max(0, min_acc * 0.95)
        max_acc = min(1, max_acc * 1.05)
        ax.set_ylim(min_acc, max_acc)

        ax.set_xlabel('Lower Bound on Predicted Probability')
        if plot_xent:
            measure = 'Cross Entropy'
        else:
            measure = 'Expected Accuracy'

        ax.set_ylabel(measure)

        fig.savefig(plot_fn, bbox_inches='tight', format='pdf')
        plt.close(fig)

