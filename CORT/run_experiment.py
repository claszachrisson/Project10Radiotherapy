import fire

from time import time
from os.path import exists

import numpy as np
import scipy as sp


import CORT
import config
import utils

################################################################################


def main(case='Prostate', loss='squared', score_method='gradnorm', m=25000, repetitions=1, w_BDY_over=1.0, w_OAR_over=1.0, w_PTV_over=4096.0, w_PTV_under=4096.0):

    """
    case = 'Prostate'
    case = 'Liver'
    case = 'HeadAndNeck'

    loss = 'absolute'
    loss = 'squared'

    score_method = 'gradnorm'
    m = 25000
    repetitions = 1
    w_BDY_over = 1.0
    w_OAR_over = 1.0
    w_PTV_over = 4096.0
    w_PTV_under = 4096.0
    """


    if loss not in ['absolute','squared']:
        exit(f'ERROR: loss is {loss} but it should be either absolute, or squared')

    if case not in ['Prostate','Liver','HeadAndNeck']:
        exit(f'ERROR: case is {case} but it should be either Prostate, Liver, or HeadAndNeck')

    if score_method not in ['full', 'uniform', 'gradnorm', 'residual']:
        # full is a dummy and is not used, m=0 indicates full
        exit(f'ERROR: score_method is {score_method} but it should be either uniform, gradnorm, or residual')


    m = int(m)
    repetitions = int(repetitions)
    w_BDY_over = float(w_BDY_over)
    w_OAR_over = float(w_OAR_over)
    w_PTV_over = float(w_PTV_over)
    w_PTV_under = float(w_PTV_under)

    # file name format is ...
    filename = f'result_{case}_{score_method}_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'
    if loss == 'squared':
        filename = f'result_{case}_{loss}_{score_method}_{m}_{repetitions}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.npz'

    # dont run the script if results file already exists!
    if exists(filename):
        print(f'ABORT: experiment {filename} already exists!')
        exit()

    cfg = config.get_config(case)
    data_path, gantry_angles, couch_angles, OBJ, PTV_structure, PTV_dose, BODY_structure, BDY_threshold, OAR_structures, OAR_threshold, eta, steps = cfg

    configuraton = {
        'case':case,
        'loss':loss,
        'score_method':score_method,
        'm':m,
        'repetitions':repetitions,
        'w_BDY_over':w_BDY_over,
        'w_OAR_over':w_OAR_over,
        'w_PTV_over':w_PTV_over,
        'w_PTV_under':w_PTV_under,
        'start_time':time()
    }
    print('configuraton:', configuraton)
    # save at least the config such that the file exists and no parallel run of
    # the same experiment can happen
    np.savez(filename, configuraton=configuraton)
 

    # load full dose influence matrix
    D_full = CORT.load_data(data_path, OBJ, list(zip(gantry_angles, couch_angles)))


    # set the indices for body (BDY), OAR, and PTV
    BDY_indices = OBJ[BODY_structure]['IDX']
    PTV_indices = OBJ[PTV_structure]['IDX']
    OAR_indices = np.unique(np.hstack([OBJ[OAR_structure]['IDX'] for OAR_structure in OAR_structures]))
    # fix the indices
    OAR_indices = np.setdiff1d(OAR_indices, PTV_indices)
    BDY_indices = np.setdiff1d(BDY_indices, np.union1d(PTV_indices, OAR_indices))

    assert len(np.intersect1d(BDY_indices, PTV_indices)) == 0
    assert len(np.intersect1d(OAR_indices, PTV_indices)) == 0
    assert len(np.intersect1d(OAR_indices, BDY_indices)) == 0

    n_BDY = len(BDY_indices)
    n_OAR = len(OAR_indices)
    n_PTV = len(PTV_indices)

    # specify the target dose
    # initialize the target dose to zero
    target_dose = np.zeros(D_full.shape[0])
    # set the PTV dose
    target_dose[OBJ[PTV_structure]['IDX']] = PTV_dose
    # set the OAR target dose to a threshold to prevent penalizing small violations
    target_dose[OAR_indices] = OAR_threshold
    # set the BDY target dose to a threshold to prevent penalizing small violations
    target_dose[BDY_indices] = BDY_threshold


    # set D and overwrite target_dose to only consider BODY, OAR, and PTV voxels,
    # i.e., skip all other voxels outside the actual BODY
    D = sp.sparse.vstack((D_full[BDY_indices],
                          D_full[OAR_indices],
                          D_full[PTV_indices]))
    target_dose = np.hstack((target_dose[BDY_indices],
                             target_dose[OAR_indices],
                             target_dose[PTV_indices]))


    D_BDY = D[:n_BDY]
    D_OAR = D[n_BDY:(n_BDY+n_OAR)]
    D_PTV = D[(n_BDY+n_OAR):]
    target_dose_PTV = target_dose[(n_BDY+n_OAR):]


    def identity(z):
        return z

    modifier = identity
    if loss == 'squared':
        modifier = np.square


################################################################################

    # prepare the weights
    weights_BDY_over  = np.ones(D_BDY.shape[0]) * (w_BDY_over  /(n_BDY*modifier(BDY_threshold)))
    weights_OAR_over  = np.ones(D_OAR.shape[0]) * (w_OAR_over  /(n_OAR*modifier(OAR_threshold)))
    weights_PTV_over  = np.ones(D_PTV.shape[0]) * (w_PTV_over  /(n_PTV*modifier(PTV_dose)))
    weights_PTV_under = np.ones(D_PTV.shape[0]) * (w_PTV_under /(n_PTV*modifier(PTV_dose)))

    def obj(x):
        return utils.objective(D_BDY, D_OAR, D_PTV, BDY_threshold, OAR_threshold,
                               target_dose_PTV, weights_BDY_over, weights_OAR_over,
                               weights_PTV_over, weights_PTV_under, x, loss=loss)

################################################################################

    if m==0:

        # solve the full problem using MOSEK
        sol_MOSEK, time_MOSEK = utils.compute_radiation_plan(D_BDY, D_OAR, D_PTV, BDY_threshold, OAR_threshold,
                                                             target_dose_PTV, weights_BDY_over, weights_OAR_over,
                                                             weights_PTV_over, weights_PTV_under,
                                                             loss=loss, verbose=True)
        obj_MOSEK = obj(sol_MOSEK)

        # save
        np.savez(filename, configuraton=configuraton,
                 sol_MOSEK=sol_MOSEK, time_MOSEK=time_MOSEK, obj_MOSEK=obj_MOSEK)

        exit('full done')

################################################################################

    """
    utils.objective(D_BDY, D_OAR, D_PTV, BDY_threshold, OAR_threshold,
                    target_dose_PTV, weights_BDY_over, weights_OAR_over,
                    weights_PTV_over, weights_PTV_under, sol_MOSEK, loss=loss)
    """


    # create a new OBJ object to use for the DVH plots
    OBJ_DVH = {}
    # inlude the PTV
    OBJ_DVH[PTV_structure] = dict(OBJ[PTV_structure])
    for structure in OAR_structures+[BODY_structure]:
        # include the relevant structures
        OBJ_DVH[structure] = dict(OBJ[structure])
        # subtract PTV voxels
        OBJ_DVH[structure]['IDX'] = np.setdiff1d(OBJ[structure]['IDX'], OBJ[PTV_structure]['IDX'])

    """
    utils.plot_DVH(D_full, sol_MOSEK, OBJ, PTV_dose)
    utils.plot_DVH(D_full, sol_MOSEK, OBJ_DVH, PTV_dose)
    """

################################################################################

    #
    # Compute Scores
    #

    # set the weights for the probing/surrogate function
    weights = np.zeros(D.shape[0])
    weights[:n_BDY] =              1.0/n_BDY
    weights[n_BDY:(n_BDY+n_OAR)] = 1.0/n_OAR
    weights[(n_BDY+n_OAR):] =      1.0/n_PTV

    # run projected gradient descent
    res = utils.compute_scores(D, target_dose, weights, eta, steps)

    # res = utils.compute_scores(sp.sparse.diags(np.sqrt(weights)) * D,
    #                            np.sqrt(weights) * target_dose,
    #                            np.ones_like(weights), eta, steps)

    # res = utils.compute_scores(D, target_dose, weights, 3.743, steps)

    x_hist, loss_hist, score_residual_hist, score_gradnorm_hist, time_PGD = res

    np.savez(filename, configuraton=configuraton,
             x_hist=x_hist, loss_hist=loss_hist,
             score_residual_hist=score_residual_hist,
             score_gradnorm_hist=score_gradnorm_hist,
             time_PGD=time_PGD)

################################################################################

    """
    for i in range(steps):
        utils.plot_DVH(D_full, sol_MOSEK, OBJ_DVH, PTV_dose,
                       dashed_x=x_hist[i], dashed_legend=f'PGD Step{i+1}',
                       title=f'result_{case}_{loss}_pgd_{eta}_{steps}_DVH_Step{i+1}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}',
                       save_location=f'result_{case}_{loss}_pgd_{eta}_{steps}_DVH_Step{i+1}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}')
    """
###############################################################################
    """
    import matplotlib.pylab as plt
    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(np.arange(1,steps+1), np.array(loss_hist)/D.shape[0], 'bo-', label='surrogate loss')
    ax.legend(loc=6)
    ax.set_xticks(np.arange(1,steps+1,3))
    ax.set_xlabel('Projected Gradient Descent Iterations')
    ax.set_ylabel('Squared Loss', c='b')
    ax2 = ax.twinx()
    ax2.plot(np.arange(1,steps+1), [obj(x) for x in x_hist], 'go-', label='weighted objective')
    ax2.axhline(obj(sol_MOSEK), c='g', ls='--', label='full solution using MOSEK')
    ax2.set_ylabel('Weighted Objective', c='g')
    ax2.legend(loc=5)

    fig.savefig(f'result_{case}_{loss}_pgd_{eta}_{steps}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.pdf', dpi=300, transparent=True, bbox_inches='tight')
    fig.savefig(f'result_{case}_{loss}_pgd_{eta}_{steps}_{w_BDY_over}_{w_OAR_over}_{w_PTV_over}_{w_PTV_under}.png', dpi=300, transparent=False, bbox_inches='tight')
    """

################################################################################

    #
    # subset
    #

    res_subset_x = []
    res_subset_time = []
    res_subset_obj = []

    for r in range(repetitions):
        # set the weights
        weights = np.zeros(D.shape[0])
        weights[:n_BDY] =              1.0/(n_BDY*modifier(BDY_threshold))
        weights[n_BDY:(n_BDY+n_OAR)] = 1.0/(n_OAR*modifier(OAR_threshold))
        weights[(n_BDY+n_OAR):] =      1.0/(n_PTV*modifier(PTV_dose))

        # set the scores
        # default is uniform
        scores = None
        if score_method == 'uniform':
            scores = np.ones(D.shape[0])
        elif score_method == 'gradnorm':
            scores = score_gradnorm_hist[-1]
        elif score_method == 'residual':
            scores = score_residual_hist[-1]

        # get the indices and new weights
        # use r as seed
        sample_idx, new_weights = utils.compute_subset(D, weights, m, r, scores)

        print(f'm={m} sampled {len(sample_idx)} points ({len(sample_idx)/m*100}%) of which {len(np.unique(sample_idx))} aka {len(np.unique(sample_idx))/len(sample_idx)*100}% are unique')
        sample_idx, sample_idx_counts = np.unique(sample_idx, return_counts=True)
        new_weights[sample_idx] = np.multiply(sample_idx_counts, new_weights[sample_idx])
        print(f'\t m={m} sampled {len(sample_idx)} points ({len(sample_idx)/m*100}%) of which {len(np.unique(sample_idx))} aka {len(np.unique(sample_idx))/len(sample_idx)*100}% are unique')

        sample_idx_BDY = sample_idx[sample_idx < n_BDY]
        sample_idx_OAR = sample_idx[np.logical_and(sample_idx >= n_BDY, sample_idx < (n_BDY+n_OAR))]
        sample_idx_PTV = sample_idx[sample_idx >= (n_BDY+n_OAR)]

        sample_n_BDY = len(sample_idx_BDY)
        sample_n_OAR = len(sample_idx_OAR)
        sample_n_PTV = len(sample_idx_PTV)

        if sample_n_BDY < 5:
            print(f'WARNING: only {sample_n_BDY} BODY voxels have been chosen, increase m which is currently m={m}')
            print('SKIP run')
            continue

        if sample_n_OAR < 5:
            print(f'WARNING: only {sample_n_OAR} OAR voxels have been chosen, increase m which is currently m={m}')
            print('SKIP run')
            continue

        if sample_n_PTV < 5:
            print(f'WARNING: only {sample_n_PTV} PTV voxels have been chosen, increase m which is currently m={m}')
            print('SKIP run')
            continue

        sample_D_BDY = D[sample_idx_BDY]
        sample_D_OAR = D[sample_idx_OAR]
        sample_D_PTV = D[sample_idx_PTV]

        sample_target_dose_PTV = np.ones(sample_n_PTV) * PTV_dose

        sp.sparse.save_npz('sample_D_BDY.npz', sample_D_BDY)
        sp.sparse.save_npz('sample_D_OAR.npz', sample_D_OAR)
        sp.sparse.save_npz('sample_D_PTV.npz', sample_D_PTV)

        np.save('sample_taget_doze_PTV.npy', sample_target_dose_PTV)


################################################################################

if __name__ == '__main__':
  fire.Fire(main)





