import torch
from scipy.stats import spearmanr

def get_imb_metrics(confusion):
    per_class_accs = confusion.diag() / confusion.sum(1)
    per_class_pred_cnt = confusion.sum(0)

    return per_class_accs, per_class_pred_cnt


@torch.no_grad()
def get_nc_metrics(classifier, features, targets):
    C = classifier.shape[-1]
    D = features.shape[-1]
    mean_features = torch.zeros(C, D, device=features.device)
    num_samples = torch.zeros(C, device=features.device)
    Sw = torch.zeros(C, D, D, device=features.device)
    for computation in ['Mean','Cov']:
        for c in range(C):
            idxs = (targets == c).nonzero(as_tuple=True)[0]
            if len(idxs) == 0: # If no class-c in this batch
                continue
            h_c = features[idxs,:]

            if computation == 'Mean':
                mean_features[c,:] += torch.sum(h_c, dim=0)
                num_samples[c] += h_c.shape[0]
            elif computation == 'Cov':
                z = h_c - mean_features[c].unsqueeze(0) # B D
                cov = torch.matmul(z.unsqueeze(-1), z.unsqueeze(1)) # B D D
                Sw[c,:,:] += torch.sum(cov, dim=0) # D D
        if computation == 'Mean':
            mean_features /= num_samples.unsqueeze(-1)
        elif computation == 'Cov':
            Sw /= num_samples.sum()
        
    # global mean
    global_means = torch.mean(mean_features, dim=0, keepdim=True) # 1 D
    # between-class covariance
    centered_means = mean_features - global_means
    Sb = torch.matmul(centered_means.T, centered_means) / C

    # avg norm
    M_norms = torch.norm(centered_means.T, dim=0)
    W_norms = torch.norm(classifier, dim=0)

    # tr{Sw Sb^-1}
    invSb = torch.linalg.pinv(Sb)
    Sw_invSb = torch.matmul(Sw, invSb).diagonal(dim1=-2, dim2=-1).sum(-1)

    cos_M_all, cos_M_nearest_all = mutual_coherence(centered_means.T/M_norms)
    cos_W_all, cos_W_nearest_all = mutual_coherence(classifier/W_norms)
    return Sw_invSb, cos_M_all, cos_W_all, cos_M_nearest_all, cos_W_nearest_all

# mutual coherence
def mutual_coherence(V):
    C = V.shape[1]
    G = V.T @ V
    G += 1 / (C-1)
    G -= torch.diag(torch.diag(G))
    margins = G.abs().sum(dim=1) / (C-1)
    margins_nearest = G.abs().max(dim=1)[0]
    return margins, margins_nearest


def get_corrs(metrics, freqs):
    res = {}
    try:
        accs, preds = metrics['per_class_accs'].numpy(), metrics['per_class_pred_cnt'].numpy()
        corr_acc, corr_pred = spearmanr(freqs, accs).statistic, spearmanr(freqs, preds).statistic
        res['corr_acc'], res['corr_pred'] = corr_acc, corr_pred
    except:
        # print('Imbalance metrics are not supported')
        pass
    try:
        Sw_invSb, cos_M_all, cos_W_all, cos_M_nearest, cos_W_nearest \
            = metrics['Sw_invSb'].numpy(), metrics['cos_M_all'].numpy(), metrics['cos_W_all'].numpy(), metrics['cos_M_nearest'].numpy(), metrics['cos_W_nearest'].numpy()
        corr_Sw_invSb, corr_cos_M_all, corr_cos_W_all, corr_cos_M_nearest, corr_cos_W_nearest \
            = spearmanr(freqs, Sw_invSb).statistic, spearmanr(freqs, cos_M_all).statistic, \
            spearmanr(freqs, cos_W_all).statistic, spearmanr(freqs, cos_M_nearest).statistic, spearmanr(freqs, cos_W_nearest).statistic
        corr_Sw_invSb_acc, corr_cos_M_all_acc, corr_cos_W_all_acc, corr_cos_M_nearest_acc, corr_cos_W_nearest_acc \
            = spearmanr(accs, Sw_invSb).statistic, spearmanr(accs, cos_M_all).statistic, spearmanr(accs, cos_W_all).statistic, \
                spearmanr(accs, cos_M_nearest).statistic, spearmanr(accs, cos_W_nearest).statistic
        res['corr_Sw_invSb'], res['corr_cos_M_all'], res['corr_cos_W_all'], res['corr_cos_M_nearest'], res['corr_cos_W_nearest'] \
            = corr_Sw_invSb, corr_cos_M_all, corr_cos_W_all, corr_cos_M_nearest, corr_cos_W_nearest
        res['corr_Sw_invSb_acc'], res['corr_cos_M_all_acc'], res['corr_cos_W_all_acc'], res['corr_cos_M_nearest_acc'], res['corr_cos_W_nearest_acc'] \
            = corr_Sw_invSb_acc, corr_cos_M_all_acc, corr_cos_W_all_acc, corr_cos_M_nearest_acc, corr_cos_W_nearest_acc
    except:
        # print('NC metrics are not supported')
        pass
    return res