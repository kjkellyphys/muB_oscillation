import yaml
import numpy as np
import matplotlib.pyplot as plt

# load yaml files

yamlDir = "." # location of downloaded yaml files
yaml_dict = {}

# constrained nueCC FC
with open(yamlDir+"/constrained_nueCC_FC.yaml", "r") as stream:
    try:
        yaml_dict["nueCC_FC_constrained"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
with open(yamlDir+"/constrained_nueCC_FC_cov.yaml", "r") as stream:
    try:
        yaml_dict["cov_nueCC_FC_constrained"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
# unconstrained 7 channels
with open(yamlDir+"/nueCC_FC.yaml", "r") as stream:
    try:
        yaml_dict["nueCC_FC"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
with open(yamlDir+"/nueCC_PC.yaml", "r") as stream:
    try:
        yaml_dict["nueCC_PC"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
with open(yamlDir+"/numuCC_FC.yaml", "r") as stream:
    try:
        yaml_dict["numuCC_FC"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
with open(yamlDir+"/numuCC_PC.yaml", "r") as stream:
    try:
        yaml_dict["numuCC_PC"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
with open(yamlDir+"/numuCCpi0_FC.yaml", "r") as stream:
    try:
        yaml_dict["numuCCpi0_FC"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
with open(yamlDir+"/numuCCpi0_PC.yaml", "r") as stream:
    try:
        yaml_dict["numuCCpi0_PC"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
with open(yamlDir+"/NCpi0.yaml", "r") as stream:
    try:
        yaml_dict["NCpi0"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open(yamlDir+"/cov_lee_x_00.yaml", "r") as stream:
    try:
        yaml_dict["cov_no_lee"] = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        
# convert yaml dictionaries into numpy arrays

nueCC_FC_constrained_data = []
nueCC_FC_constrained_data_lower_err = []
nueCC_FC_constrained_data_upper_err = []
nueCC_FC_constrained_pred = []
for i in range(len(yaml_dict["nueCC_FC_constrained"]['dependent_variables'][0]['values'])):
    nueCC_FC_constrained_data.append(yaml_dict["nueCC_FC_constrained"]['dependent_variables'][0]['values'][i]["value"])
    nueCC_FC_constrained_data_lower_err.append(yaml_dict["nueCC_FC_constrained"]['dependent_variables'][0]['values'][i]["errors"][0]["asymerror"]["minus"])
    nueCC_FC_constrained_data_upper_err.append(yaml_dict["nueCC_FC_constrained"]['dependent_variables'][0]['values'][i]["errors"][0]["asymerror"]["plus"])
    nueCC_FC_constrained_pred.append(yaml_dict["nueCC_FC_constrained"]['dependent_variables'][2]['values'][i]["value"])
nueCC_FC_constrained_data_arr = np.array(nueCC_FC_constrained_data)
nueCC_FC_constrained_data_lower_err_arr = np.array(nueCC_FC_constrained_data_lower_err)
nueCC_FC_constrained_data_upper_err_arr = np.array(nueCC_FC_constrained_data_upper_err)
nueCC_FC_constrained_pred_arr = np.array(nueCC_FC_constrained_pred)

nueCC_FC_constrained_cov = []
for i in range(len(yaml_dict["cov_nueCC_FC_constrained"]['dependent_variables'][0]['values'])):
    if i % 26 == 0:
        nueCC_FC_constrained_cov.append([])
    nueCC_FC_constrained_cov[-1].append(yaml_dict["cov_nueCC_FC_constrained"]['dependent_variables'][0]['values'][i]["value"])
nueCC_FC_constrained_cov = np.array(nueCC_FC_constrained_cov)

nueCC_FC_data = []
nueCC_FC_data_lower_err = []
nueCC_FC_data_upper_err = []
nueCC_FC_pred = []
for i in range(len(yaml_dict["nueCC_FC"]['dependent_variables'][0]['values'])):
    nueCC_FC_data.append(yaml_dict["nueCC_FC"]['dependent_variables'][0]['values'][i]["value"])
    nueCC_FC_data_lower_err.append(yaml_dict["nueCC_FC"]['dependent_variables'][0]['values'][i]["errors"][0]["asymerror"]["minus"])
    nueCC_FC_data_upper_err.append(yaml_dict["nueCC_FC"]['dependent_variables'][0]['values'][i]["errors"][0]["asymerror"]["plus"])
    nueCC_FC_pred.append(yaml_dict["nueCC_FC"]['dependent_variables'][2]['values'][i]["value"])
nueCC_FC_data_arr = np.array(nueCC_FC_data)
nueCC_FC_data_lower_err_arr = np.array(nueCC_FC_data_lower_err)
nueCC_FC_data_upper_err_arr = np.array(nueCC_FC_data_upper_err)
nueCC_FC_pred_arr = np.array(nueCC_FC_pred)

nueCC_PC_data = []
nueCC_PC_pred = []
for i in range(len(yaml_dict["nueCC_PC"]['dependent_variables'][0]['values'])):
    nueCC_PC_data.append(yaml_dict["nueCC_PC"]['dependent_variables'][0]['values'][i]["value"])
    nueCC_PC_pred.append(yaml_dict["nueCC_PC"]['dependent_variables'][2]['values'][i]["value"])
nueCC_PC_data_arr = np.array(nueCC_PC_data)
nueCC_PC_pred_arr = np.array(nueCC_PC_pred)

numuCC_FC_data = []
numuCC_FC_pred = []
for i in range(len(yaml_dict["numuCC_FC"]['dependent_variables'][0]['values'])):
    numuCC_FC_data.append(yaml_dict["numuCC_FC"]['dependent_variables'][0]['values'][i]["value"])
    numuCC_FC_pred.append(yaml_dict["numuCC_FC"]['dependent_variables'][2]['values'][i]["value"])
numuCC_FC_data_arr = np.array(numuCC_FC_data)
numuCC_FC_pred_arr = np.array(numuCC_FC_pred)

numuCC_PC_data = []
numuCC_PC_pred = []
for i in range(len(yaml_dict["numuCC_PC"]['dependent_variables'][0]['values'])):
    numuCC_PC_data.append(yaml_dict["numuCC_PC"]['dependent_variables'][0]['values'][i]["value"])
    numuCC_PC_pred.append(yaml_dict["numuCC_PC"]['dependent_variables'][2]['values'][i]["value"])
numuCC_PC_data_arr = np.array(numuCC_PC_data)
numuCC_PC_pred_arr = np.array(numuCC_PC_pred)

numuCCpi0_FC_data = []
numuCCpi0_FC_pred = []
for i in range(len(yaml_dict["numuCCpi0_FC"]['dependent_variables'][0]['values'])):
    numuCCpi0_FC_data.append(yaml_dict["numuCCpi0_FC"]['dependent_variables'][0]['values'][i]["value"])
    numuCCpi0_FC_pred.append(yaml_dict["numuCCpi0_FC"]['dependent_variables'][2]['values'][i]["value"])
numuCCpi0_FC_data_arr = np.array(numuCCpi0_FC_data)
numuCCpi0_FC_pred_arr = np.array(numuCCpi0_FC_pred)

numuCCpi0_PC_data = []
numuCCpi0_PC_pred = []
for i in range(len(yaml_dict["numuCCpi0_PC"]['dependent_variables'][0]['values'])):
    numuCCpi0_PC_data.append(yaml_dict["numuCCpi0_PC"]['dependent_variables'][0]['values'][i]["value"])
    numuCCpi0_PC_pred.append(yaml_dict["numuCCpi0_PC"]['dependent_variables'][2]['values'][i]["value"])
numuCCpi0_PC_data_arr = np.array(numuCCpi0_PC_data)
numuCCpi0_PC_pred_arr = np.array(numuCCpi0_PC_pred)

NCpi0_data = []
NCpi0_pred = []
for i in range(len(yaml_dict["NCpi0"]['dependent_variables'][0]['values'])):
    NCpi0_data.append(yaml_dict["NCpi0"]['dependent_variables'][0]['values'][i]["value"])
    NCpi0_pred.append(yaml_dict["NCpi0"]['dependent_variables'][2]['values'][i]["value"])
NCpi0_data_arr = np.array(NCpi0_data)
NCpi0_pred_arr = np.array(NCpi0_pred)

cov = []
for i in range(len(yaml_dict["cov_no_lee"]['dependent_variables'][0]['values'])):
    if i % 137 == 0:
        cov.append([])
    cov[-1].append(yaml_dict["cov_no_lee"]['dependent_variables'][0]['values'][i]["value"])
cov = np.array(cov)

all_pred_arr = np.concatenate([nueCC_FC_pred_arr, nueCC_PC_pred_arr, numuCC_FC_pred_arr, numuCC_PC_pred_arr, numuCCpi0_FC_pred_arr, numuCCpi0_PC_pred_arr, NCpi0_pred_arr])
all_data_arr = np.concatenate([nueCC_FC_data_arr, nueCC_PC_data_arr, numuCC_FC_data_arr, numuCC_PC_data_arr, numuCCpi0_FC_data_arr, numuCCpi0_PC_data_arr, NCpi0_data_arr])



# make unconstrained plot and goodness of fit

bins = np.linspace(0, 2.5, 26)
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

# removing overflow bins
nueCC_FC_pred = nueCC_FC_pred_arr[:25]
nueCC_FC_data = nueCC_FC_data_arr[:25]
nueCC_FC_data_lower_err = nueCC_FC_data_lower_err_arr[:25]
nueCC_FC_data_upper_err = nueCC_FC_data_upper_err_arr[:25]
nueCC_cov = cov[:25, :25]

nueCC_FC_sigma = np.array([np.sqrt(nueCC_cov[i, i]) for i in range(25)])

# adding diagonal Pearson covariance matrix for the data statistical uncertainty to the covariance matrix for the GoF test
nueCC_cov_with_stat = np.copy(nueCC_cov)
for i in range(25):
    nueCC_cov_with_stat[i, i] += nueCC_FC_pred[i]  
chi2 = np.linalg.multi_dot(
    [np.transpose(nueCC_FC_data - nueCC_FC_pred),
     np.linalg.inv(nueCC_cov_with_stat), 
     nueCC_FC_data - nueCC_FC_pred])

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1]})
ax1.hist(bin_centers, weights=nueCC_FC_pred, bins=bins, label="Unconstrained Total Prediction")
ax1.errorbar(bin_centers, nueCC_FC_data, 
             yerr=np.abs(np.array([nueCC_FC_data_lower_err, nueCC_FC_data_upper_err])), 
             xerr=0.05, fmt="none", c="k", capsize=4)
ax1.scatter(bin_centers, nueCC_FC_data, c="k", label="6.369e20 POT MicroBooNE BNB Data", zorder=10)
ax1.bar(bin_centers, [2 * nueCC_FC_sigma[i] for i in range(25)], 0.1, [nueCC_FC_pred[i] - nueCC_FC_sigma[i] for i in range(25)], 
        alpha=0.5, color="gray", label="Unconstrained Systematic Error")
ax1.set_title(r"Unconstrained $\nu_e$ CC FC")
ax1.set_ylabel("Count / 0.1 GeV")
ax1.text(2., 30, fr"$\chi^2/ndf$ = {np.round(chi2, 2)}/25")
ax1.set_xlim((0., 2.5))
ax1.set_ylim((0., 40))

ax2.errorbar(bin_centers, nueCC_FC_data / nueCC_FC_pred, 
             yerr=np.abs(np.array([nueCC_FC_data_lower_err / nueCC_FC_pred, nueCC_FC_data_upper_err / nueCC_FC_pred])), 
             xerr=0.05, fmt="none", c="k", capsize=4)
ax2.scatter(bin_centers, nueCC_FC_data / nueCC_FC_pred, c="k", label="6.369e20 POT MicroBooNE BNB Data", zorder=10)

ax2.bar(bin_centers, [2 * nueCC_FC_sigma[i] / nueCC_FC_pred[i] for i in range(25)], 0.1, [1. - nueCC_FC_sigma[i] / nueCC_FC_pred[i] for i in range(25)], 
        alpha=0.5, color="gray", label="Unconstrained Systematic Error")

ax2.set_ylim((0, 2))
ax2.set_xlabel("Reconstructed Neutrino Energy (GeV)")
ax2.set_ylabel("Data/Pred")

ax2.axhline(1, c="k", ls="--")
ax1.legend()
plt.xlabel
#plt.show() # matches Fig. 24a


# make constrained plot and goodness of fit from constrained file

bins = np.linspace(0, 2.5, 26)
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

# removing overflow bins
nueCC_FC_constrained_pred = nueCC_FC_constrained_pred_arr[:25]
nueCC_FC_constrained_data = nueCC_FC_constrained_data_arr[:25]
nueCC_FC_constrained_data_lower_err = nueCC_FC_constrained_data_lower_err_arr[:25]
nueCC_FC_constrained_data_upper_err = nueCC_FC_constrained_data_upper_err_arr[:25]
nueCC_FC_constrained_cov = nueCC_FC_constrained_cov[:25, :25]

nueCC_FC_constrained_sigma = np.array([np.sqrt(nueCC_FC_constrained_cov[i, i]) for i in range(25)])

# adding diagonal Pearson covariance matrix for the data statistical uncertainty to the covariance matrix for the GoF test
nueCC_constrained_cov_with_stat = np.copy(nueCC_FC_constrained_cov)
for i in range(25):
    nueCC_constrained_cov_with_stat[i, i] += nueCC_FC_constrained_pred[i]  
chi2 = np.linalg.multi_dot(
    [np.transpose(nueCC_FC_constrained_data - nueCC_FC_constrained_pred),
     np.linalg.inv(nueCC_constrained_cov_with_stat), 
     nueCC_FC_constrained_data - nueCC_FC_constrained_pred])

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1]})
ax1.hist(bin_centers, weights=nueCC_FC_constrained_pred, bins=bins, label="Constrained Total Prediction")
ax1.errorbar(bin_centers, nueCC_FC_constrained_data, 
             yerr=np.abs(np.array([nueCC_FC_constrained_data_lower_err, nueCC_FC_constrained_data_upper_err])), 
             xerr=0.05, fmt="none", c="k", capsize=4)
ax1.scatter(bin_centers, nueCC_FC_constrained_data, c="k", label="6.369e20 POT MicroBooNE BNB Data", zorder=10)
ax1.bar(bin_centers, [2 * nueCC_FC_constrained_sigma[i] for i in range(25)], 0.1, [nueCC_FC_constrained_pred[i] - nueCC_FC_constrained_sigma[i] for i in range(25)], 
        alpha=0.5, color="gray", label="Constrained Systematic Error")
ax1.set_title(r"Constrained $\nu_e$ CC FC From Constrained Files")
ax1.set_ylabel("Count / 0.1 GeV")
ax1.text(2., 30, fr"$\chi^2/ndf$ = {np.round(chi2, 2)}/25")
ax1.set_xlim((0., 2.5))
ax1.set_ylim((0., 40))

ax2.errorbar(bin_centers, nueCC_FC_constrained_data / nueCC_FC_constrained_pred, 
             yerr=np.abs(np.array([nueCC_FC_constrained_data_lower_err / nueCC_FC_constrained_pred, nueCC_FC_constrained_data_upper_err / nueCC_FC_constrained_pred])), 
             xerr=0.05, fmt="none", c="k", capsize=4)
ax2.scatter(bin_centers, nueCC_FC_constrained_data / nueCC_FC_constrained_pred, c="k", label="6.369e20 POT MicroBooNE BNB Data", zorder=10)

ax2.bar(bin_centers, [2 * nueCC_FC_constrained_sigma[i] / nueCC_FC_constrained_pred[i] for i in range(25)], 0.1, [1. - nueCC_FC_constrained_sigma[i] / nueCC_FC_constrained_pred[i] for i in range(25)], 
        alpha=0.5, color="gray", label="Constrained Systematic Error")

ax2.set_ylim((0, 2))
ax2.set_xlabel("Reconstructed Neutrino Energy (GeV)")
ax2.set_ylabel("Data/Pred")

ax2.axhline(1, c="k", ls="--")
ax1.legend()
plt.xlabel
#plt.show() # matches Fig. 25

# make constrained plot and goodness of fit from unconstrained 7 channels

constraining_pred = all_pred_arr[26:]
constraining_data = all_data_arr[26:]

nueCC_cov_with_overflow = cov[:26, :26]
cov_constraining = cov[26:, 26:]
cov_cross = cov[:26, 26:]

# adding diagonal Pearson covariance matrix for the data statistical uncertainty to the constraining covariance matrix
cov_constraining_with_stat = np.copy(cov_constraining)
for i in range(25):
    cov_constraining_with_stat[i, i] += constraining_pred[i]

nueCC_FC_constrained_pred_with_overflow = nueCC_FC_pred_arr + np.linalg.multi_dot(
    [cov_cross, np.linalg.inv(cov_constraining_with_stat), constraining_data - constraining_pred])
nueCC_FC_cov_constrained = nueCC_cov_with_overflow - np.linalg.multi_dot(
    [cov_cross, np.linalg.inv(cov_constraining_with_stat), np.transpose(cov_cross)])

nueCC_FC_constrained_pred = nueCC_FC_constrained_pred_with_overflow[:-1] # removing overflow bin

nueCC_FC_cov_constrained_no_overflow_with_stat = np.copy(nueCC_FC_cov_constrained)[:25, :25]
for i in range(25):
    nueCC_FC_cov_constrained_no_overflow_with_stat[i, i] += nueCC_FC_constrained_pred[i]
    
constrained_chi2 = np.linalg.multi_dot(
    [np.transpose(nueCC_FC_data - nueCC_FC_constrained_pred),
     np.linalg.inv(nueCC_FC_cov_constrained_no_overflow_with_stat),
     nueCC_FC_data - nueCC_FC_constrained_pred])

nueCC_FC_constrained_sigma = [np.sqrt(nueCC_FC_cov_constrained[i][i]) for i in range(25)]

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1]})
ax1.hist(bin_centers, weights=nueCC_FC_constrained_pred, bins=bins, label="Constrained Total Prediction")
ax1.errorbar(bin_centers, nueCC_FC_data, 
             yerr=np.abs(np.array([nueCC_FC_data_lower_err, nueCC_FC_data_upper_err])), 
             xerr=0.05, fmt="none", c="k", capsize=4)
ax1.scatter(bin_centers, nueCC_FC_data, c="k", label="6.369e20 POT MicroBooNE BNB Data", zorder=10)
ax1.bar(bin_centers, [2 * nueCC_FC_constrained_sigma[i] for i in range(25)], 0.1, [nueCC_FC_constrained_pred[i] - nueCC_FC_constrained_sigma[i] for i in range(25)], 
        alpha=0.5, color="gray", label="Constrained Systematic Error")
ax1.set_title(r"Constrained $\nu_e$ CC FC From Unconstrained 7 Channel Files")
ax1.set_ylabel("Count / 0.1 GeV")
ax1.text(2., 30, fr"$\chi^2/ndf$ = {np.round(constrained_chi2, 2)}/25")
ax1.set_xlim((0., 2.5))
ax1.set_ylim((0., 40))


ax2.errorbar(bin_centers, nueCC_FC_data / nueCC_FC_constrained_pred, 
             yerr=np.abs(np.array([nueCC_FC_data_lower_err / nueCC_FC_constrained_pred, nueCC_FC_data_upper_err / nueCC_FC_constrained_pred])), 
             xerr=0.05, fmt="none", c="k", capsize=4)
ax2.scatter(bin_centers, nueCC_FC_data / nueCC_FC_constrained_pred, c="k", label="6.369e20 POT MicroBooNE BNB Data", zorder=10)

ax2.bar(bin_centers, [2 * nueCC_FC_constrained_sigma[i] / nueCC_FC_constrained_pred[i] for i in range(25)], 0.1, [1. - nueCC_FC_constrained_sigma[i] / nueCC_FC_constrained_pred[i] for i in range(25)], 
        alpha=0.5, color="gray", label="Constrained Systematic Error")


ax2.set_ylim((0, 2))
ax2.set_xlabel("Reconstructed Neutrino Energy (GeV)")
ax2.set_ylabel("Data/Pred")

ax2.axhline(1, c="k", ls="--")
ax1.legend()
plt.xlabel
plt.show() # matches Fig. 25


# Note that these two constrained plots are very similar but not 100% identical; This is due to the fact that the second plot sums the constrained
# signal and background, while the third plot sums signal and background and only then constrains. Due to correlations between signal and background,
# there is a small difference in these procedures.
