#Open Cluster Automatic Membership Determination with Gaia data (gaia_oc_amd) is a
#machine-learning-based tool designed to determine open cluster membership using Gaia DR3 data. 
#It was developed by M. G. J. van Groeningen.

#!!!This code is based on the training examples provided in the original implementation.
#For more details, see van Groeningen et al. (2023) and the corresponding GitHub repository associated with that work.

#Before starting, you should download the code directly from GitHub using: git clone https://github.com/MGJvanGroeningen/gaia_oc_amd
#Then, go to the repository directory and install the dependencies: pip install -r requirements.txt
#Create an username and password to log into the Gaia archive.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gaia_oc_amd.data_preparation.query import query_vizier_catalog
from gaia_oc_amd.io import load_cluster_parameters
from gaia_oc_amd.data_preparation.cluster import Cluster
from gaia_oc_amd import create_gaia_credentials_file
from gaia_oc_amd.data_preparation.query import cone_search
from gaia_oc_amd.io import load_cone
from gaia_oc_amd.data_preparation.cone_preprocessing import cone_preprocessing
from gaia_oc_amd.io import load_members
from gaia_oc_amd.data_preparation.source_sets import member_set
from gaia_oc_amd.candidate_evaluation.visualization import plot_sources
from gaia_oc_amd.data_preparation.query import query_isochrone
from gaia_oc_amd.io import load_isochrone
from gaia_oc_amd.data_preparation.isochrone_preprocessing import isochrone_preprocessing
from gaia_oc_amd.data_preparation.source_sets import candidate_and_non_member_set
from gaia_oc_amd.data_preparation.source_sets import get_duplicate_sources
from gaia_oc_amd.data_preparation.features import add_features
from gaia_oc_amd.io import save_sets, save_cluster
from gaia_oc_amd.data_preparation.datasets import property_mean_and_std, MultiClusterDeepSetsDataset
from torch.utils.data import DataLoader
from gaia_oc_amd.machine_learning.deepsets_zaheer import D5
from gaia_oc_amd.machine_learning.training import train_model
from gaia_oc_amd.candidate_evaluation.visualization import plot_metrics
from gaia_oc_amd.candidate_evaluation.membership_probability import calculate_candidate_probs
from gaia_oc_amd.io import load_sets
import importlib

#Create a directory to store data in
data_save_dir = './data'
if not os.path.exists(data_save_dir):
    os.mkdir(data_save_dir)
    
#Query the VizieR catalogue to download cluster parameter data.
#In this case, we use the Hunt-Reffert+23 catalogue (training dataset 1). 
HR23_cluster_params_catalog = 'J/A+A/673/A114/clusters'
HR23_cluster_params_path = os.path.join(data_save_dir, 'cluster_parameters_HR23_6259.csv')

#The stellar parameters should be modified depending on the training dataset.
if not os.path.exists(HR23_cluster_params_path):
    query_vizier_catalog(HR23_cluster_params_catalog, 
                         save_path=HR23_cluster_params_path,
                         columns=['Name', 'RA_ICRS', 'DE_ICRS', 'pmRA', 'pmDE', 's_pmRA', 
                                   's_pmDE', 'Plx', 's_Plx','logAge84', 'AV84', 'dist84'],
                         new_column_names={'Name': 'name', 
                                           'RA_ICRS': 'ra', 
                                           'DE_ICRS': 'dec', 
                                           'pmRA': 'pmra',
                                           'pmDE': 'pmdec', 
                                           's_pmRA': 'pmra_error', 
                                           's_pmDE': 'pmdec_error',
                                           'Plx': 'parallax', 
                                           's_Plx': 'parallax_error', 
                                           'logAge84': 'age', 
                                           'AV84': 'a0', 
                                           'dist84': 'dist'})
    print('Downloaded Hunt_Reffert+23 NGC6259 cluster parameters.')
    
#If you change the VizieR catalogue to Cantat-Gaudin+20 (training dataset 1), update the column names as follows:

#cg20_cluster_params_catalog = 'J/A+A/640/A1/table1'
#cg20_cluster_params_path = os.path.join(data_save_dir, 'cluster_parameters.csv')

#if not os.path.exists(cg20_cluster_params_path):
#    query_vizier_catalog(cg20_cluster_params_catalog, 
#                         save_path=cg20_cluster_params_path,
#                         columns=['Cluster', 'RA_ICRS', 'DE_ICRS', 'pmRA*', 'pmDE', 'e_pmRA*', 
#                                   'e_pmDE', 'plx', 'e_plx','AgeNN', 'AVNN', 'DistPc'],
#                         new_column_names={'Cluster': 'name', 
#                                           'RA_ICRS': 'ra', 
#                                           'DE_ICRS': 'dec', 
#                                           'pmRA_': 'pmra',
#                                           'pmDE': 'pmdec', 
#                                           'e_pmRA_': 'pmra_error', 
#                                           'e_pmDE': 'pmdec_error',
#                                           'plx': 'parallax', 
#                                           'e_plx': 'parallax_error', 
#                                           'AgeNN': 'age', 
#                                           'AVNN': 'a0', 
#                                           'DistPc': 'dist'})
#    print('Downloaded Cantat-Gaudin+20 cluster parameters.')
#cluster_name = 'NGC_6259'
#cluster_params = load_cluster_parameters(cg20_cluster_params_path, cluster_name)

# Our 'Cluster' object is NGC 6259
cluster_name = 'NGC_6259'
cluster_params = load_cluster_parameters(HR23_cluster_params_path, cluster_name)
cluster = Cluster(cluster_params)
print(cluster_params)

#Now we download sources from the Gaia archive that have might be members of the cluster
gaia_credentials_path = './gaia_credentials'

if not os.path.exists(gaia_credentials_path):
    create_gaia_credentials_file(save_path=gaia_credentials_path)

clusters_save_dir = os.path.join(data_save_dir, 'cluster_NGC6259_from_HR23')

cone_search(cluster=cluster, save_dir=clusters_save_dir, gaia_credentials_path=gaia_credentials_path,
            output_format='votable', cone_radius=50., pm_sigmas=10., plx_sigmas=10.)

#Apply cone_preprocessing to clean and prepare the Gaia cone for membership analysis.

cluster_dir = os.path.join(clusters_save_dir, cluster.name)
cone = load_cone(cluster_dir)
cone = cone_preprocessing(cone)

print("Total cone sources :", (len(cone)))

#TRAIN MEMBERS
HR23_members_catalogue = 'J/A+A/673/A114/members'
HR23_members_path = os.path.join(data_save_dir, 'HR23_members.csv')

if not os.path.exists(HR23_members_path):
    query_vizier_catalog(HR23_members_catalogue, 
                         save_path = HR23_members_path,
                         columns=['Name', 'GaiaDR3', 'Prob'],
                         new_column_names={'Name': 'cluster', 'GaiaDR3': 'source_id', 'Prob': 'PMemb'})
    print('Download Hunt_Reffert+23 cluster members.')

HR23_members = load_members(HR23_members_path, cluster.name)

print(HR23_members.head())
print('Total HR23 members:', len(HR23_members))
print('Mean member probability:', HR23_members['PMemb'].mean())

prob_threshold = 0.8
train_members = HR23_members.query(f'PMemb >= {prob_threshold}')
print('Number of train members:', len(train_members))

# Construct the member set
train_members = member_set(cone, train_members['source_id'], train_members['PMemb'])

print(train_members.head())
plot_sources(train_members)

#ISOCHRONE
# Download the isochrone corresponding to the cluster age (assuming a default metallicity),
#which is used to evaluate source membership based on its position in the CMD
import gaia_oc_amd.data_preparation.query as query
importlib.reload(query)
from gaia_oc_amd.data_preparation.query import query_isochrone

print('Cluster log(age):', cluster.age)

isochrone_path = os.path.join(data_save_dir, 'isochrones_test_HR.dat')

if not os.path.exists(isochrone_path):
    query_isochrone(isochrone_path, 
                    log_age_min=cluster.age, 
                    log_age_max=cluster.age, 
                    log_age_step=0., 
                    metal_frac=0.0152)
isochrone = load_isochrone(isochrone_path, cluster.age)
print(isochrone.head())

fig, ax = plt.subplots(1,1,figsize=(6, 6))
ax.plot(isochrone['phot_g_mean_mag'] - isochrone['G_RPmag'], isochrone['phot_g_mean_mag'], zorder=-1)
ax.scatter(isochrone['phot_g_mean_mag'] - isochrone['G_RPmag'], isochrone['phot_g_mean_mag'], s=5, c='orange', zorder=0)
ax.invert_yaxis()
ax.set_xlabel('G - RP')
ax.set_ylabel('G')
ax.set_title(f'Isochrone of log(age)={cluster.age}')
plt.show()

#Correct the isochrone magnitudes for the cluster distance and the interstellar extinction
print('Cluster distance:', cluster.dist, 'pc')
print('Cluster extinction:', cluster.a0)

processed_isochrone = isochrone_preprocessing(isochrone, cluster.dist, colour='g_rp', a0=cluster.a0, 
                                              oldest_stage=7, interpolation_density=5., 
                                              oldest_stage_to_interpolate=5)
old_isochrone = isochrone_preprocessing(isochrone, cluster.dist, colour='g_rp', a0=cluster.a0, 
                                        oldest_stage=7, interpolation_density=0., 
                                        oldest_stage_to_interpolate=5)

print(processed_isochrone.head())
fig, ax = plt.subplots(1,1,figsize=(6, 6))
ax.plot(processed_isochrone['g_rp'], processed_isochrone['phot_g_mean_mag'], zorder=-1)
ax.scatter(old_isochrone['g_rp'], old_isochrone['phot_g_mean_mag'], 
           s=5, c='orange', zorder=1, label='original data')
ax.scatter(processed_isochrone['g_rp'], processed_isochrone['phot_g_mean_mag'], 
           s=5, c='red', zorder=0, label='additional interpolated data')
ax.invert_yaxis()
ax.set_xlabel('G - RP')
ax.set_ylabel('G')
ax.set_title('Isochrone points used in candidate selection')
ax.legend()
plt.show()

#LABELING
#Label the cone sources as either candidate or non-member

# Update the cluster parameters based on the members
cluster.update_astrometric_parameters(train_members)

# Set cluster parameters that are relevant for the candidate selection and training features
cluster.set_candidate_selection_parameters(train_members, processed_isochrone, colour='g_rp', 
                                           source_error_weight=3., pm_error_weight=3., r_max_margin=15., 
                                           zpt_error=0.015, c_margin=0.1, 
                                           g_margin=0.8, r_threshold_member_fraction=0.90,
                                           iso_threshold_member_fraction=0.90, alpha=0.05)

candidates, non_members = candidate_and_non_member_set(cone, cluster)

print('Members:', len(train_members))
print('Candidates:', len(candidates))
print('Non members:', len(non_members))

dubious_members = get_duplicate_sources(train_members, non_members, keep='last')

print('Number of dubious sources:', len(dubious_members))

if len(dubious_members) > 0:
    train_members = train_members[~train_members['source_id'].isin(dubious_members['source_id'])].copy()
    non_members = non_members[~non_members['source_id'].isin(dubious_members['source_id'])].copy()
    candidates = pd.concat((candidates, dubious_members))

limits = {'phot_g_mean_mag': [4, 21.5], 'g_rp': [-0.5, 2.0]}

plot_sources(train_members, colour='g_rp', candidates=candidates, field_sources=non_members, 
             plot_type='candidates', cluster=cluster, show_boundaries=True, show_isochrone=True, 
             limits=limits, save=True)

#FEATURES
# Define and compute a set of astrometric and photometric features used to distinguish cluster members from non-memberS

add_features([train_members, candidates, non_members], cluster)
print(train_members[['f_r', 'f_pm', 'f_plx', 'f_c', 'f_g']][:5])

plot_sources(train_members, colour='g_rp', candidates=candidates, field_sources=non_members, 
             plot_type='candidates', cluster=cluster, show_features_source_id=candidates['source_id'].values[0], 
             show_boundaries=True, show_isochrone=True, limits=limits)
save_cluster(cluster_dir, cluster)
save_sets(cluster_dir, train_members, candidates, non_members)

#Creating a training dataset
#Build the training dataset combining members and non-members suitable for training a Deep Set model
cluster_names = [cluster_name]
source_features = ['f_r', 'f_pm', 'f_plx', 'f_c', 'f_g']
cluster_features = ['a0', 'age', 'parallax']

means, stds = property_mean_and_std(clusters_save_dir, cluster_names, source_features, cluster_features)

train_dataset = DataLoader(MultiClusterDeepSetsDataset(clusters_save_dir, cluster_names, source_features,
                                                       source_feature_means=means['source'], 
                                                       source_feature_stds=stds['source'],
                                                       cluster_feature_names=cluster_features,
                                                       cluster_feature_means=means['cluster'],
                                                       cluster_feature_stds=stds['cluster'],
                                                       n_pos_duplicates=2,
                                                       neg_pos_ratio=5,
                                                       n_min_members=15,
                                                       size_support_set=10), 
                           batch_size=32, shuffle=True)
model_input, label = next(iter(train_dataset))
print('Input shape:', model_input.shape)
print('Model input:', model_input[0])
print('Label:', label[0])

#Training the model
hidden_size = 64
model = D5(hidden_size, x_dim=2 * len(source_features) + len(cluster_features), pool='mean', out_dim=2)
model_dir = './tutorial_model'

metrics = train_model(model, model_dir, train_dataset, val_dataset=None, 
                      num_epochs=20, lr=1e-5, l2=1e-5, weight_imbalance=5.)
plot_metrics(metrics, model_dir, save=True)

#Evaluating candidates
# Evaluate candidate members using a trained model by resampling their features according to measurement uncertainties 
#and deriving membership probabilities from repeated model predictions

n_samples = 100
calculate_candidate_probs(cluster_dir, model_dir, n_samples=n_samples)

_, candidates, _, _ = load_sets(cluster_dir)

prob_threshold = 0.9
member_candidates = candidates.query(f'PMemb >= {prob_threshold}')
non_member_candidates = pd.concat((non_members, candidates.query(f'PMemb < {prob_threshold}')))

plot_title = f'{cluster.name}'.replace('_', ' ')

plot_sources(member_candidates, field_sources=non_member_candidates,
             members_label=f'probable members ($p\\geq${prob_threshold})', 
             title=plot_title, limits=limits, save=True)
print('Number of member candidates:', len(member_candidates))
print('Mean of member candidates:', member_candidates['PMemb'].mean())
member_candidates.to_csv('resultsNN_HR_NGC6259.csv')
print('Results saved in resultsNN_HR_NGC6259.csv.')

