import numpy as np
import pandas as pd
from scipy.stats import pearsonr, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns

import codecademylib3
np.set_printoptions(suppress=True, precision = 2)

nba = pd.read_csv('./nba_games.csv')

# Subset Data to 2010 Season, 2014 Season
nba_2010 = nba[nba.year_id == 2010]
nba_2014 = nba[nba.year_id == 2014]

print(nba_2010.head())
print(nba_2014.head())

knicks_pts_10 = nba_2010.pts[nba.fran_id=='Knicks']
nets_pts_10 = nba_2010.pts[nba.fran_id=='Nets']

#difference between 2 teams avg points:
mean_knicks = np.mean(knicks_pts_10)
mean_nets = np.mean(nets_pts_10)

diff_means_2010 = mean_knicks - mean_nets
print(diff_means_2010)

#histogram:
plt.hist(knicks_pts_10 , color="blue", label="knicks", normed=True, alpha=0.5)
plt.hist(nets_pts_10 , color="pink", label="nets", normed=True, alpha=0.5)
plt.legend()
plt.title('2010 season')
plt.show()


#2014 season:
knicks_pts_14 = nba_2014.pts[nba.fran_id=='Knicks']
nets_pts_14 = nba_2014.pts[nba.fran_id=='Nets']

#mean difference:
knicks_mean_14 = np.mean(knicks_pts_14)
nets_mean_14 = np.mean(nets_pts_14)

diff_mean_14 = knicks_mean_14 - nets_mean_14
print(diff_mean_14)

plt.hist(knicks_pts_14 , color="green", label="knicks", normed=True, alpha=0.5)
plt.hist(nets_pts_14 , color="orange", label="nets", normed=True, alpha=0.5)
plt.legend()
plt.title('2014 season')
plt.show()

#box-plots:
sns.boxplot(data=nba_2010, x='fran_id',y='pts')
plt.show()

#Categorical variables:
location_result_freq = pd.crosstab(nba_2010.game_result,nba_2010.game_location)
print(location_result_freq)

#convert to proportions:
location_result_proportions = location_result_freq/len(nba_2010)
print(location_result_proportions)

#expected contingency table:
chi2, pval, dof, expected = chi2_contingency(location_result_freq)
print(np.round(expected))
print(chi2)

#Quantitative variables:
point_diff_forecast_cov = np.cov(nba_2010.forecast, nba_2010.point_diff)
print(point_diff_forecast_cov)

point_diff_forecast_corr, p = pearsonr(nba_2010.forecast, nba_2010.point_diff)
print(point_diff_forecast_corr)

#scatter plot:
plt.clf()
plt.scatter(x = nba_2010.forecast, y = nba_2010.point_diff)
plt.xlabel('Forecast')
plt.ylabel('Point Diff')
plt.show()