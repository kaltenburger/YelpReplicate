#10/5/2018
# about: we fit a baseline model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

execfile('../python_libraries.py')
cuisine_df = pd.read_pickle('../../data/features_pkl/cuisine.pickle')
dta = pd.read_pickle('../../data/features_pkl/dta.pickle')
zip_code_df = pd.read_pickle('../../data/features_pkl/zip_code_df.pickle')


file_output = open('./results/kang_results_baseline.csv', 'wt')
writer = csv.writer(file_output)
writer.writerow( ('iteration','train_split',  'mae', 'mse'))

dta_analysis_tmp = dta.copy()
dta_analysis = dta_analysis_tmp[['inspection_id', 'inspection_average_prev_penalty_scores',
                                 'inspection_prev_penalty_score', 'inspection_penalty_score',
                                 'review_count', 'non_positive_review_count', 'average_review_rating']]

kf = KFold(n_splits=10,     # 10-fold CV is used in paper
      shuffle = True,      # assuming they randomly select train/test
      random_state = 123)  # random.seed for our own internal replication purposes

mse_features = []
m = 0
for train, test in kf.split(dta_analysis):
    y_train = dta_analysis.inspection_penalty_score.iloc[train]
    y_test = dta_analysis.inspection_penalty_score.iloc[test]
    y_null = np.repeat(np.mean(y_train), len(y_test))
    mse_features.append(mean_squared_error(y_test, y_null))
    print mse_features
    m = m+1
    writer.writerow( (m, train,
                          mean_absolute_error(y_test, y_null),
                          mean_squared_error(y_test, y_null)))
print ''
 
