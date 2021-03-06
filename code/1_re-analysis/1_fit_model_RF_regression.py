#10/31/2018
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

execfile('../python_libraries.py')
cuisine_df = pd.read_pickle('../../data/features_pkl/cuisine.pickle')
dta = pd.read_pickle('../../data/features_pkl/dta.pickle')
zip_code_df = pd.read_pickle('../../data/features_pkl/zip_code_df.pickle')


CV_model = True
model_type = 'RF' # RF SVR
search_type = 'random' #random grid
kf = KFold(n_splits = 3, shuffle = True, random_state = 5)



if CV_model:
    if model_type=='RF':
        max_depth = [None, 5, 10]
        min_samples_leaf = [0.0005, 0.01, 0.05, 0.1]
        min_samples_split = [2, 5, 10] #check
        n_estimators = [100, 200, 500]
        max_features = [None, 0.25, 0.5, 0.75]
        param_grid = {'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_leaf': min_samples_leaf,
                        'n_estimators':n_estimators}
        clf = sklearn.ensemble.RandomForestRegressor(random_state = 123)
        if search_type == 'grid':
            model = GridSearchCV(estimator = clf,
                         param_grid = param_grid,
                         cv = kf, verbose=0,
                         n_jobs = 4, scoring = 'neg_mean_squared_error')
        if search_type == 'random':
            model = RandomizedSearchCV(estimator = clf, random_state = 123,
                                      param_distributions = param_grid,
                                      cv = kf, verbose=0, n_iter = 25,
                                      n_jobs = 5, scoring = 'neg_mean_squared_error')
    if model_type =='SVR':
        Cs = [0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = {'C': Cs}
        model = GridSearchCV(estimator = SVR(), param_grid = param_grid, cv=kf,
                             verbose = 0, n_jobs = 3, scoring = 'neg_mean_squared_error')

else:
    if model_type=='RF':
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=200, random_state = 123)
    if model_type=='SVR':
        model = SVR(C=100)


if CV_model:
    file_output = open('./results/kang_results_regression_otherFeatures_SEED_'+model_type+'_search_'+search_type+'.csv', 'wt')
    writer = csv.writer(file_output)
    writer.writerow( ('iteration','feature', 'train_split', 'model_params', #'cv_results',
                      'mae', 'mse'))

dta_analysis_tmp = dta.copy()
dta_analysis = dta_analysis_tmp[['inspection_id', 'inspection_average_prev_penalty_scores',
                                 'inspection_prev_penalty_score', 'inspection_penalty_score',
                                 'review_count', 'non_positive_review_count', 'average_review_rating']]


y_null = np.repeat(np.mean(dta_analysis_tmp.inspection_penalty_score), len(dta_analysis_tmp))
print mean_squared_error(dta_analysis_tmp.inspection_penalty_score, y_null)


dta_analysis = dta_analysis.merge(cuisine_df,
              on = 'inspection_id',
              how = 'left')
dta_analysis = dta_analysis.merge(zip_code_df,
              on = 'inspection_id',
              how = 'left')


## loop through each of the features considered in Kang et al. Table 1
subset = np.array(dta_analysis.columns)[np.array(dta_analysis.columns)!='inspection_penalty_score']
subset=subset[subset!='inspection_id']
counter = 0
feature_set = np.array([
                        [u'review_count'], # 0
                        [u'non_positive_review_count'], # 1
                        subset[np.in1d(subset,cuisine_df.columns[cuisine_df.columns!='inspection_id'])], # 2
                        subset[np.in1d(subset,zip_code_df.columns[zip_code_df.columns!='inspection_id'])], # 3
                        [u'average_review_rating'], # 4
                        ['inspection_average_prev_penalty_scores', 'inspection_prev_penalty_score']#, # 5
                        ])


for subset in feature_set:
    print 'counter: ', counter
    print 'features: ', subset ## ignore this value for 9
    kf = KFold(n_splits=10,     # 10-fold CV is used in paper
          shuffle = True,      # assuming they randomly select train/test
          random_state = 123)  # random.seed for our own internal replication purposes
    mse_features = []
    m = 0
    #for k, (train, test) in enumerate(k_fold):
    for train, test in kf.split(dta_analysis):
    #for train, test in kf.split(dta_analysis.loc[:,subset], dta_analysis.inspection_penalty_score):
        print m
        x_train = dta_analysis.iloc[train,:][subset]
        if m == 0:
            print 'columns used as sanity check: ',x_train.columns
            print len(x_train.columns)
            print len(subset)
        y_train = dta_analysis.inspection_penalty_score.iloc[train]
        x_test = dta_analysis.iloc[test,:][subset]
        y_test = dta_analysis.inspection_penalty_score.iloc[test]
        model.fit(x_train,
                  np.ravel(y_train))
        y_predict = model.predict(x_test)
        mse_features.append(mean_squared_error(y_test, y_predict))
        print mse_features
        m = m+1
        if CV_model:
            writer.writerow( (m, subset, train,model.best_params_.values(), 
                              mean_absolute_error(y_test, y_predict),
                              mean_squared_error(y_test, y_predict)))
        else:
            writer.writerow( (m, subset, train,'0',#''model.best_params_.values(),
                          mean_absolute_error(y_test, y_predict),
                          mean_squared_error(y_test, y_predict)))
    print ''
 
