#11/1/2018
#universal embedding
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV


execfile('../python_libraries.py')
dta = pd.read_pickle('../../data/features_pkl/dta.pickle')

ndim = 100 #100, 200
window=5  #try smaller
min_count=3 #3-5
embedding_df = pd.read_pickle('../../data/features_pkl/embedding_df_ndim_'+ str(ndim)+'_window_'+str(window)+'_min_count_'+str(min_count) + '.pickle')



# other features
cuisine_df = pd.read_pickle('../../data/features_pkl/cuisine.pickle')
dta = pd.read_pickle('../../data/features_pkl/dta.pickle')
zip_code_df = pd.read_pickle('../../data/features_pkl/zip_code_df.pickle')


CV_model = True # False True
model_type = 'RF' # 'RF' 'SVR'
search_type = 'random' # 'random' 'grid'

## Predictive Models to Try
kf = KFold(n_splits = 3, shuffle = True, random_state = 5)


if CV_model:
    if model_type=='RF':
        max_depth = [None, 5, 10]
        min_samples_leaf = [0.0005, 0.01, 0.05, 0.1]
        min_samples_split = [2, 5, 10]
        n_estimators = [100, 200, 500]
        max_features = [None, 0.25, 0.5, 0.75]
        param_grid = {'max_features': max_features,
            'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                    'n_estimators':n_estimators}
        clf = sklearn.ensemble.RandomForestRegressor(random_state = 123)
        if search_type == 'grid':
            model_full = GridSearchCV(estimator = clf,
                         param_grid = param_grid,
                         cv = kf, verbose=0,
                         n_jobs = 3, scoring = 'neg_mean_squared_error')
        if search_type == 'random':
            model_full = RandomizedSearchCV(estimator = clf, random_state = 123,
                                      param_distributions = param_grid,
                                      cv = kf, verbose=0, n_iter = 25,
                                      n_jobs = 5, scoring = 'neg_mean_squared_error')
    if model_type =='SVR':
        Cs = [0.001, 0.01, 0.1, 1, 10, 100]
        param_grid = {'C': Cs}
        model_full = GridSearchCV(estimator = SVR(), param_grid = param_grid, cv=kf,
                             verbose = 0, n_jobs = 3, scoring = 'neg_mean_squared_error')
else:
    if model_type=='RF':
        model_full = sklearn.ensemble.RandomForestRegressor(n_estimators=200)
    if model_type=='SVR':
        model_full = SVR(C=100)


# save output
file_output = open('./results/kang_results_regression_CV_model_SEED_'+str(CV_model)+ '_Yelp_inspection_history_and_zip_'+str(ndim)+'_window_'+str(window)+'_min_count_'+str(min_count)+ model_type+'_search_'+search_type+'.csv', 'wt')
writer = csv.writer(file_output)
writer.writerow( ('iteration','feature', 'train_split', 'model_params', 'mae', 'mse'))



dta_analysis_tmp = dta.copy()

dta_analysis = dta_analysis_tmp[['inspection_id', 'inspection_average_prev_penalty_scores',
                                 'inspection_prev_penalty_score', 'inspection_penalty_score']]
# merge embeddings
dta_analysis = dta_analysis.merge(embedding_df,
              on = 'inspection_id',
              how = 'left')

# merge other features
#dta_analysis = dta_analysis.merge(cuisine_df,
#              on = 'inspection_id',
#              how = 'left')
dta_analysis = dta_analysis.merge(zip_code_df,
              on = 'inspection_id',
              how = 'left')

subset = np.array(dta_analysis.columns)[np.array(dta_analysis.columns)!='inspection_penalty_score']
subset=subset[subset!='inspection_id']
counter = 0
feature_set_without_embedding = subset[~np.in1d(subset,embedding_df.columns[embedding_df.columns!='inspection_id'])]
feature_set_without_embedding = feature_set_without_embedding[~np.in1d(feature_set_without_embedding,[ 'review_count', 'non_positive_review_count', 'average_review_rating'])]

feature_set = [[feature_set_without_embedding], [subset]]

for subset in feature_set:
    print 'counter: ', counter
    print 'features: ', subset ## ignore this value for 9
    kf = KFold(n_splits=10,     # 10-fold CV is used in paper
          shuffle = True,      # assuming they randomly select train/test
          random_state = 123)  # random.seed for our own internal replication purposes
    mse_features = []
    m = 0
    for train, test in kf.split(dta_analysis):
        print m
        x_train = dta_analysis.iloc[train,:][subset[0]]
        if m == 0:
            print 'columns used as sanity check: ',x_train.columns
            print len(x_train.columns)
            print len(subset)
        y_train = dta_analysis.inspection_penalty_score.iloc[train]
        x_test = dta_analysis.iloc[test,:][subset[0]]
        y_test = dta_analysis.inspection_penalty_score.iloc[test]
        
        model_full.fit(x_train,
                  np.ravel(y_train))
        y_predict = model_full.predict(x_test)

        mse_features.append(mean_squared_error(y_test, y_predict))
        print mse_features
        m = m+1
        if CV_model:
            writer.writerow( (m, subset[0], train,model_full.best_params_.values(),
                              mean_absolute_error(y_test, y_predict),
                              mean_squared_error(y_test, y_predict)))
        else:
            writer.writerow( (m, subset[0], train,'0',
                          mean_absolute_error(y_test, y_predict),
                          mean_squared_error(y_test, y_predict)))
    print ''
 
