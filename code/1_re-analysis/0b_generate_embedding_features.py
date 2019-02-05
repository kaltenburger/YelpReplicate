#10/8/2018
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
from random import seed
seed(123)

execfile('../python_libraries.py')
dta = pd.read_csv('../instances_mergerd_seattle_routine_only.csv') #'instances_mergerd_seattle.csv')
print np.shape(dta)
print dta.columns

## note there's 1-observation with -1 inspection score. We set to 0.
np.sum(dta.inspection_prev_penalty_score==-1)
dta.inspection_penalty_score[dta.inspection_penalty_score==-1]=0
dta.inspection_average_prev_penalty_scores[dta.inspection_average_prev_penalty_scores==-1]=0
dta.inspection_prev_penalty_score[dta.inspection_prev_penalty_score==-1]=0
np.sum(dta.inspection_prev_penalty_score==-1)


## Subset to relevant features
features = ['inspection_id','restaurant_id', 'review_contents','inspection_penalty_score']
dta = dta[features]

#https://fzr72725.github.io/2018/01/14/genism-guide.html
tagged = dta.apply(lambda r: TaggedDocument(words=gensim.utils.simple_preprocess(r['review_contents']),
                                            tags=[r.inspection_id]),
                   axis=1)
documents_train = tagged.values


ndim = 100 #100, 200
window=3  #try smaller
min_count=5 #3-5
model = Doc2Vec(documents_train, vector_size=ndim, window=window, min_count=min_count, seed=123, workers = 1)#, workers=4)

#model = Doc2Vec(documents_train, vector_size=ndim, window=window, min_count=min_count, workers=4)


#fname = get_tmpfile("my_doc2vec_model")
#model.save(fname)
#model = Doc2Vec.load(fname)
#model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

#x_train_d2v = map(model.infer_vector,cleaned_review_df['texts'])
#embedding_df = pd.DataFrame(np.matrix(x_train_d2v))
#embedding_df['inspection_id']=cleaned_review_df.inspection_id
steps = 20 #None
def vec_for_learning(doc2vec_model, tagged_docs):
    sents = tagged_docs.values
    if steps == None:
        targets, regressors = zip(*[(doc.tags[0], doc2vec_model.infer_vector(doc.words)) for doc in sents])
    else:
        targets, regressors = zip(*[(doc.tags[0], doc2vec_model.infer_vector(doc.words, steps = steps)) for doc in sents])
    return targets, regressors

inspection_id, X_train = vec_for_learning(model, tagged)

id = pd.DataFrame({'inspection_id':inspection_id
                  })
embed = pd.DataFrame(np.matrix(X_train))
embedding_df = pd.concat((id, embed),1)

if steps == None:
    embedding_df.to_pickle('../../data/features_pkl/embedding_df_ndim_'+ str(ndim)+'_window_'+str(window)+'_min_count_'+str(min_count) + '_steps_'+ str(steps)+ '.pickle')
else:
    embedding_df.to_pickle('../../data/features_pkl/embedding_df_ndim_'+ str(ndim)+'_window_'+str(window)+'_min_count_'+str(min_count) + '.pickle')
