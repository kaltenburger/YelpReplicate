## modified: 10/4/2018 to be limited to only routine inspections
## create features except doc2vec

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
features = ['inspection_id','restaurant_id','inspection_average_prev_penalty_scores', 'inspection_prev_penalty_score',
           'cuisines', 'inspection_penalty_score',
            'zip_code', 'review_count', 'non_positive_review_count', 
            'average_review_rating', 'review_contents']
dta = dta[features]
dta.to_pickle('../../data/features_pkl/dta.pickle')


##
## Create tf-idf features
##

## standard text cleaning functions.
## stem
def stemming(text):
    ps = PorterStemmer()
    return(' '.join(ps.stem(word) for word in text.split(' ')))


## remove numbers
def remove_number(text):
    return(re.sub('\d+','',text))

## remove punctuation
def remove_punct(text):
    regex_pat = re.compile(r'[^a-zA-Z\s]', 
                           flags=re.IGNORECASE)
    return(re.sub(regex_pat, '', text))


docs = map(gensim.parsing.stem_text, dta.review_contents)
docs = map(remove_number, docs)
docs = map(remove_punct, docs)


from sklearn.feature_extraction.text import TfidfVectorizer

## unigram features
vec = TfidfVectorizer(ngram_range = (1,1),
                      lowercase = True,
                     stop_words = 'english',
                     min_df = 0.025)
X = vec.fit_transform(docs)
unigram_df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())


## bigram features
vec = TfidfVectorizer(ngram_range = (2,2),
                      lowercase = True,
                     stop_words = 'english',
                     min_df = 0.025)
X = vec.fit_transform(docs)
bigram_df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
uni_bigram_df = pd.concat([unigram_df, bigram_df], axis = 1)



## add inspection-id
unigram_df['inspection_id'] = dta.inspection_id
bigram_df['inspection_id'] = dta.inspection_id
uni_bigram_df['inspection_id'] = dta.inspection_id

unigram_df.to_pickle('../../data/features_pkl/unigram.pickle')
bigram_df.to_pickle('../../data/features_pkl/bigram.pickle')
uni_bigram_df.to_pickle('../../data/features_pkl/uni_bigram.pickle')

xs = re.sub('\s+', '', dta.cuisines[0])
a  = np.array(ast.literal_eval(xs))

def convert_cuisines(column):
    xs = re.sub('\s+', '', column)
    a  = np.array(ast.literal_eval(xs))
    return(a)

dta['cuisines_converted'] = map(convert_cuisines, dta.cuisines)

tmp = pd.DataFrame(dta.cuisines_converted.values.tolist(), index = dta.index)

## at most - any restaurant as 4 distinct classifications
cuisines = np.unique(np.concatenate((np.unique(tmp[0]),
                    np.unique(tmp[1]),
                    np.unique(tmp[2]),
                    np.unique(tmp[3]),
                    np.unique(tmp[4]))))

## create cuisine indicator-feature
cuisine_df = pd.DataFrame(np.zeros(shape=(len(dta.cuisines_converted), len(cuisines))))
cuisine_df.columns = cuisines
cuisine_df.head()


for j in range(np.shape(cuisine_df)[0]):
    cuisine_df.iloc[j]=np.in1d(np.array(cuisine_df.columns), np.array(tmp.iloc[j])[np.array(tmp.iloc[j])!=None])+0

cuisine_df.head()

## drop none column
cuisine_df = cuisine_df[np.array(cuisine_df.columns[1:100])]


## we drop the restaurant feature since it's 1 for all obs.
cuisine_df.drop('Restaurants', axis =1, inplace = True)
cuisine_df['inspection_id'] = dta.inspection_id
cuisine_df.to_pickle('../../data/features_pkl/cuisine.pickle')


## Zip Code Features
dta.zip_code = dta.zip_code.astype('str')
print len(np.unique(dta.zip_code))
zip_code_df = pd.get_dummies(dta.zip_code, columns = ['zip_code'],
                        drop_first = True, dummy_na = False)#.iloc[:,1:] # k-1 coding
zip_code_df['inspection_id'] = dta.inspection_id
zip_code_df.to_pickle('../../data/features_pkl/zip_code_df.pickle')

