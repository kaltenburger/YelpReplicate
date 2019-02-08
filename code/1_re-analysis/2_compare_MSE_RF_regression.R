#2/3/2019

# About: This code is for 2nd part of paper, redoing the original Yelp-prediction analysis where
# we're setting up as a regression problem trying to predict inspection scores.

rm(list=ls())
setwd('~/Dropbox/YelpReplicate/code/GithubWWW2019/YelpReplicate/code/1_re-analysis/results/')
#setwd('~/Dropbox (Stanford Law School)/YelpReplicate/code/GithubWWW2019/YelpReplicate/code/1_re-analysis/results/')


##
## Random Gridsearch
##
baseline <- read.csv('kang_results_baseline.csv')
random_other_RF <- read.csv('kang_results_regression_otherFeatures_SEED_RF_search_random.csv')
random_nembed_RF <- read.csv('kang_results_regression_februaryCVuniversalembedding_SEED_ndim_100_window_5_min_count_3RF_search_random.csv') #'kang_results_regression_octoberCVuniversalembedding_ndim_100_window_5_min_count_3RF_search_random.csv')
random_nembed_RF2 <- read.csv('kang_results_regression_februaryCVuniversalembedding_SEED_with_steps_ndim_100_window_5_min_count_3RF_search_random.csv') #'kang_results_regression_octoberCVuniversalembedding_ndim_100_window_5_min_count_3RF_search_random.csv')

random_nembed_RF$feature <- 'embed'
randomRF <- rbind(random_other_RF, random_nembed_RF)

## first row of Table 2
wilcox.test(baseline$mse, # WITHOUT Yelp
            random_nembed_RF$mse,  # WITH Yelp
            alternative = 'greater', # want to test if WITH yelp feature has smaller MSE than WITHOUT yelp, meaning difference would be > 0 
            paired = TRUE,
            exact = FALSE)

#> mean(baseline$mse)
#[1] 222.1343
#> mean(random_nembed_RF$mse)
#[1] 209.9276

## Read in Files for Comparing with/without Yelp Features
compare_with_without_embedding <- read.csv('kang_results_regression_CV_model_SEED_True_all_and_embed_100_window_5_min_count_3RF_search_random.csv')

yelp_vs_zip <- read.csv('kang_results_regression_CV_model_SEED_True_Yelp_zip_code_100_window_5_min_count_3RF_search_random.csv')
yelp_vs_cuisine <- read.csv('kang_results_regression_CV_model_SEED_True_Yelp_cuisine_100_window_5_min_count_3RF_search_random.csv')
yelp_vs_zip_vs_cuisine <- read.csv('kang_results_regression_CV_model_SEED_True_Yelp_cuisine_and_zip_100_window_5_min_count_3RF_search_random.csv')
yelp_vs_inspection_zip <- read.csv('kang_results_regression_CV_model_SEED_True_Yelp_inspection_history_and_zip_100_window_5_min_count_3RF_search_random.csv')
yelp_vs_inspection_vs_cuisine <- read.csv('kang_results_regression_CV_model_SEED_True_Yelp_inspection_history_and_cuisine_100_window_5_min_count_3RF_search_random.csv')
yelp_vs_inspection <- read.csv('kang_results_regression_CV_model_SEED_True_Yelp_inspection_history_100_window_5_min_count_3RF_search_random.csv')
yelp_vs_all <- read.csv('kang_results_regression_CV_model_SEED_True_Yelp_cuisine_and_zip_and_inspections_100_window_5_min_count_3RF_search_random.csv')
#files <- list(yelp_vs_zip,
#              yelp_vs_cuisine,
#              yelp_vs_inspection,
#              yelp_vs_zip_vs_cuisine,
#              yelp_vs_inspection_zip,
#              yelp_vs_inspection_vs_cuisine,
#              yelp_vs_all) 


files <- list(yelp_vs_zip,
              yelp_vs_cuisine,
              yelp_vs_inspection,
              yelp_vs_zip_vs_cuisine,
              yelp_vs_inspection_zip,
              yelp_vs_inspection_vs_cuisine,
              yelp_vs_all) 


for( i in 1:length(files)){
  file_i <- files[[i]]
  # feature-X only
  print('feature-X only: ')
  print(mean(file_i$mse[file_i$feature==unique(file_i$feature)[1]]))
  print(as.character(unique(file_i$feature)[1]))
  
  # feature-X + yelp
  print('feature-X AND Yelp embedding: ')
  print(mean(file_i$mse[file_i$feature==unique(file_i$feature)[2]]))
  print(as.character(unique(file_i$feature)[2]))
  
  ## wilcox paired test
  print(wilcox.test(file_i$mse[file_i$feature==unique(file_i$feature)[1]], # WITHOUT Yelp
              file_i$mse[file_i$feature==unique(file_i$feature)[2]],  # WITH Yelp
              alternative = 'greater', # want to test if WITH yelp feature has smaller MSE than WITHOUT yelp, meaning difference would be > 0 
                               paired = TRUE,
              exact = FALSE))
  print('')
}



## 1) Overall -- this is what's reported in the text
# all w/o embedding, review counts, or ratings
mean(compare_with_without_embedding$mse[compare_with_without_embedding$feature==unique(compare_with_without_embedding$feature)[1]])
#[1] 183.6095
as.character(unique(compare_with_without_embedding$feature)[1])

# all w/ embedding
mean(compare_with_without_embedding$mse[compare_with_without_embedding$feature==unique(compare_with_without_embedding$feature)[2]])
#[1] 185.6955
as.character(unique(compare_with_without_embedding$feature)[2])

wilcox.test(compare_with_without_embedding$mse[compare_with_without_embedding$feature==unique(compare_with_without_embedding$feature)[1]], # without yelp
            compare_with_without_embedding$mse[compare_with_without_embedding$feature==unique(compare_with_without_embedding$feature)[2]], # with yelp
            alternative='greater', #two.sided, less, greater
            paired = TRUE, 
            exact=FALSE)

add.density <- function(xx = baseline$mse, at = 10, my.col=rgb(0,0,0,0.4), scale=3){
    dens <- density(xx)
    polygon(c(dens$x, rev(dens$x)),
            at + c(scale*dens$y, -rev(scale*dens$y)), border=NA, col=my.col)
}

## Re-design of figure 
cex_pt = 1.5
pdf("../../../figs/RF_hyperparameter-tuned_final2_SEED.pdf", width=3.5,height=4)
layout(matrix(c(1,1),2,2))
par(mar=c(3,0.5,2,0.5), mgp=c(1.5,0.5,0),tcl=-0.3)
plot(baseline$mse, rep(1.5, 10), pch = 16, 
     col=rgb(0.5,0.5,0.5,0.5), xlim = c(150, 300), ylim = c(-2.25,1.6), ylab='', yaxt='n',
     xlab = 'MSE', main = 'Feature Prediction in Regression',bty="n",
     cex = cex_pt, type='n')
abline(v=mean(baseline$mse), col = rgb(0.5,0.5,0.5,0.5), lty=2)
add.density(baseline$mse, at=1.5)
points(mean(baseline$mse), 1.5, pch = '|', 
       cex=cex_pt)

count <- 0

labels <- c('Baseline','Review count', 'Neg. review count', 'Cuisine',
            'ZIP code', 'Avg. review rating', 'Inspection history','Yelp review text')

for(j in unique(randomRF$feature)){
    #print(as.character(j))
  #print('')
  if(labels[count+2]=="Yelp review text"){
      my.col <- rgb(0,0,1,0.4)
  } else if(labels[count+2]=="Inspection history"){
      print("EWARS")
      my.col <- rgb(1,0,0,0.4)
  } else {my.col <- rgb(0,0,0,0.4)}
  add.density(randomRF$mse[randomRF$feature==j], at=1-0.5*count, my.col=my.col)
  points(mean(randomRF$mse[randomRF$feature== j]), 1-0.5*count, pch = '|', 
         cex=cex_pt, col=my.col)
  count <- count+1
}

unique(randomRF$feature)
            #'unigram', 'bigram', 'unigram+bigram')
text(rep(250, length(labels)), c(1.5,1,0.5,0,-0.5,-1, -1.5,-2),#, -2.5, -3,-3.5), 
     labels, cex = 0.9, pos=4)
dev.off()



## compare word2vec with other parameter settings
w2v_100_3_3 <- read.csv('kang_results_regression_februaryCVuniversalembedding_SEED_ndim_100_window_3_min_count_3RF_search_random.csv')
w2v_100_3_5 <- read.csv('kang_results_regression_februaryCVuniversalembedding_SEED_ndim_100_window_3_min_count_5RF_search_random.csv')
w2v_100_5_3<- read.csv('kang_results_regression_februaryCVuniversalembedding_SEED_ndim_100_window_5_min_count_3RF_search_random.csv')
w2v_200_3_3 <- read.csv('kang_results_regression_februaryCVuniversalembedding_SEED_ndim_200_window_3_min_count_3RF_search_random.csv')
w2v_200_3_5<- read.csv('kang_results_regression_februaryCVuniversalembedding_SEED_ndim_200_window_3_min_count_5RF_search_random.csv')
w2v_tfidf<- read.csv('kang_results_regression_CV_model_True_tfidf_RF_search_random.csv')


cex_pt = 1.5
pdf("../../../figs/doc2vec_different_params_final.pdf", width=4,height=5)
layout(matrix(c(1,1),2,2))
par(mar=c(3,0.5,2,0.5), mgp=c(1.5,0.5,0),tcl=-0.3)
plot(baseline$mse, rep(1.5, 10), pch = 16, 
     col=rgb(0.5,0.5,0.5,0.5), xlim = c(150, 300), ylim = c(-1.5,1.6), ylab='', yaxt='n',
     xlab = 'MSE', main = 'Compare Doc2Vec Settings',bty="n",
     cex = cex_pt)
points(mean(baseline$mse), 1.5, pch = 16, 
       col=rgb(1,0.5,0.5,1), cex=cex_pt)
abline(v=mean(baseline$mse), col = rgb(0.5,0.5,0.5,0.5), lty=2)
text(250, 1.5,
     'baseline', cex = 0.9, pos=4)

points(w2v_100_3_3$mse, rep(1, 10), pch = 16, 
       col=rgb(0.5,0.5,0.5,0.5), cex =cex_pt)
points(mean(w2v_100_3_3$mse), 1, pch = 16, 
       col=rgb(1,0.5,0.5,1), cex=cex_pt)
text(250, 1,
     'ndim=100 \nwindow=3\nmin_count=3', cex = 0.9, pos=4)

points(w2v_200_3_5$mse,
       rep(1-0.5*4, 10), pch = 16, col=rgb(0.5,0.5,0.5,0.5), cex=cex_pt)
points(mean(w2v_200_3_5$mse), 1-0.5*4, pch = 16, 
       col=rgb(1,0.5,0.5,1), cex=cex_pt)
text(250, -1,
     'ndim=200 \nwindow=3\nmin_count=5', cex = 0.9, pos=4)


points(w2v_200_3_3$mse,
       rep(1-0.5*3, 10), pch = 16, col=rgb(0.5,0.5,0.5,0.5), cex=cex_pt)
points(mean(w2v_200_3_3$mse), 1-0.5*3, pch = 16, 
       col=rgb(1,0.5,0.5,1), cex=cex_pt)
text(250, -0.5,
     'ndim=200 \nwindow=3\nmin_count=3', cex = 0.9, pos=4)


points(w2v_100_5_3$mse,
         rep(1-0.5*2, 10), pch = 16, col=rgb(0.5,0.5,0.5,0.5), cex=cex_pt)
points(mean(w2v_100_5_3$mse), 1-0.5*2, pch = 16, 
         col=rgb(1,0.5,0.5,1), cex=cex_pt)
text(250, 0,
     'ndim=100 \nwindow=5\nmin_count=3', cex = 0.9, pos=4)


points(w2v_100_3_5$mse,
       rep(1-0.5*1, 10), pch = 16, col=rgb(0.5,0.5,0.5,0.5), cex=cex_pt)
points(mean(w2v_100_3_5$mse), 1-0.5*1, pch = 16, 
       col=rgb(1,0.5,0.5,1), cex=cex_pt)
text(250, 0.5,
     'ndim=100 \nwindow=3\nmin_count=5', cex = 0.9, pos=4)

points(w2v_tfidf$mse,
       rep(1-0.5*5, 10), pch = 16, col=rgb(0.5,0.5,0.5,0.5), cex=cex_pt)
points(mean(w2v_tfidf$mse), 1-0.5*5, pch = 16, 
       col=rgb(1,0.5,0.5,1), cex=cex_pt)
text(250, 1-0.5*5,
     'TF-IDF (unigrams)', cex = 0.9, pos=4)
dev.off()
