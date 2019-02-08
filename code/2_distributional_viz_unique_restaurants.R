# 1/27/2019
# Create Figure 1: distribution of scores and what subsets of the data were used in the original analysis
# Create Figure 2: distribution of HYGIENIC ID-frequencies in original analysis and in one random sample instance

rm(list=ls())
set.seed(123)

setwd('~/Dropbox/YelpReplicate/code/GithubWWW2019/YelpReplicate/code/0_original_analysis_restaurant_hygiene/hygiene_anaylysis/a_original_analysis/')
#setwd('~/Dropbox (Stanford Law School)/YelpReplicate/code/GithubWWW2019/YelpReplicate/code/0_original_analysis_restaurant_hygiene/hygiene_anaylysis/a_original_analysis/')

## get paper's original n=612 observations
original_df <- read.csv('./output/ids_seattle_yelp_00.csv',
                        header = FALSE)

## one other spot-check
#original_df <- read.csv('../old/a_original_analysis_execution_type_one_testing/output/ids_seattle_yelp_.csv', header = FALSE)
#head(original_df)
#nrow(original_df[307:612,])
#par(mar=c(3,3,2,0.5), mgp=c(1.5,0.5,0),tcl=-0.3)
#tN <- table(as.array(table(as.character(original_df[307:612,'V1']))))
#tN <- as.data.frame(tN)
#tN$Freq <- tN$Freq/sum(tN$Freq)
#tN$Var1 <- as.character(tN$Var1)

#barplot(tN$Freq, space = 1.5, col = 'gray', border = 'white', names.arg = tN$Var1, #ylim = c(0,550),
#        main = 'Original Analysis \n Hygienic Restaurants', ylab = 'Density', 
#        xlab = 'Occurrences of Restaurant IDs', ylim = c(0, 1))#, xlim = c(1,5))

full_df <- read.csv('../../dataset/instances_mergerd_seattle.csv')
sum(full_df$inspection_penalty_score==-1)
sum(full_df$inspection_penalty_score==0)

## Descriptive stats reported in Sec 2
mean(full_df$inspection_penalty_score > 50)
#[1] 0.02300925

mean(full_df$inspection_penalty_score == 0)
#[1] 0.3436349

sum(full_df$inspection_penalty_score == 0 | full_df$inspection_penalty_score == -1)

mean(full_df$inspection_penalty_score > 0 & full_df$inspection_penalty_score <= 50)
#[1] 0.6332807




## Histogram of inspection scores
## Recode for visibility
cutoff <- 70 
yy <- full_df$inspection_penalty_score
yy[yy==-1] <- 0
yy[yy>=cutoff] <- cutoff

## Set breaks and colors
my.breaks <- c(0,1,seq(5,cutoff+5,by=5))
my.col <- c(rgb(0,1,0,0.6),
            rep(rgb(0,0,0,0.4), sum(my.breaks<50)-1),
            rep(rgb(1,0,0,0.6), sum(my.breaks>=50)))

## Calculate MSE
lm1 <- lm(inspection_penalty_score ~ 1, data=full_df)
mse <- mean(lm1$residuals^2)

## Proportions
mean(yy==0)
mean(yy>50)
mean(yy<=50 & yy>0)

pdf("../../../../figs/outcome_distribution.pdf", width=4,height=3.5)
par(mar=c(3,3,2,0.5), mgp=c(1.5,0.5,0),tcl=-0.3)
par(lwd=0.1)
hist(yy, breaks=my.breaks, border="white", col=my.col,
     main="Inspection Scores", xlab="Score", freq = F)
abline(v=50, lty=2, col=rgb(0,0,0,0.4),lwd=1)
abline(v=1, lty=2, col=rgb(0,0,0,0.4),lwd=1)
text(10,0.3,"\"Hygienic\"", cex=0.8, col="green")
text(65,0.05,"\"Unhygienic\"",cex=0.8,col="red")
text(20,0.08,"Discarded data",cex=0.8)
text(cutoff/2,0.35,paste0("MSE = ", round(mse,2)),cex=0.8, xpd=T)
dev.off()

#sum(tN$Freq)-tN$Freq[1] # proportion repeated duplicates
#[1] 0.7157895


pdf("../../../../figs/distribution_hygienic_306_original_vs_random_final.pdf", width=6,height=2.75)
layout(matrix(c(1,2),1,2))
par(mar=c(3,3,2,0.5), mgp=c(1.5,0.5,0),tcl=-0.3)
tN <- table(as.array(table(as.character(original_df[307:612,'V1']))))
tN <- as.data.frame(tN)
tN$Freq <- tN$Freq/sum(tN$Freq)
tN$Var1 <- as.character(tN$Var1)

barplot(tN$Freq, space = 0.5, col = rgb(0,0,0,0.6), border = 'white', names.arg = tN$Var1, #ylim = c(0,550),
        main = 'Original Analysis', ylab = 'Proportion',#'Density', 
        xlab = 'ID frequency', ylim = c(0, 1))#, xlim = c(1,5))
abline(h=0, col = 'black')

full_df_sample <- full_df[full_df$inspection_penalty_score==0 | full_df$inspection_penalty_score==-1,]
full_df_sample <- full_df_sample[order(full_df_sample$inspection_penalty_score),]
sample_306 <- sample(full_df_sample$restaurant_id,size = 306, replace = FALSE)
par(mar=c(3,3,2,0.5), mgp=c(1.5,0.5,0),tcl=-0.3)

tN <- table(as.array(table(as.character(sample_306))))
tN <- as.data.frame(tN)
tN
tN$Freq <- tN$Freq/sum(tN$Freq)
tN$Var1 <- as.character(tN$Var1)
1-tN$Freq[1]

## manually check here where to add
if(nrow(tN)==3){
  tN[4:9,"Var1"] <- c("4","5","6","7","8","9")
  tN[4:9,"Freq"] <- c(0,0,0,0,0,0)
}
if(nrow(tN)==4){
  tN[5:9,"Var1"] <- c("5","6","7","8","9")
  tN[5:9,"Freq"] <- c(0,0,0,0,0)
}


barplot(tN$Freq, space = 0.5, col = rgb(0,0,0,0.6), border = 'white', ylim = c(0,1), names.arg = tN$Var1, #xlim=c(0,5), #ylim = c(0,550),
        main = 'Random Sample',  
        xlab = 'ID frequency')
abline(h=0, col = 'black')
dev.off()

## try repeated sampling to get estimate of the number of duplicates
random_duplicates <- c()

for(j in 1:10000){
  full_df_sample <- full_df[full_df$inspection_penalty_score==0 | full_df$inspection_penalty_score==-1,]
  full_df_sample <- full_df_sample[order(full_df_sample$inspection_penalty_score),]
  sample_306 <- sample(full_df_sample$restaurant_id,size = 306, replace = FALSE)
  par(mar=c(3,3,2,0.5), mgp=c(1.5,0.5,0),tcl=-0.3)
  
  tN <- table(as.array(table(as.character(sample_306))))
  tN <- as.data.frame(tN)
  tN
  tN$Freq <- tN$Freq/sum(tN$Freq)
  tN$Var1 <- as.character(tN$Var1)
  random_duplicates <- c(random_duplicates,1-tN$Freq[1])
}
mean(random_duplicates)





## testing only sanity check
#dups <- read.csv('../c_random_hygienic/output/ids_seattle_yelp_00.csv',
#                        header = FALSE)
#layout(matrix(c(1,2),1,2))
#par(mar=c(3,3,2,0.5), mgp=c(1.5,0.5,0),tcl=-0.3)
#tN <- table(as.array(table(as.character(dups[307:612,'V1']))))
#tN <- as.data.frame(tN)
#tN$Freq <- tN$Freq/sum(tN$Freq)
#tN$Var1 <- as.character(tN$Var1)

#barplot(tN$Freq, space = 0.5, col = rgb(0,0,0,0.6), border = 'white', names.arg = tN$Var1, #ylim = c(0,550),
#        main = 'Prior Analysis', ylab = 'Proportion',#'Density', 
#        xlab = 'ID frequency', ylim = c(0, 1))#, xlim = c(1,5))
#abline(h=0, col = 'black')