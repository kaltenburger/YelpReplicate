# about: we run this code to subset to routine inspections only
rm(list=ls())

setwd('~/Dropbox/YelpReplicate/code/GithubWWW2019/YelpReplicate/code/0_original_analysis_restaurant_hygiene/dataset/')
kang_df <- read.csv('instances_mergerd_seattle.csv')
head(kang_df)
kang_df_inspections <- kang_df$inspection_id


full_DB <- read.csv('../../../../../../../King_DB/3Master Datasets/master_inspections.csv')
colnames(full_DB)
head(full_DB)
full_DB_inspections <- full_DB$inspection_serial_num
length(full_DB_inspections)
length(unique(full_DB_inspections)) # just spotcheck

length(kang_df_inspections)
#[1] 13299

length(which(kang_df_inspections %in% full_DB_inspections))
#[1] 13297

subset_full_idx <- which(full_DB$inspection_serial_num %in% kang_df_inspections)

table(full_DB$type_description[subset_full_idx])
#Complaint Investigation  Consultation/Education - Field                     Peer Review 
#0                               2                               0 
#Permit Investigation             Plan Review - Field        Pre-Occupancy Inspection 
#0                               0                               0 
#Return Inspection Routine Inspection/Field Review                         UNKNOWN 
#682                           12613                               0 


# let's drop the  682 return inspections, 2 education trainings, and the 2 original Kang observations
# which do not have matches in the full_DB
full_DB_kang_subset <- full_DB[subset_full_idx,]

subset_routine_inspection <- full_DB_kang_subset$inspection_serial_num[full_DB_kang_subset$type_description=='Routine Inspection/Field Review']


kang_df_routine <- kang_df[which(kang_df$inspection_id %in% subset_routine_inspection),]
nrow(kang_df_routine)
#write.csv(kang_df_routine, '../../../data/instances_mergerd_seattle_routine_only.csv', row.names=FALSE)


