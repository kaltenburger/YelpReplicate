## YelpReplicate
Replication code for WWW'19 Paper ``Is Yelp Actually Cleaning Up the Restaurant Industry? A Re-Analysis on the Relative Usefulness of Consumer Reviews''

### Documentation
This repository contains all the corresponding code to recreate tables and figures corresponding to the re-analysis portion in ``Is Yelp Actually Cleaning Up the Restaurant Industry? A Re-Analysis on the Relative Usefulness of Consumer Reviews''.

### Directions
1. Save the raw Yelp data called 'instances_mergerd_seattle.csv' from Kang et al. 2013 <a href="http://www3.cs.stonybrook.edu/~junkang/hygiene/">[link]</a> to the data folder in this repo.

2. How to run code where folder organization is briefly described below: 
  * ./data/: includes a link to the original data subsetted to routine-inspections only as used for the re-analysis portion of the paper and includes folders for storing features from the routine-only subsetted dataset.
  * ./code/0_original_analysis_restaurant_hygiene/: placeholder for including original Kang et al. code.
  * ./code/1_re-analysis/: code for re-analysis.
  * ./figs/: folder for all figures
  * ./appendix/: folder contains document with supplemental figures that would not fit in main paper.


All code was writted and tested for Python 2.712

### Authors
* Kristen M. Altenburger, kaltenb@stanford.edu
* Daniel E. Ho, dho@law.stanford.edu
