# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)
library(rules)
library(baguette)
registerDoMC(detectCores())
getDoParWorkers()

set.seed(3013)

folds_pca <- read_rds('model selection/model_objects/folds_pca.rds')

# svm_tuned_pca
# nn_tuned_pca
# bt_tuned_pca


svm_tuned_pca<- read_rds("model selection/model_objects/svm_tuned_pca.rds")

svm_tuned_pca %>% collect_predictions()



folds_pca$splits[[1]] %>% analysis()

