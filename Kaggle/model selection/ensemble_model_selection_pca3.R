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
pca_train <- read_csv('data/pca_train3.csv') %>% mutate(x516 = as_factor(x516))
pca_test <- read_csv('data/pca_test3.csv') %>% mutate(x516 = as_factor(x516))

bt_tuned_pca <- read_rds('model selection/model_objects/bt_tuned_pca3.rds')
nn_tuned_pca <- read_rds('model selection/model_objects/nn_tuned_pca3.rds')
svm_tuned_pca <- read_rds('model selection/model_objects/svm_tuned_pca3.rds')
bmlp_tuned_pca <- read_rds('model selection/model_objects/bmlp_tuned_pca3.rds')
cube_tuned_pca <- read_rds('model selection/model_objects/cube_tuned_pca3.rds')
en_tuned_pca <- read_rds('model selection/model_objects/en_tuned_pca3.rds')
rf_tuned_pca <- read_rds('model selection/model_objects/rf_tuned_pca3.rds')




ensemble_model_pca <- 
  stacks() %>%
  add_candidates(svm_tuned_pca) %>%
  add_candidates(nn_tuned_pca) %>%
  add_candidates(bt_tuned_pca) 
  #add_candidates(bag_mars_tuned_pca) %>%
  # add_candidates(bag_tree_tuned_pca) %>%
  #add_candidates(bmlp_tuned_pca) %>%
  #add_candidates(cube_tuned_pca) %>%
  #add_candidates(en_tuned_pca) 
#add_candidates(mars_tuned_pca) %>%
# add_candidates(rf_tuned_pca)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed)
set.seed(9876)
ensemble_st_pca <-
  ensemble_model_pca %>%
  blend_predictions(penalty = blend_penalty)


# ensemble_workflow <- workflow() %>%
#   add_model(ensemble_st)

ensemble_fit_pca <-
  ensemble_st_pca %>%
  fit_members()


theme_set(theme_bw())
autoplot(ensemble_fit_pca)
autoplot(ensemble_fit_pca, type = "members")
autoplot(ensemble_fit_pca, type = "weights")



pca_train%>%
  bind_cols(predict(ensemble_fit_pca, .)) %>%
  select(.pred,
         y
  ) %>%
  mutate(.pred = exp(.pred),
         y = exp(y),
         .pred = .pred + 1.5
  ) %>%
  rmse(truth = y, 
       estimate = .pred
  )

train_predictions <- pca_train%>%
  bind_cols(predict(ensemble_fit_pca, .)) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )


ensemble_test_pca <-
  pca_test %>%
  bind_cols(predict(ensemble_fit_pca, .)) %>%
  select(id, .pred
  ) %>%
  mutate(.pred = exp(.pred),
         .pred = .pred + 1.7,
         
  )


prediction56 <- ensemble_test_pca %>% rename(y = .pred)

write_csv(prediction56, 'predictions/prediction56.csv')




