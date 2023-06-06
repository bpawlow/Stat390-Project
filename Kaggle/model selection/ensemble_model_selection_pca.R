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
train <- read_csv("data/train_cleaned.csv")  %>% mutate(x516 = as_factor(x516))
pca_train <- read_csv('data/pca_train.csv') %>% mutate(x516 = as_factor(x516))
pca_test <- read_csv('data/pca_test.csv') %>% mutate(x516 = as_factor(x516))
id <- (read_csv('data/test.csv'))$id %>% as_tibble()

bt_tuned_pca <- read_rds('model selection/model_objects/bt_tuned_pca.rds')
nn_tuned_pca <- read_rds('model selection/model_objects/nn_tuned_pca.rds')
svm_tuned_pca <- read_rds('model selection/model_objects/svm_tuned_pca.rds')
bag_mars_tuned_pca <- read_rds('model selection/model_objects/bag_mars_tuned_pca.rds')
bag_tree_tuned_pca <- read_rds('model selection/model_objects/bag_tree_tuned_pca.rds')
bmlp_tuned_pca <- read_rds('model selection/model_objects/bmlp_tuned_pca.rds')
cube_tuned_pca <- read_rds('model selection/model_objects/cube_tuned_pca.rds')
en_tuned_pca <- read_rds('model selection/model_objects/en_tuned_pca.rds')
mars_tuned_pca <- read_rds('model selection/model_objects/mars_tuned_pca.rds')
rf_tuned_pca <- read_rds('model selection/model_objects/rf_tuned_pca.rds')




ensemble_model_pca <- 
  stacks() %>%
  add_candidates(svm_tuned_pca) %>%
  add_candidates(nn_tuned_pca) %>%
  add_candidates(bt_tuned_pca) %>%  
 # add_candidates(bag_mars_tuned_pca) %>%
 # add_candidates(bag_tree_tuned_pca) %>%
  #add_candidates(bmlp_tuned_pca) %>%
  #add_candidates(cube_tuned_pca) %>%
  #add_candidates(en_tuned_pca) %>% 
 # add_candidates(mars_tuned_pca) %>%
  add_candidates(rf_tuned_pca)

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
         .pred = .pred +1.8
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
  bind_cols(predict(ensemble_fit_pca, .))



 prediction47 <- cbind(id, ensemble_test_pca)%>%
   select(value, .pred) %>%
   rename(y = .pred, id = value) %>%
   mutate(y = exp(y),
          y = y + 2
          )


 write_csv(prediction47, 'predictions/final_prediction2.csv')




