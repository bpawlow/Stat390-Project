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
train <- read_csv('data/train.csv') %>% select(y) 
test <- read_csv('data/test.csv') %>% mutate(x516 = as_factor(x516))

bt_tuned <- read_rds('model selection/model_objects/bt_tuned3.rds')
nn_tuned <- read_rds('model selection/model_objects/nn_tuned3.rds')
svm_tuned <- read_rds('model selection/model_objects/svm_tuned3.rds')
bmlp_tuned <- read_rds('model selection/model_objects/bmlp_tuned3.rds')
cube_tuned <- read_rds('model selection/model_objects/cube_tuned3.rds')
en_tuned <- read_rds('model selection/model_objects/en_tuned3.rds')
rf_tuned <- read_rds('model selection/model_objects/rf_tuned3.rds')




ensemble_model <- 
  stacks() %>%
  add_candidates(svm_tuned) %>%
  add_candidates(nn_tuned) %>%
  add_candidates(bt_tuned) 
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
ensemble_st <-
  ensemble_model %>%
  blend_predictions(penalty = blend_penalty)


# ensemble_workflow <- workflow() %>%
#   add_model(ensemble_st)

ensemble_fit <-
  ensemble_st %>%
  fit_members()


theme_set(theme_bw())
autoplot(ensemble_fit)
autoplot(ensemble_fit, type = "members")
autoplot(ensemble_fit, type = "weights")



train%>%
  bind_cols(predict(ensemble_fit, .)) %>%
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


ensemble_test <-
  test %>%
  bind_cols(predict(ensemble_fit, .)) %>%
  select(id, .pred
  ) %>%
  mutate(.pred = exp(.pred),
         .pred = .pred + 1.7,
         .pred = ifelse(.pred >=100,100,.pred)
         
  )


prediction57 <- ensemble_test %>% rename(y = .pred)

write_csv(prediction57, 'predictions/prediction57.csv')




read_csv("predictions/prediction59.csv")

