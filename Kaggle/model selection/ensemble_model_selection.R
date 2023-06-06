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

test <- read_csv('data/test.csv') %>% mutate(x516 = as_factor(x516))


ensemble_model <- 
  stacks() %>%
  add_candidates(svm_tuned) %>%
  add_candidates(nn_tuned) %>%
  add_candidates(bt_tuned) %>%
  add_candidates(bmlp_tuned)

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
         .pred = .pred + 2
  ) %>%
  rmse(truth = y, 
       estimate = .pred
  )

train_predictions <- pca_train%>%
  bind_cols(predict(ensemble_fit, .)) %>%
  mutate(.pred = exp(.pred),
         y = exp(y)
  )


ensemble_test <-
  pca_test %>%
  bind_cols(predict(ensemble_fit, .))



