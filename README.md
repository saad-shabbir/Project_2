---
title: "Project2"
author: "Saad"
date: "2024-03-07"
output: html_document
---

```{r cars}
library(readr)     # For read_csv
library(dplyr)     # For data manipulation verbs
library(ggplot2)   # For plotting
library(broom)     # For extracting information from model objects


library(plotROC)   # For visualization of classification error
library(glmnet)    # For regularized regression

swift <- read_csv("../data/swift_spotify.csv")
```
## Create Test and Training based on random sample
```{r}
split <- swift |>
  filter(album != c("Midnights"))

dt = sort(sample(nrow(split), nrow(split)*.7))
swift_train<-split[dt,]
swift_test<-split[-dt,]
```


## Popularity Model 

First I looked at the largest positive coefficient between popularity and the variables with a high R^2 to explain variability so I picked danceability, key,  loudness, speechiness, liveness, valence, and duration_ms.

Model 1: Included all variables with multiple predictors and interactions with variables chosen above

Model 2: Included only variables chosen above woth interactions with variables
```{r}
model1 <- lm(popularity ~ danceability, data = split)
model2 <- lm(popularity ~  energy, data = split)
model3 <- lm(popularity ~ key, data = split)
model4 <- lm(popularity ~ loudness, data = split) #6.16
model5 <- lm(popularity ~ speechiness, data = split) #5.27
model6 <- lm(popularity ~ acousticness, data = split)
model7 <- lm(popularity ~ instrumentalness, data = split)
model8 <- lm(popularity ~ liveness, data = split) #7.29
model9 <- lm(popularity ~ valence, data = split) #
model10 <- lm(popularity ~ tempo, data = split)
model11 <- lm(popularity ~ duration_ms, data = split) #
model12 <- lm(popularity ~ time_signature, data = split)



# Model 1
model_lm <- lm(popularity ~ danceability + energy + key + acousticness + instrumentalness + tempo + time_signature + loudness * speechiness * liveness * valence * duration_ms, data = split)

glance(model_lm)

# Model 2
model_lm1 <- lm(popularity ~ danceability * key * loudness * speechiness * liveness * valence * duration_ms, data = split)


glance(model_lm1)



```

Based on the R squared values, Model 2 does a better job explaining 98% of the variability in the data.

I ended choosing Model 2 as the RMSE is much lower and I believe it is not overfit as both test training data have similar values. Additionally, the MSE is much lower for model 2.

## Final Popularity model
```{r}
train <- rep(NA, 2)
test <- rep(NA, 2)

train[1] <- augment(model_lm1, newdata = swift_train) |> 
  summarize(RMSE = sqrt(mean(.resid ^ 2)))

test[1] <- augment(model_lm1, newdata = swift_test) |> 
  summarize(RMSE = sqrt(mean(.resid ^ 2)))

test[2] <- augment(model_lm, newdata = swift_test) |> 
  summarize(RMSE = sqrt(mean(.resid ^ 2)))

train[2] <- augment(model_lm, newdata = swift_train) |> 
  summarize(RMSE = sqrt(mean(.resid ^ 2)))

train
test


augment(model_lm1, newdata = swift_test) |> 
  select(id, album, name, popularity, .fitted, .resid) |>
  summarize(MSE_lm1_train = mean((.resid)^2))

augment(model_lm1, newdata = swift_train) |> 
  select(id, album, name, popularity, .fitted, .resid) |>
  summarize(MSE_lm1_test = mean((.resid)^2))


augment(model_lm, newdata = swift_train) |> 
  select(id, album, name, popularity, .fitted, .resid) |>
  summarize(MSE_lm_train = mean((.resid)^2))


augment(model_lm, newdata = swift_test) |> 
  select(id, album, name, popularity, .fitted, .resid) |>
  summarize(MSE_lm_test = mean((.resid)^2))



```


##Eras Model 1

First I looked at the largest positive coefficient between eras and the variables so I picked energy, key,  valence, and time_signature

Model 1: Included all variables with multiple predictors and interactions with variables chosen above
```{r}
model_logit <- glm(eras ~ danceability + energy + key + loudness + speechiness + 
                  acousticness + instrumentalness + liveness + valence + 
                  tempo + duration_ms + time_signature, data = split, family = "binomial")

tidy(model_logit)

# Model 1
model_logit1 <- glm(eras ~ danceability + loudness + acousticness + instrumentalness + liveness + tempo + duration_ms + energy * key * valence * time_signature, data = split, family = "binomial")

augment(model_logit1, newdata = split, type.predict = "response") |> 
  mutate(eraspredict = as.numeric(.fitted > 0.5)) |> 
  group_by(eras, eraspredict) |> 
  summarize(n = n()) 

augment(model_logit1, newdata = split, type.predict = "response") |> 
    mutate(eraspredict = as.numeric(.fitted > 0.5)) |> 
  mutate(mean = mean(eraspredict)) |>
  summarize(MSE_split = mean((eraspredict - mean)^2))

  
```

##Eras Model 2

Model 2: My preferred model for predicting whether each Midnight song will be on the Eras tour set list is a logistic regression model. This model takes into account  energy, key,  valence, and time_signature to predict the probability of each song being included in the set list. The in-sample prediction was low indicating that it will give good predictions of whether or not each Midnight song will be on the Eras tour set list.

```{r}
# Model 2 
model_logit2 <- glm(eras ~ energy * key * valence * time_signature, data = split, family = "binomial")


augment(model_logit2, newdata = split, type.predict = "response") |> 
  mutate(eraspredict = as.numeric(.fitted > 0.5)) |> 
  group_by(eras, eraspredict) |> 
  summarize(n = n()) 

augment(model_logit2, newdata = split, type.predict = "response") |> 
    mutate(eraspredict = as.numeric(.fitted > 0.5)) |> 
  mutate(mean = mean(eraspredict)) |>
  summarize(MSE_train = mean((eraspredict - mean)^2))

augment(model_logit2, newdata = swift_test, type.predict = "response") |> 
  mutate(eraspredict = as.numeric(.fitted > 0.5)) |> 
  group_by(eras, eraspredict) |> 
  summarize(n = n()) 

augment(model_logit2, newdata = swift_test, type.predict = "response") |> 
    mutate(eraspredict = as.numeric(.fitted > 0.5)) |> 
  mutate(mean = mean(eraspredict)) |>
  summarize(MSE_test = mean((eraspredict - mean)^2))
  

swift <- augment(model_lm1, newdata = split) %>%
  mutate(popularity_predict = .fitted) |>
  select(-.fitted, -.resid)
  
final_predictions <- augment(model_logit1, newdata = swift, type.predict = "response") %>%
  mutate(eraspredict = as.numeric(.fitted > 0.5)) |>
  select(-.fitted)

```

```{r}
myname <- "saadshabbir"

write_csv(final_predictions, paste0("../data/", myname, "_swift_predictions.csv"))
```

