---
title: Results
nav_include: 3
---

## Contents
{:.no_toc}
*
{: toc}






## Preferred Models
In our approach to creating a rating system for Yelp users and restaurants, we looked at a number of different models to predict the ratings that certain users would give a certain restaurants.  The approaches we took to create these models were the following: a baseline estimate derived by taking the means of specific user ratings and specific restaurant ratings as bias terms, a baseline estimate derived by using Ridge regression to fit the individual bias terms for each user and each restaurant, a latent factor estimate optimized via an implementation of matrix factorization and stochastic gradient descent, a neighbor model fit using the k-nearest neighbors algorithm, and an ensemble regression method to combine the results of some of the previous models.

### Baseline Model 
Our modeling process began with two baseline estimates.  The first of these two attempted to predict the rating by a given user for a given business via a model of the form $$\hat{Y}_{um} = \mu + \theta_u + \gamma_m$$,where $\hat{Y}_{um}$ represents the rating of restaurant $m$ by user $u$, $\mu$ represents an intercept term, $\theta_u$ represents the bias of user $u$, and $\gamma_m$ represents the bias of restaurant $m$.  For example, a user might be biased in that they tend to generally rate restaurants higher than other users, or a restaurant might be biased in that it tends to be rated lower than other restaurants.

To estimate these biases in this simple baseline, we first took the mean of all ratings to act as the intercept (i.e. a global average).  We then represented the bias of user u as the difference between the average rating of user u and the global average.  We similarly represented the bias of restaurant m as the difference between the average rating of restaurant m and the global average.

### Baseline Model Graphs
![png](graphs/basemodel.png)

### Regularised Regression Model
The second of these two baseline estimates utilized a Ridge regression model to fit for the biases of each user and restaurant via a model of the form $\hat{Y}_{um} = \mu + \bar{\theta} \cdot I_u + \bar{\gamma} \cdot I_m$, where $\bar{\theta} \cdot I_u$ represents a vector of user biases multiplied by an indicator variable of user $u$ and $\bar{\gamma} \cdot I_m$ represents a vector of restaurant biases multipled by an indicator variable of restaurant $m$.

To estimate these bias vectors in the regression baseline, we one-hot encoded the users and businesses in the sample to act as the indicator variables (i.e. the column for user $u$ would have a 1 in rows where the user gave the review, and zeroes elsewhere, and similarly for restaurant m).  We then used sklearn's RidgeCV method to fit the data with a regularized regression model (cross-validated for the optimal regularization parameter) and obtain the vectors of coefficients for each user and each restaurant, as well as obtain an intercept estimate.  We could then use these coefficient estimates and intercept estimates to obtain predicted ratings for given users and restaurants.

### Regularised Regression Model Graphs
![png](graphs/regularisedregression.png)

### Matrix Factorisation  Model

Our third modeling approach used a latent factor model to attempt to predict ratings based on latent features between users and restaurants.  Latent features are interactions between users and properties of restaurants that may be significant in determining a rating - for example, a user may have a penchant for spicy food or cheap prices.  Given a matrix of restaurant ratings $R$ with users as rows and restaurants as columns, we can attempt to decompose this matrix into the form $P \times Q^T$, where each row of $P$ represents the associations between a user and a latent feature and each row of $Q$ represents the associations between a restaurant and a latent feature.  Of course, we do not know what the numerical values for these associations are - thus, we estimate them through this matrix factorization via optimizing by a stochastic gradient descent method.
In all, we look to minimize the loss function $$e^2_{um} = (Y_{um} - \sum_{k=1}^K{p_{uk}q_{kj}})^2 + \frac{\beta}{2}\sum_{k=1}^K{||P||^2 + ||Q^2||}$$, where the second term represents a regularization to prevent from overfitting on the user and restaurant data in P and Q.  We perform this minimization through an implementation of stochastic gradient descent, which iterates through loops to take the partial gradients for $p$ and $q$, create an update rule for each iteration, and update the values of each $p$ and $q$ according to our previous estimates and the new update rule.  After enough iterations, the process converges on estimates for the latent feature associations, which can be converted back into a full predicted ratings matrix $R$.  Due to runtime issues, we could not achieve absolute convergence with stochastic gradient descent and limited the number of iterations.

### Matrix Factorisation on Stars Model Graphs
![png](graphs/matfactstars.png)

### Matrix Factorisation on Residuals Model Graphs
![png](graphs/matfactresid.png)

### K-Nearest Neighbours Model

Our fourth modeling approach used a neighbor model to attempt to predict ratings based on user and restaurant associations with their nearest neighbors.  To implement this neighbor model, we used sklearn's KNeighborsRegressor method, which performs regression based on a k-NN algorithm.  We one-hot encoded the users and restaurants as we did in the Ridge regression model.  We then fit the k-NN regressor on the training set and used it to make predictions for ratings given a specific user and restaurant.

### K-Nearest Neighbours Model Graphs
![png](graphs/KNN.png)

### Ensemble Model

Our fifth modeling approach applied an ensembling method to combine the results of all of our previous models.  The ensemble method we implemented was a form of stacked regression, where the outputs of the previous models were gathered as predictors and used in a least squares regression on the validation set to deliver a set of weights assigned to each model's outputs.  We then used this regression model to predict ratings for given users and restaurants on the training and validation set, as well as a testing set.

### Ensemble Model Graphs on Training and Validation Sets
![png](graphs/ensembletrainval.png)

### Ensemble Model Graphs on Test Set
![png](graphs/ensembletest.png)


## Strengths

## Shortcomings

## Moving Forward




