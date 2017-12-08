---
title: Yelp Restaurant Recommendations

---
### Allison Wiggins, Peter Bearse, Christopher Fenaroli, Daniel Polatajko


![alt text](http://www.practicevelocity.com/wp-content/uploads/2016/08/yelp2.png)


## Our Goal

>In our project, our goal is to create a recommendation system for restaurants as outlined by the project guidelines and by the Yelp Dataset Challenge (www.yelp/Dataset/challenge).  Taking a specific user and specific restaurant as input, we plan to build a model that will predict the rating for a review of that restaurant for that user on a scale of 1-5.  In creating this model, we plan to test a number of different approaches, such as Neighbor Models (a more localized approach for comparing users to one another), Matrix Factorization (a more globalized approach for discovering latent factors between users and restaurants), and an ensemble method to combine the results of these approaches.  After implementing these methods in Python, we will evaluate the accuracy of each on training, test, and validation sets. In the test and validation sets, we only include users and restaurants that have been seen before in the training set to improve the model’s predictive power in transferring it's learning from the training set to other data.

## Our Motivation

>Recommendation algorithms are highly useful for websites like Yelp that contain high amounts of data. Often, users will go online to find various different reviews, ratings, and data on a restaurant. However, the average user will not expand the time necessary to make sense of all of the reviews and data. Thus, a recommendation system that provides high accuracy for a user's taste and preferences would be extremely beneficial.


## Our Challenges

>Our first challenge arose when we initially downloaded the Yelp json files. The files are extremely large, and thus we had to take random samples of the review and user files in order to be able to read them in on our machines. We explored the different data variables to try to see if any of them would serve as powerful model predictors. We found that many of the variables were highly correlated or had terrible distributions. Thus, we proceeded to make our baseline models and build off of them to make  a matrix factorization model. This model allows us to use only the prior stars reviews along with the business ids and user ids that correspond with the review. Thus, in each model we are capturing some sort or prior user and restaurant bias from the mean review in order to try to predict what a user might rate a restaurant. We also had to decipher what to do with the various users and restaurants. In order for our model to have any predictive power, we needed to ensure that the same users and restaurants that exist in the test and validation sets also exist in the training set.
