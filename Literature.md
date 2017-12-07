---
title: Literature Review
nav_include: 2
---

## Contents
{:.no_toc}
*  
{: toc}

## **Chen's** Winning the Netflix Prize
Chen's blog post describes a method for constructing increasingly powerful recommendation systems in a way that is very similar to the methodology we eventually used. In particular, he describes the naive approach we used for our base model, as well as highlighting K-nearest neighbours and matrix factorisation as useful tools for constructing recommenders.
>Chen, Edwin. "Winning the Netflix Prize: A Summary". Edwin Chen, Oct. 2011, http://blog.echen.me/2011/10/24/winning-the-netflix-prize-a-summary/.

## **Bell, Koren and Volinksy's** Matrix Factorization Techniques
By reference to the Netflix Prize recommendation competition, Bell, Koren and Volinksy's paper describes a general overview of matrix factorisation as well as some of the underlying mathematics. In particular, it was useful to us in comparing the stochastic gradient descent and alternating least squares methods of matrix factorisation and deciding which method would be better for us to implement.
>Bell, Robert, Yehuda Koren and Chris Volinksy. "Matrix Factorization Techniques for Recommender Systems". Datajobs, Aug. 2009, https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf.

## **Mackey's** Timeline of the Netflix Prize Competition
The lecture slides of Mackey describe a timeline of the Netflix Prize competition, and in particular describe the way in which some of the most successful competitors used a stacked linear regression ensembling method to combine matrix factorisation and KNN models to produce one superior model.
>Mackey, Lester. "The Story of the Netflix Prize: an Ensembler's Tale." Stanford University, Sept. 2012, http://web.stanford.edu/~lmackey/papers/netflix_story-nas11-slides.pdf.

## **Li's** Book Recommender System
Li explains how she used a kNN and a Matrix Factorization model to find similarities among book readers in order to predict what a reader would review a new book. This is obviously very similar in structure to our project, as we also were looking at user, or restaurant attendee, in order to predict what he or she would rate a restaurant.
>Li, Susan. "How Did We Build Book Recommender Systems in An Hour Part 2 — k Nearest Neighbors and Matrix Factorization." Towards Data Science, Medium, Sept. 2017, https://towardsdatascience.com/how-did-we-build-book-recommender-systems-in-an-hour-part-2-k-nearest-neighbors-and-matrix-c04b3c2ef55c.

## **Yeung's** Intro to Matrix Factorization
Yeung's blog post gave us an introduction to the theory behind matrix factorization as well as the methodology to implementing such a model in Python. Yeung explains very clearly how matrix factorization can be used to find latent factors underlying the interactions between users and and items. It is essentially allowing us to find what is hidden underneath the data that we could not otherwise quantify or use words to explain.  
>Au Yeung, Anthony. "Matrix Factorization: A Simple Tutorial and Implementation in Python." Quuxlabs, Sept. 2010, http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/#source-code.

## **Rosenthal's** Matrix Factorization for Movie Recommendations
Rosenthal shows how to use Matrix Factorization to predict movie ratings. This is where we learned about both the alternating least squares method and the stochastic gradient descent method. Rosenthal explains both the math behind the method and how to further implement the math in code. We ended up adopting his gradient descent method in order to train our matrix factorization model.
>Rosenthal, Ethan. "Explicit Matrix Factorization: ALS, SGD, and All That Jazz". Insight Data Science, Medium, Mar. 2016, https://blog.insightdatascience.com/explicit-matrix-factorization-als-sgd-and-all-that-jazz-b00e4d9b21ea.
