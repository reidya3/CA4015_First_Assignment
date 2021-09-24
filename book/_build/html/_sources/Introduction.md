# Introduction
---

The Iowa Gambling task {cite:p}`bechara1994insensitivity` asses real-life decision making. Developed by researchers at the University of Iowa, subjects participate in a simulated card game. Participants start with $2,000 and are presented with 4 card decks (A - D), with each deck more likely to yield monetary rewards or penalities over time i.e some decks are "bad" while others are "good". Down below, A and B are considered "bad" as they have a negative expected value whilst C and D are consider "good" as they are associated with a positive expected value. Test-takers obtain feedback on the amount either lost or gained and the running total after each choice (trail). The subject's goal is to adapt their pattern of choices to maximize the reward received.

![Iowa-Gambling-Task](images/iowagambling.png)

In standard setups, the task typically lasts 100 trials. Empirical Investigations  have shown that healthy (neurotypical) test takers generally become aware of the "good" and "bad" decks after 20 to 40 trials {cite:p}`brand1`. However, patients who suffer from orbitofrontal cortex (OFC) dysfunction tend to continue choosing bad decks even though realization of continued monetary loss may have already occurred in these participants.

## Description of Datasets
This investigation utilizes a dataset from a "many labs" initiative on the Iowa Gambling task grouping 10 studies and containing data from 617 healthy participants {cite:p}`steing1`. The data consist of the choices of each participant on each trial, and the resulting rewards and losses
:::{note}
All studies Not all studies had the same number of trials. They number of trails varied from 95, 100 and 150. 
:::
The Table below summarizes the multiple datasets used in the current study.
|    Study      | Number of Participants |    Trails      | Gender |
| :------------ | -------------: | :------------ | -------------: |
|        0      |        5       |        0      |        5       |
|     13720     |      2744      |     13720     |      2744      |

Further explanation of these datasets is provided in the Initial Data Exploration Section 

 The model's parameters can be interrupted as several different psychological processes such as learnt behavior, impulsivity, stress reactions to punishments etc. Lili et al. describes three  models:

- Values-Plus-Perseverance Mode (VVP)
- Prospect Valence Learning Model with Delta (PVL-Delta)
- Outcome Representation Learning Mode (ORL)

:::{note}
Each reinforcement model's parameters are contained in one dataset. 
:::

In this investigation, we seek to use a variety of clustering approaches to segment the participants into well-defined groups. To start, we perform an initial data exploration to perform minimal transformations & data sanitization checks; acquire  rudimentary statistics of the datasets; measure cluster tendency and validate any assumptions required by our chosen clustering algorithm (K-means). Next, we perform cluster analysis  and evaluate our clusters using metrics such as Silhouette Coefficient and an Elbow curve. 
These clusters represent participants that exhibit similar decision-making patterns, captured by the parameters of the underlying reinforcement model. Next, we adapt to form a federated k-means algorithim and compare results.Finally, we conclude with the most important outcomes of our work. 