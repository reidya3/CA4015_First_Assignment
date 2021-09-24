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
The Table below summarizes the multiple datasets used in this investigation.
|    Study      | Number of Participants |    Trails      | Gender || IGT verision |
| :------------ | -------------: | :------------ | -------------: |
|        0      |        15      |        95     |        5       |
|     13720     |      162      |     100     |      2744      |
|        0      |        19     |        100      |        5       |
|     13720     |      40     |     100     |      2744      |
|        0      |        25       |        100      |        5       |
|     13720     |      70      |     100     |      2744      |
|        0      |        57       |        150      |        5       |
|     13720     |      41      |     150    |      2744      |
|        0      |        153       |        100      |        5       |
|     13720     |      35      |     100     |      2744      |



Here, we restricted the analysis to the subset of 7 studies which used the classical 100 trials version of the IGT, resulting in 504 participants (age range: 18–88 years; for the 5 studies with available information about sex: 54% of females). Within this dataset, 153 participants come from a single study on aging [15]. Among these participants, 63 are older adults (61–88 years old; 17 males) and 90 are younger adults (18–35 years old; 22 males) matched in terms of education level and intelligence (WASI vocabulary)Further explanation of these datasets is provided in the Initial Data Exploration Section.

In this investigation, we seek to use a variety of clustering approaches to segment the participants into well-defined groups. 
To start, we perform an initial data exploration to perform minimal transformations & data sanitization checks; acquire  rudimentary statistics of the datasets; measure cluster tendency and validate any assumptions required by our chosen clustering algorithm (K-means). Next, we perform cluster analysis  and evaluate our clusters using metrics such as Silhouette Coefficient and an Elbow curve. 
These clusters represent participants that exhibit similar decision-making patterns, and may have similar underlying psychological qualties such as impulsivity, stress  . 

Next, we attempt to form a federated k-means algorithm and compare results.Finally, we conclude with the most important outcomes of our work. 