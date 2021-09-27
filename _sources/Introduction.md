# Introduction
---

The Iowa Gambling task {cite:p}`bechara1994insensitivity` asses real-life decision making. Developed by researchers at the University of Iowa, subjects participate in a simulated card game. Participants start with $2,000 and are presented with 4 card decks (A - D), with each deck differing more likely to yield monetary rewards or penalities over time i.e some decks are "bad" while others are "good". Down below, A and B are considered "bad" as they have a negative expected value whilst C and D are consider "good" as they are associated with a positive expected value. Test-takers obtain feedback on the amount either lost or gained and the running total after each choice (trail). The subject's goal is to adapt their pattern of choices to maximize the reward received.

![Iowa-Gambling-Task](images/iowagambling.png)

In standard setups, the task typically lasts 100 trials. Empirical Investigations  have shown that healthy (neurotypical) test takers generally become aware of the "good" and "bad" decks after 20 to 40 trials {cite:p}`brand1`. However, patients who suffer from orbitofrontal cortex (OFC) dysfunction tend to continue choosing bad decks even though realization of continued monetary loss may have already occurred in these participants. As presented above, participants must choose long-term advantageous choices over short-term favourable picks to achieve the greatest monetary gain. Therefore, IGT remains a popular choice to evaluate decision making and by extension, impulsivity as it does not suffer from the self-reflection biases that questionnaires tend to display. 

## Description of Datasets
This investigation utilizes a dataset from a "many labs" initiative on the Iowa Gambling task grouping 10 studies and containing data from 617 healthy participants {cite:p}`steing1`. The data consist of the choices of each participant on each trial, and the resulting rewards and losses.
:::{note}
Not all studies had the same number of trials. They number of trails varied from 95, 100 and 150. 
:::
The Table below summarizes the multiple datasets used in this investigation.
| Labs         | Number of Participants | Trails | IGT version | 
| :------------ | -------------: | :------------ | -------------: | 
| {cite:t}`FRIDBERG201028` | 15 | 95 | Modified | 
| {cite:t}`horstmann2012iowa` | 162 | 100 | Original  | 
| {cite:t}`kjome2010relationship` | 19 | 100 | Original |
| {cite:t}`maia2004reexamination` | 40 | 100 | Original | 
| {cite:t}`premkumar2008emotional` | 25 | 100 | Original | 
| {cite:t}`vsmiracognitive` | 70 | 100 | Original | 
| {cite:t}`stein2` | 57 | 150 | Modified | 
| {cite:t}`wetzels2010bayesian` | 41 | 150 | Modified |
| {cite:t}`wood2005older` | 153 | 100 | Original | 
| {cite:t}`worthy2013decomposing` | 35 | 100 | Original |

For further clarification of the different IGT versions used, please consult this [paper](http://irep.ntu.ac.uk/id/eprint/20294/1/220623_2604.pdf). In addition, explanation of these datasets is provided in the Initial Data Exploration Section.

In this investigation, we seek to use a variety of clustering approaches to segment the participants into well-defined groups. uch as learnt behavior, impulsivity, stress reactions to punishments etc. 
To start, we perform an initial data exploration to perform transformations & data sanitization checks; acquire  rudimentary statistics of the datasets; measure cluster tendency and validate any assumptions required by our chosen clustering algorithm (K-means). Next, we perform cluster analysis  and evaluate our clusters using metrics such as Silhouette Coefficient and an Elbow curve. 
These clusters represent participants that exhibit similar decision-making patterns, and may have similar underlying psychological qualities such as impulsivity, stress reaction level to punishments or similar learnt experiences. Next, we attempt to form a federated k-means algorithm to preserve the privacy of the individual labs and compare results. Finally, we conclude with the most important outcomes of our work. 