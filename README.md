Kip McCharen (cam7cu)
Clair McLafferty (cm2rh)

# DS 6014 Project: Fraud Detection in D&D Dice Rolls

## Problem description

On first blush, a Bayesian analysis of the probabilities surrounding whether a podcast host is cheating at rolling dice in Dungeons & Dragons appears to be a trivial exercise. However, it is a much more nuanced and novel problem than we initially expected, and has applications across numerous industries. Though much research exists on the application of Bayesian techniques to transactional data to discern the potential for fraudulent transactions or default, these methods examine individual data points within a larger corpus. In comparison, we attempt to discern whether a collection of Travis McElroy’s dice rolls in the podcast The Adventure Zone is indicative of fraud.

Despite our whimsical application, this approach could potentially be used to identify individuals or countries whose self-reported data is aberrant from a body as a whole, student cheating in standardized testing, or even faulty gambling machines. In fact, the Educational Testing Service, also known as the parent company of the GRE and the SAT, is currently doing work in exactly this field [Sinharay 2020]. However, we did not realize this feature of the problem until after applying a Naive Bayes approach. 

In basic statistics, dice are often used to model simple, frequentist probability exercises, but are generally not the subject of real world analysis. However, many games including D&D use dice as a mechanism for advancing game play. Since the beginning of the pandemic, online platforms such as Roll20 that have built-in, visible dice rolling functions have gained popularity, but many extant podcasts with scattered players’ rolls remain self-reported and unrecorded. Such is the case with Travis in The Adventure Zone. Our intuition is that the probability of fraud is high, especially in high-stakes moments. To analyze the uncertainty within this problem, we used a number of Bayesian applications to explore the difference between the expected posterior distribution of a particular player’s rolls (here, Travis McElroy) and his Roll20-generated counterpart.

As co-host Justin McElroy notes on Twitter, his brother and co-host Travis’s unusually successful dice rolls over the course of the show have caused controversy [McElroy, J., tweet]. So much so, in fact, that fans have manually documented rolls made by Travis over several seasons [Reddit thread]. Reddit user UltimaGabe described the imbalanced distribution of Travis’ rolls, asserting: 

> “Yes, it's possible that his rolls are the result of pure chance. (It's theoretically possible for someone to roll nothing but natural 20s a thousand times in a row.) But if you're talking statistics and probabilities, it's very unlikely to be a legitimate randomization.”

To do so, we downloaded the text file created by UltimaGabe and used regular expressions to extract the 170 individual rolls into a spreadsheet. This includes the reported roll’s outcome, any modifiers related to character attributes or in-game circumstances, its categorical basis, or type of roll, and the presence of advantage. From this, we can also extrapolate the parent category of the basis related to the roll.

Since this file only contained information on Travis and not the other players in the game, we were unable to build a model comparing individuals’ performances. Lacking that, we found the source code from Roll20’s dice rolling functionality, modified it slightly, and simulated 2,000 rolls of a 20-sided die. We then calculated the proportional representation of other categorical variables within the observed data and randomly assigned these characteristics based on this  to the data. In addition, rows assigned “advantage” or “disadvantage” were compared to an additional “roll,” and the higher number was selected for advantage, lower for disadvantage.

Travis’ character, Magnus Burnsides, is a barbarian warrior. The character succeeds in-game primarily by winning fights, indicating that Travis might be tempted to cheat in combat, as well as in instances where the character is at a disadvantage. In our dataset, those rolls account for 57 of the 170 observations.

Once the data were extracted and generated, we plotted the two sets to get a visual understanding of the differences between them. As shown in Figures (2-4), there is a fairly marked difference between the two distributions. However, since advantage and disadvantage change how the rolls are calculated, we will update our estimate of the expected/frequentist probability using a very basic Bayesian approach.

## Bayesian methods

Since we are working with a real data set with assumed rather than professionally applied labels attached, we decided up front to apply a variety of Bayesian approaches to see what worked. We started with a naive Bayes approach, which we thought would be relatively straightforward. This approach to predictive classification modeling assumes independent priors and makes determinations for a data point’s class label based on the combination of associated features. To use this approach, we combined and randomized our data, split it into test and training sets, and fitted our algorithm. The result was 0% accurate on all runs (figure 5), as it classified each of Travis’ rolls as being randomly generated. Though this might sound like evidence that he is not cheating, it was actually an indication of a flaw in our reasoning. We had assumed that we would be modeling fraud as if the observed rolls were independent events instead of each roll being a feature of the model. It was at this point that we began researching test fraud detection and trying to apply those approaches instead of predicting the classification of individual rolls.

However, this breakthrough came with its own difficulties. Due to the sheer volume of predictors, sampling techniques to approximate the posterior probability distribution function would not be usable as the posterior was potentially intractable. Although this was, again, not useful as a predictive model in and of itself, when it finished running after two-and-a-half hours we used the results for feature selection to determine which features were significant.

## Mathematical linkage between the problem and method(s)

To quantify uncertainty in the reported rolls as exactly as possible, we first calculated a basic Bayesian update to the frequentist probabilities attached to a 20-sided die, i.e., 0.05 across the board, using the formula represented at right. Using Python, we implemented code to find how our expectations would change and then to bin and graph the data accordingly to compare the simulated and observed data with the basic Bayesian approach (below). We saw that the expected probability of a roll from 12 to 20 was now 0.547, while a low roll, from 1 to 11, now had probability 0.453.

In addition to this exploration of the data, we also used more sophisticated Bayesian models to explore the probabilities of fraud within our datasets. As previously stated, the mathematical approach for our  Naive Bayes model with a Bernoulli posterior was unsuccessful. However, we can mathematically represent our findings here as the below, where  is the probability of an observation x being in class i given a combination of features h and j. 
 
The generated probabilities are then generated, and the class label associated with the largest probability is chosen. Mathematically,  gibeing chosen as the predicted label will be represented as P(xk | gi) >P(xk | gj) with ties broken arbitrarily.
	The hierarchical model that we ended up using for feature selection is a bit more complex. Because most of our features were categorical, we had originally dummy coded each of them to ensure that they could be inputted into this model. As a result, we had a truly monstrous number of priors represented generally by $yi = j[i] + j[i]Xi + i$

In this mathematical representation, Xirepresents all of our parameters and hyperparameters, and j[i]andj[i]are the parameters associated with group j (here, each of the roll outcomes, or k 1, 2,..., 20). In this run, most of the model parameters had an rmuch greater than one, indicating that they did not significantly contribute to the model. However, several did not, so we went on from here to build our logistic model using a heavily reduced model with parameters advantage, disadvantage, and attack_roll. We then used these to create several further models, which interestingly matched our hunch about what attributes would render Travis more likely to potentially lie about his rolls.

Unfortunately, our sampling approach to the reduced logistic models for the separate Travis data and the generated data did not converge, even with repeated increases of the tuning and tree depth parameters. The equation we used for this analysis is given by
    
$Determination = 0 + 1*advantage + 2*disadvantage +3*no advantage +4 * attack roll$

From this, we posited that it was likely that the posterior distribution remained intractable, but could potentially be approximated with variational inference, specifically using ADVI. For completion’s sake, we also decided to run a model with all of the categorical predictors’ categories replaced by numeric placeholders. This model was given by

$Determination = 0 + 1* roll outcome + 2*advantage type +3*statistic +4 * basis + 5 * roll type + 6 * importance$

where the values were the coefficients fitted by the code and fairly narrow convergence as seen below.

As a whole, the variational inference approach optimizes the bias-variance tradeoff to make an approximation of the posterior probability distribution for each curve. Though we had initially planned to compare our models using Bayes Factor, we were somewhat chagrined to find that this would have been possible with sample, but was not with ADVI. To test our hypothesis, we decided to check out the resulting graph of the posterior distribution as predicted by ADVI (below).

Considering that these are plots of the posterior distribution and not the ADVI approximation as shown in the presentation, we can say more confidently that the probability distribution of the posterior of Travis’ rolls is not equivalent to the probability given by the random dataset. 

## Conclusions

This analysis explored the probability that Travis’ rolls did not conform to what would be expected from a set of truly random, unweighted die rolls, and found many Bayesian demonstrations of high uncertainty about the authenticity of Travis’ rolls. Although we cannot use this analysis as hard proof that he is cheating, our beliefs about the controversy have been updated, and the results do not indicate truthfulness.

Because it’s a fun and interesting application of a more widely applicable and potentially impactful type of problem, more research is warranted on this question. Potential avenues for exploration include, but are not limited to, scraping and text mining full episodic transcripts to gather rolls for all characters and comparing the resulting posterior probability distributions using the methods collected above. Further, with more cleaned data, a sampling approach -- with a resulting possibility for Bayes Factor comparison of the models -- would be possible, leading to less uncertainty in classifying Travis’ behavior.

## References

McElroy, J. @justinmcleroy. (2016 March 17). “Predictably, the subject of lying about dice rolls on 
@TheZoneCast is a hot topic on the MF subreddit.” Retrieved from https://twitter.com/justinmcelroy/status/710638075689340928?lang=en

McElroy et al. The Adventure Zone. https://maximumfun.org/podcasts/adventure-zone/

“Roll20.” Retrieved from https://roll20.net/welcome

Sinharay, S (2018). Application of Bayesian Methods for Detecting Fraudulent Behavior on Tests. Measurement: Interdisciplinary Research and Perspectives, 16(2), 100-113, DOI: 10.1080/15366367.2018.1437308.

Sinharay, S. (2020). Detecting test fraud using Bayes factors. Behaviormetrika, 47(2), 339.doi:10.1007/s41237-020-00113-9

[UltimaGabe]. (2017, July 27). I'm back with more stats on Travis' rolls- it's the Eleventh Hour this time! [Online forum post]. Retrieved from https://www.reddit.com/r/TheAdventureZone/comments/6pxwdr/