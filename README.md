# Shapelets Sampling and Quality Measures (SSQM)

Imagine a bird tweeting, and lets assume that birds of different species can be classified by the sounds of their tweets alone.
Thus, if one looked at the birds tweet sound wave, one would expect to find some subsequences that are unique to each birds 
species (patterns). These subsequences of time series data that represents patterns are called shapelets, and the goal 
of my master's work was to find good shapelets for accurate classification of data.

## Objectives

This work has two main goals: the first one is related to ranking the shapelets by its usefulness (for classification); and the
second one is about reducing the search-space (often it is too large to be fully explored).

1. Continuing with the example of a bird, in its sound wave we expect to find some patterns that represents silent moments, 
and those would also appear in recordings of other birds species, thus not every pattern is usefull. This lead to the 
development of quality measures that ranks these patterns. At the literature the standard quality measure has been the 
information gain, and recently it was proposed the use of f-statistic; at this work we propose a new one called in-class 
transitions.

2. From a data set of time series, initially every possible subsequence is a candidate for a good shapelet. However, to explore this
full search-space can be computationally prohibitive, thus a reduction of this search-space is required. At this work it is explored
how the use of random sampling at different levels of sampling affects the quality of good shapelets found and its impact on the
classifiers accuracy.

## Results and Experiments

My experiments are 100% reproducible and can be easily executed by running **SSQM.Rmd** with all of the supporting files contained
in this folder. The output of this .Rmd is the [**SSQM.md** report](https://github.com/lucasschmidtc/SSQM/blob/master/SSQM.md),
which mixes code, graphs and text to detail each step of my research/experiments.
