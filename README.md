Project 6: Short text similarity

We aim in this project to study a new semantic similarity for two short text document

1. Consider two sentences S1 and S2 which are tokenized as which are tokenized for instance as S1=(w1, w2,..,wn) and S2= (p1, p2, …, pm). Consider the approach implemented in Lab2 for calculating the semantic similarity between sentences using WordNet semantic similarity. Check the output for pair of sentences  (This city is awful these days, The city can improve better if better management is held) and comment on the result. Test various preprocessing stages and discuss the impact of preprocessing on the result

2. We want to test this strategy on publicly available sentence database. For this purpose, use STSS-131 dataset, available in “A new benchmark dataset with production methodology for short text semantic similarity algorithms” by O’Shea, Bandar and Crockett (ACM Trans. on Speech and Language Processing, 10, 2013). Use Pearson correlation coefficient to test the correlation of your result with the provided human judgment. 

3.  Now we want to implement a new semantic similarity based measure. The idea is to use some hierarchical reasoning and explore the WordNet Hierarchy. For this purpose, proceed in the following. For each sentence, use the parser tree to identify various part-of-speech of individual token of the sentence. Generate the list of hypernyms H1 and hyponyms H2 of each noun of the sentence. Repeat this process for each verb. Compute the semantic similarity between the two sentences as  

![math](https://user-images.githubusercontent.com/39261760/71151310-20afdd80-223c-11ea-98cb-73eb07c5701c.png)

Implement the above similarity expression in your python code

4. Test the above similarity on STSS-131 database and report the Pearson correlation with human judgments.
5. Study another text similarity using both wordnet semantic similarity and string similarity provided in https://github.com/pritishyuvraj/SOC-PMI-Short-Text-Similarity-. Check the behavior of program for some intuitive sentences (very similar sentences, ambiguous and very dissimilar ones)
6. Report the result of the above similarity on STSS-131 and report the corresponding Pearson correlation with human judgments
7. Suggest an interface of your choice that would allow the user to input a textual query in the form of a pair of sentences and output the similarity score according to the various methods described above.
