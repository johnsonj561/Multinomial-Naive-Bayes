### Multinomial Naive Bayes
#### In Relation to Text Classification

Probabilistic supervised learning method. 
Maximum a posteriori (MAP) - the best class, the most likely, for a given document d
```
c<sub>map</sub> = argmax P(c | d) = argmax P( c ) ∏  P( t<sub>k</sub> | c)
```
Multiplication acrossed many probabilities will lead to overflow. Alternative solution is to calculate c<sub>map</sub> using sum of logs.
```
c<sub>map</sub> = argmax [log P( c ) + ∑ P( t<sub>k</sub> | c)]
```

MLE - max likelihood estimate
```
P(c) = N<sub>c</sub> / N      

P(t | c) = T<sub>ct</sub> / ( ∑<sub>t' in V</sub> T<sub>ct'</sub> )   

```

N is total number of docs.
N<sub>c</sub> is number of docs in class c
T<sub>ct</sub> is number of occurences of t in training docs from class c

Note - this definition assumes positional independence.

Problem - the MLE estimate will be zero for term-class pair that did not occur in training set. The conditional probability of one event can zero out the entire MLE estimate.

Solution - add one smoothing (add-one or Laplace smoothing) simply adds 1 to each count to prevent zero'ing out.







main.py implements the Multinomial Naive Bayes algorithm and tests on a sample collection.