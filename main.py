# Multinomial Naive Bayes Implementation/Testing

# To be implemented:
TrainMultinomialNB(C, D):
  V <- extractVocab(D)
  N <- countDocs(D)
  for each c in C
    do Nc <- countDocsInClass(D,c)
    prior[c] <- Nc / N
    textc <- concatenateTextOfAllDocsInClass(D, c)
    for each t in V
      do Tct <- countTokensOfTerm(textc, t)
      for each t in V
        do condprob[t][c] <- ( Tct + 1 / Summation of (Tct + 1))
  return V, prior, condprob

ApplyMultinomialNB(C, V, prior, condprob, d):
  W <- extractTokensFromDoc(V,d)
  for each c in C
  do scores[c] < log prior[c]
    for each t in W
      do score[c] += log condprob[t][c]
  return argmax(score[c]);