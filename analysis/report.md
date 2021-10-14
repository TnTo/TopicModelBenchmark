---
title: "Report preliminare - Benchmark di algoritmi per Topic Modeling"
author: "Michele Ciruzzi"
output:
    pdf_document:
        keep_tex: yes
        toc: true
    html_document:
        keep_md: true
        toc: true
---





## Prelude

* NMI is the normalized mutual information
* SoftNMi is an extension of NMI for softclustering as defined by Lei, Y., Bezdek, J. C., Chan, J., Xuan Vinh, N., Romano, S., & Bailey, J. (2014). Generalized information theoretic cluster validity indices for soft clusterings. 2014 IEEE Symposium on Computational Intelligence and Data Mining (CIDM), 24–31. https://doi.org/10.1109/CIDM.2014.7008144
* autoNMI is computed against another model instead of against the labels

## Datasets
| Dataset      |   #Documents |   #Words |   #Labels |   #Tokens |
|:-------------|-------------:|---------:|----------:|----------:|
| 20newsgroups |        18846 |    70537 |        20 |   2553284 |
| wos          |        46985 |    72871 |         7 |   5366416 |
| wos2         |        46985 |    72871 |       143 |   5366416 |

## Time
<img src="report_files/figure-html/unnamed-chunk-3-1.png" width="1444" />

hSBM appears to be consistently slower than the other methods on wos dataset. Similarly, both hSBM and TM appear to be slower on 20newsgrops dataset.

For hSBM the conversion into our custom format to store the topic model is really time-consuming and probably could be optimized (e.g. by parallelizing the code). On the other side, this is the only hierarchical algorithm, which means that it fits multiple topic models at the same time, making the loss of speed sometime acceptable.

TM requires a pruning of the graph which is time-consuming and not parallelized nor cached in the implementation used. A sketch of a parallelized and cacheable version is provided alongside the original code, but it is not integrated in the binary executable.

TM and hSBM are the only non-parallelized algorithms: in order to parallelize TM it is necessary to switch from the standard version of InfoMap clustering algorithm to a parallel (and less accurate) one; for hSBM it is necessary to reimplement the whole MCMC inference with another library capable of parallelism (even though it is not assured that a parallel implementation will be faster without requiring too much RAM).

The times showed are for the single fit, so for parametric algorithms they have to be multiplied by the number of numbers of topics to be considered.

Finally, the right side outliers, where present, can be caused by the conversion of the dataset in the necessary format (which is cached after the first time).

## Goodness of fit

<img src="report_files/figure-html/unnamed-chunk-4-3.png" width="1946" />

The maximum value for each seed has been considered: these are the best level for hierarchical algorithms and the best number of topics for parametric algorithms.

The Gibbs Sampling (i.e. MCMC) versions of both LDA and HDP appear to be more accurate to infer the topic structure of the dataset than the variational versions (VB) in quite every context.
The comparison between NMI and SoftNMI suggests us that the increased number of iterations in LDAGSlonger, as compared to LDAGS, (and partially also for LDAVBlong respect to LDAVB) produces a more clear classification (i.e. the probability distribution is more skewed towards one of the vertices of the simplex).

TM and NMF show very good consistency in the results while HDPGS and hSBM show the greater variability.

20newsgoups has 20 different labels to be predicted and LDA gives the better results, while HDP, TM and hSBM give still good results (particularly in terms of SoftNMI). WoS has 7 labels and hSBM performs poorly as compared to the other algorithms, while it outperforms every other algorithm for the WoS2 dataset with 143 labels.
This could suggest avoiding hSBM where very few topics are to be inferred, while it can be a very useful tool when a rich fine-grained description is needed.

## Reproducibility

<img src="report_files/figure-html/unnamed-chunk-5-5.png" width="1946" />

Looking at autoNMI emerge the high degree of reproducibility of TM. At the same time, the GS versions of HDP and LDA as well as NMF still achieve a good level of reproducibility, even if lower than TM.

On the other hand, I'm not sure about how to interpret the right plots: TM still have small error bars (i.e. are all differents in the same way) but HDPGS achieve better results (maybe predicts skewer probabilities?).

hSBM (we will see more later) gives a different number of levels in each run, which makes the coupling problematic and lowers the results: I choose to start with the root level (minimum number of clusters) and coupling counting from it.

## Parametric algorithms

### LogLikelihood vs Perplexity
<img src="report_files/figure-html/unnamed-chunk-6-7.png" width="1946" />

First we note that the perplexity, a common used measure, is monotonically decreasing with the likelihood of the model, and so they can be considered as equivalent.
We will use the loglikelihood.

### NMI vs #Topics
<img src="report_files/figure-html/unnamed-chunk-7-9.png" width="1897" />
Second, we note that for NMF and the two versions of LDAGS NMI has a similar shape for every random seed.
Moreover for these three algorithms, NMI has a maximum for a given number of topics which is generally close to the number of labels to be predicted.

### GoF (Error or LogLikelihood) vs #Topics
<img src="report_files/figure-html/unnamed-chunk-8-11.png" width="1966" />
For NMF and the two versions of LDAVB the correlation between the loglikelihood and the NMI (i.e unsupervised and supervised goodness of fit) is (almost perfectly) linear and so we have no way in a supervised setting to infer the optimal number of topic using unsupervised metrics.
But, for LDAGS and particularly LDAGSlonger there is a clear minimum which could be a way to define an optimal number of topics, even if it requires to fit a certain number of models to find the minimum.
It is important to highlight that this number of topics is quite different from the number of labels.

### GoF (Error or LogLikelihood) vs NMI
<img src="report_files/figure-html/unnamed-chunk-9-13.png" width="1897" />
This relation should be further explored, since for some combination of datasets and algorithms this relation is clearly U-shaped (or C-shaped).

It suggests that the number of topics considered optimal by the model is not that same a human who classifies the corpus would use.

### Assigned #Topics
<img src="report_files/figure-html/unnamed-chunk-10-15.png" width="1947" />
We can exclude that this could be a good metrics to chose the optimal number of topics.

## #Topics
<img src="report_files/figure-html/unnamed-chunk-11-1.png" width="1947" />

HDPGS and TM show a very good accordance in the number of topics inferred. However, HDPGS assigns documents to a (slighty) lower number of first topics.
LDAGS is coherent among different seeds, but gives different results from HDPGS and TM.

We note that also for non-parametric algorithms the number of topics predicted is not in accordance with the manually defined number of labels.

## Hierarchical

### #Topics
<img src="report_files/figure-html/unnamed-chunk-12-17.png" width="1961" />
The number of topics is comparable to the other non-parametric methods if we consider the first, second or third levels.
The usefulness of the deeper level has to be demonstrated, since the number of topics inferred in them reaches the same magnitude of the number of documents.

### Reproducibility

```
<string>:1: UserWarning: Attempted to set non-positive bottom ylim on a log-scaled axis.
Invalid limit will be ignored.
```

<img src="report_files/figure-html/unnamed-chunk-13-19.png" width="1986" />
<img src="report_files/figure-html/unnamed-chunk-14-21.png" width="1986" />
Only the first levels appear for every seed, so the increase of reproducibility in the deeper levels can be an artifact given by the lower numerosity.

## Conclusions

NMF and the two versions of LDAVB have the huge downside that it is not possible to find the optimal number of topics in an unsupervised setting, and so the choice of the actual model is arbitrary.

Also, variational implementations of LDA and HDP perform worse than the Gibbs Sampling ones.

hSBM shows better results in the deeper levels, which have more than one hundred different topics predicted. Moreover, the numbers of levels and topics vary a lot among the random seeds. Finally, the model perform better when the clusters of documents are considered, which is an orthogonal interpretative approach to the other models.

TM shows an incredible level of consistency among different random seeds.

At the end of this assessment, the three possible choices are LDAGSlonger, TM and HDPGS: the reason to prefer HDPGS is the better reproducibility score in terms of softNMI; LDAGSlonger reaches better NMI results but requires to fit multiple models to find a minimum; TM shows good reproducibility. But as longer versions of LDA improve the SoftNMI of shorter versions of LDA, it is reasonable to hypothesize that with an increased number of iterations for the LDA step of TM, SoftNMI can improve as well.

Finally, the worst results of HDPGS and TM are for the WoS2 dataset, which have a really high number of labels to be predicted (more than one hundred): if the aim of the topic model is to synthesize information in a manageable number of topics, the underestimation of the number of topics should be considered as a positive rather that a negative side effect.

All considered, I suggest adopting TM as the topic model algorithm of choice for the analysis of the dataset.
