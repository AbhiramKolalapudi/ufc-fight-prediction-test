# Results and comparison

## Dataset and evaluation

The run retained 4,073 bouts dated 2010-03-21 through 2019-12-21. It excluded no-contests and had 2,372 Red wins and 1,701 Blue wins. No draws occurred in this date slice, so the measured task is binary even though the reference describes a win/draw/loss problem.

The evaluation used 20 chronological `TimeSeriesSplit` folds. Training windows expand over time; each test block follows its training data. Preprocessing is fitted within each training fold. The reported values below are means across folds. Standard deviation is across test-fold accuracy.

## Reported vs. reproduced

| Model | Paper test accuracy | This reproduction | Difference |
|---|---:|---:|---:|
| SGD, perceptron loss | 58.018% | 54.1% | -3.9 pp |
| MLP (5, 2), L-BFGS | 57.736% | 54.6% | -3.1 pp |
| Decision tree, `ccp_alpha=0.003` | 60.252% | 55.1% | -5.2 pp |
| Gradient Boosting, learning rate 0.01 | 61.226% | 58.2% | -3.0 pp |

The replacement results are lower than the paper for all four reproduced models. The most likely reasons are the source data beginning in 2010 rather than 1993, the smaller feature set, the lack of draws in this slice, and details the paper does not fully specify (including exact fold construction and estimator defaults).

## Added model

| Model | Mean training accuracy | Mean test accuracy | Test accuracy SD | Mean macro-F1 |
|---|---:|---:|---:|---:|
| Random Forest (extension) | 99.3% | 58.2% | 3.4 pp | 0.531 |
| Gradient Boosting (best reproduced) | 63.7% | 58.2% | 3.4 pp | 0.410 |

Random Forest performs similarly on accuracy and scores higher on macro-F1 than Gradient Boosting, but its 99.3% mean training accuracy shows substantial overfitting. It does not improve test accuracy over Gradient Boosting and remains about 3 percentage points below the paper's reported Gradient Boosting result.

## Complete model summary

| Model | Mean train accuracy | Mean test accuracy | Test accuracy SD | Mean macro-F1 |
|---|---:|---:|---:|---:|
| SGD perceptron | 56.4% | 54.1% | 4.7 pp | 0.523 |
| MLP (5, 2) | 68.5% | 54.6% | 5.0 pp | 0.484 |
| Decision tree | 69.1% | 55.1% | 4.7 pp | 0.428 |
| Gradient Boosting | 63.7% | 58.2% | 3.4 pp | 0.410 |
| Random Forest (extension) | 99.3% | 58.2% | 3.4 pp | 0.531 |

## Interpretation

The Always-Red baseline is 58.2% on the selected sample, so Gradient Boosting and Random Forest match that simple baseline on mean accuracy. Fold-level performance is more informative than one random split because each evaluation block occurs later than its training data. In this dataset, the extension's macro-F1 suggests a more balanced Red/Blue prediction mix, but its large train-test gap makes that result less compelling as a generalization improvement.

The MLP reached the 200-iteration cap in some folds and emitted convergence warnings. Its scores should be read as the result of the documented implementation settings, not a fully tuned optimum.
