# Three-minute project demo

Use the project terminal during the recording. Total target duration: about 2 minutes 30 seconds.

## Run

From the project directory, run:

```powershell
python src\train.py
```

## Spoken walkthrough

**0:00–0:25 | Problem and data**  
“We predict the winning UFC corner using information available before each fight. The paper's original data link is unavailable, so we use a public UFC dataset with dated pre-fight fighter statistics. We restrict it to 2010 through 2019, matching the paper's end date, and exclude odds and post-fight information. The chosen period contains Red and Blue outcomes but no draws.”

**0:25–1:05 | Reproduction models**  
“The script sorts the bouts by date and evaluates four reported model families: perceptron SGD, a two-layer MLP, a pruned decision tree, and gradient boosting. We use 20 expanding time folds. Preprocessing fits within each training fold, so each test block stands later in time.”

**1:05–1:35 | Extension model**  
“Random Forest is the additional model. It uses exactly the same folds and inputs as the reference models. The script reports accuracy and macro-F1, which gives equal weight to Red and Blue performance.”

**1:35–2:10 | Results**  
“The output table reports average training accuracy, average test accuracy, variation across folds, and macro-F1. We compare the four test accuracies with the paper's reported 58.0, 57.7, 60.3, and 61.2 percent. Any difference can reflect the replacement dataset, feature set, and library settings.”

**2:10–2:30 | Takeaway**  
“Gradient boosting was the strongest model in the reference. On this replacement dataset, Random Forest matches it on accuracy and scores higher on macro-F1, though its training accuracy is much higher than its test accuracy. The dataset has no draws in our selected period.”

Replace the bracketed metric references in your spoken narration with the values printed by your run. The same metrics are in `results/model_comparison.csv`.
