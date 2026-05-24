
================

# Executive Summary

Our approach to this project began with exploratory data analysis,
followed by the creation of new features to enhance predictive accuracy.
We applied four supervised learning methods—linear regression, logistic
regression, random forest, and gradient boosting—on the training data.
Model performance was evaluated using cross-validation based on the Area
Under the Curve (AUC) metric for both loan acceptance and default
predictions. Boosting achieved the highest AUC, indicating superior
discriminative ability. We then used a train-test split with the
boosting model to calculate profit outcomes and guide approval decisions
for the 1,000 new loan applications.

# Feature Engineering

To improve predictive performance and capture nonlinear risk factors,
four new features were created:

Income to loan ratio: income_to_loan = annual_income / (loan_amount \*
(1 + offered_rate)) Captures affordability, where low ratios indicate
financial strain and higher default risk.

Loan to credit lines: loan_to_credit_lines = loan_amount /
num_credit_lines Measures how large the requested loan is relative to
existing credit history.

Inquiries to delinquencies: inquiries_to_delinquencies =
inquiries_6months \* 4 / delinquencies_2yrs Captures short-term
credit-seeking behavior relative to past repayment problems. We multiply
by four to normalize recent inquiries to two years, aligning it with the
delinquency time frame of two years.

Delinquencies to credit lines: delinquencies_to_credit_lines =
delinquencies_2yrs / num_credit_lines Quantifies past credit problems
relative to available credit.

These newly engineered features help us zone in on variables that are
useful to our model, reducing noise and enhancing the representation of
financial relationships that will help the model make better
predictions. It also improves model interpretability, especially for
tree-based algorithms that capture more complex interactions. Our
featured engineering variables helped differentiate low-risk from
high-risk borrowers more effectively than the raw unprocessed variables.

# Model Development

We tested four supervised learning methods: linear regression, logistic
regression, random forest, and gradient boosting. We used
cross-validation to ensure model robustness, with AUC and classification
accuracy as the primary evaluation metrics.

Linear regression provided a baseline but performed very poorly in
comparison due to nonlinearity in loan risk patterns.

Logistic regression improved interpretability and captured key
acceptance characteristics, but it struggled with complex feature
interactions.

Random forest handled non-linearities well and achieved good predictive
accuracy.

Boosting outperformed all others, with almost the highest AUC for both
acceptance and default predictions, demonstrating superior ability to
capture subtle nonlinear dependencies.

# Model Selection Justification:

To determine the best model among the four, we calculated the AUC value
for each model because it provides a common performance measure across
the models.

Below are the metrics that we calculated for each model.

Linear model: AUC (Acceptance = 0.796; Default = 0.622), plus R² and MSE
metrics.

Logistic model: AUC (Acceptance = 0.889; Default = 0.6241), with
accuracy, precision, recall, and specificity also computed.

Random forest: AUC (Acceptance = 0.812; Default = 0.555), along with
out-of-bag error.

Boosting (GBM): AUC (Acceptance = 0.882; Default = 0.7192).

Although additional metrics were calculated, AUC served as the most
meaningful for direct comparison. The boosting and logistic models
achieved similar acceptance AUCs, but boosting outperformed logistic
regression in default prediction (approximately 72% versus 62%). As a
result, we decided on using the boost gbm model as our final model for
profit-estimation and decision-making.

# Business Strategy

Interest rate setting: In our code, interest rates were determined by an
individualized profit maximization approach. The procedure evaluated a
set of possible interest rates for each loan and calculated the expected
profit associated with each one. For every loan, the model identified
the rate that produced the highest expected profit, which became the
optimal rate for that borrower. The model then determined whether
issuing the loan was financially worthwhile: if the optimal expected
profit was positive, the loan was approved; if the expected profit was
negative, the loan was denied and the profit was set to zero. After
applying this strategy across all 1,000 loans in the test dataset, the
total expected profit was \$1,335,189. This total represents the return
expected under a policy that selects the most profitable rate for each
borrower and approves only those loans predicted to yield positive
returns.

Approve/deny decisions: Once the optimal interest rate and maximum
expected profit were identified for each applicant, the expected profit
value was used as the decision criterion. Applications with positive
expected profit were approved, while those with zero or negative profit
were denied. This approach ensures that every loan issued is estimated
to add value to the overall lending portfolio.

Risk management approach: The model’s risk management framework
integrates the predicted probability of default directly into the profit
calculation. For each borrower, the formula weighs potential earnings if
repayment occurs against potential losses in case of default. By
systematically identifying the offered rate that maximizes total profit,
the model balances expected return and risk without relying on arbitrary
thresholds. This structured, data-driven approach supports sustainable
lending decisions and long-term portfolio profitability.

# Model Interpretation

The feature importance analysis from the boosting model helped identify
which factors most strongly influence loan acceptance and default. For
loan acceptance, the most important factors were the offered interest
rate, income-to-loan ratio, and annual income. Borrowers with higher
income and a lower income-to-loan ratio were much more likely to accept
a loan offer, showing that affordability plays a major role in
decision-making. On the other hand, a higher offered rate made borrowers
less likely to accept, suggesting price sensitivity among applicants.
For loan default, the bar chart of feature importance showed that credit
utilization, income-to-loan ratio, and credit score were the top
predictors. Borrowers who used a large portion of their available credit
or had lower credit scores were more likely to default. Those with
stronger financial stability and a lower share of their income going
toward loan payments were less likely to do so. From a business
standpoint, these results highlight two key insights. First,
understanding what drives acceptance helps design competitive offers and
improve customer targeting. Second, knowing which factors increase
default risk helps in setting fair but profitable interest rates and
making smarter approval decisions. Overall, the findings show that
affordability, past credit behavior, and how borrowers manage their
credit are the most important factors in predicting both acceptance and
default.

# Conclusion

This project showed how combining financial reasoning with advanced
machine learning can improve credit decision-making. By using models
that learn patterns in borrower data, we were able to predict both who
is likely to accept a loan and who is likely to default with a high
degree of accuracy. Among all the models tested, the boosting model gave
the most consistent and reliable results, performing well on both
acceptance and default predictions. The process involved several key
steps—creating meaningful new features, carefully comparing model
performance, and using a profit-based metric to guide decisions.
Together, these steps built a system that tailors loan interest rates to
each borrower, approves only profitable loans, and manages overall risk
more effectively. From a business perspective, this approach can
increase total profit while maintaining fair lending practices and
financial stability. It provides a framework that can easily scale,
adapt to new data, and support smart, data-driven lending strategies
over time.

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(tidyr)
library(broom)
library(purrr)
library(ggplot2)
library(gbm)
```

    ## Loaded gbm 2.2.2

    ## This version of gbm is no longer under development. Consider transitioning to gbm3, https://github.com/gbm-developers/gbm3

``` r
library(pROC)
```

    ## Type 'citation("pROC")' for a citation.

    ## 
    ## Attaching package: 'pROC'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     cov, smooth, var

``` r
library(randomForest)
```

    ## randomForest 4.7-1.2

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    ## The following object is masked from 'package:dplyr':
    ## 
    ##     combine

``` r
train_data <- read.csv("/Users/lllllnx/Downloads/loan_training_data.csv")
```

``` r
data(train_data, package = "ISLR2")
```

    ## Warning in data(train_data, package = "ISLR2"): data set 'train_data' not found

``` r
head(train_data)
```

    ##   loan_id age annual_income employment_length home_ownership credit_score
    ## 1       1  54         51000          < 1 year           RENT          614
    ## 2       2  31         88000         3-5 years           RENT          631
    ## 3       3  42         59000         3-5 years           RENT          600
    ## 4       4  46         45000          < 1 year           RENT          614
    ## 5       5  43         40000         3-5 years            OWN          792
    ## 6       6  37         46000         3-5 years       MORTGAGE          850
    ##   num_credit_lines delinquencies_2yrs credit_utilization inquiries_6months
    ## 1                6                  0               31.9                 0
    ## 2                6                  1               21.7                 0
    ## 3                7                  0               37.3                 0
    ## 4               12                  0               16.1                 1
    ## 5                8                  2               55.9                 0
    ## 6                3                  1               36.2                 0
    ##       loan_purpose loan_amount offered_rate offer_accepted default
    ## 1   small_business       19000   0.17423249              0      NA
    ## 2   small_business       31000   0.15378363              1       0
    ## 3      credit_card       14000   0.12252312              0      NA
    ## 4 home_improvement       12000   0.16928202              0      NA
    ## 5              car        9000   0.09339519              1       0
    ## 6      credit_card        9000   0.10000000              1       0

``` r
#create new variables and clean the dataset by removing NAs and Infs
library(dplyr)
library(tidyr)
train_clean <- train_data %>% mutate(
  income_to_loan = annual_income / (loan_amount * (1 + offered_rate)),
  loan_to_credit_lines = loan_amount / num_credit_lines,
  inquiries_to_delinquencies = inquiries_6months * 4 / delinquencies_2yrs,
  delinquencies_to_credit_lines = delinquencies_2yrs / num_credit_lines
) %>%
  drop_na(income_to_loan, loan_to_credit_lines, 
          inquiries_to_delinquencies, delinquencies_to_credit_lines) %>%
  filter(if_all(everything(), ~ !is.infinite(.)))
```

``` r
train_clean$default <- as.factor(train_clean$default)
train_clean$offer_accepted <- as.factor(train_clean$offer_accepted)
```

# Logistic Regression Model

``` r
#build the logistic regression model for offer acceptance
model_logistic_acceptance <- glm(
  offer_accepted ~ age + annual_income + credit_score + num_credit_lines + delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies + delinquencies_to_credit_lines + offered_rate,
  data = train_clean,
  family = binomial
)
summary(model_logistic_acceptance)
```

    ## 
    ## Call:
    ## glm(formula = offer_accepted ~ age + annual_income + credit_score + 
    ##     num_credit_lines + delinquencies_2yrs + credit_utilization + 
    ##     loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines + 
    ##     inquiries_to_delinquencies + delinquencies_to_credit_lines + 
    ##     offered_rate, family = binomial, data = train_clean)
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)                    1.221e+01  1.664e+00   7.335 2.22e-13 ***
    ## age                           -4.353e-03  6.754e-03  -0.645  0.51924    
    ## annual_income                 -1.748e-05  2.570e-06  -6.805 1.01e-11 ***
    ## credit_score                  -4.555e-03  1.668e-03  -2.731  0.00632 ** 
    ## num_credit_lines              -2.100e-02  4.083e-02  -0.514  0.60703    
    ## delinquencies_2yrs            -1.450e-01  3.276e-01  -0.443  0.65812    
    ## credit_utilization             1.903e-02  3.939e-03   4.832 1.35e-06 ***
    ## loan_amount                    3.760e-05  1.618e-05   2.324  0.02011 *  
    ## inquiries_6months              5.131e-01  4.624e-01   1.110  0.26714    
    ## income_to_loan                -4.215e-03  9.431e-03  -0.447  0.65494    
    ## loan_to_credit_lines          -1.110e-05  8.365e-05  -0.133  0.89447    
    ## inquiries_to_delinquencies    -1.012e-01  1.216e-01  -0.832  0.40540    
    ## delinquencies_to_credit_lines  1.173e+00  1.815e+00   0.646  0.51825    
    ## offered_rate                  -6.968e+01  4.821e+00 -14.454  < 2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1691.5  on 1375  degrees of freedom
    ## Residual deviance: 1031.9  on 1362  degrees of freedom
    ## AIC: 1059.9
    ## 
    ## Number of Fisher Scoring iterations: 6

``` r
#make the coefficients readable
library(broom)
tr_clean_summary <- tidy(model_logistic_acceptance)
tr_clean_summary$odds_ratio <- exp(tr_clean_summary$estimate)
tr_clean_summary %>% select(term, estimate, odds_ratio, p.value)
```

    ## # A tibble: 14 × 4
    ##    term                             estimate odds_ratio  p.value
    ##    <chr>                               <dbl>      <dbl>    <dbl>
    ##  1 (Intercept)                    12.2        2.00 e+ 5 2.22e-13
    ##  2 age                            -0.00435    9.96 e- 1 5.19e- 1
    ##  3 annual_income                  -0.0000175  1.000e+ 0 1.01e-11
    ##  4 credit_score                   -0.00456    9.95 e- 1 6.32e- 3
    ##  5 num_credit_lines               -0.0210     9.79 e- 1 6.07e- 1
    ##  6 delinquencies_2yrs             -0.145      8.65 e- 1 6.58e- 1
    ##  7 credit_utilization              0.0190     1.02 e+ 0 1.35e- 6
    ##  8 loan_amount                     0.0000376  1.00 e+ 0 2.01e- 2
    ##  9 inquiries_6months               0.513      1.67 e+ 0 2.67e- 1
    ## 10 income_to_loan                 -0.00421    9.96 e- 1 6.55e- 1
    ## 11 loan_to_credit_lines           -0.0000111  1.000e+ 0 8.94e- 1
    ## 12 inquiries_to_delinquencies     -0.101      9.04 e- 1 4.05e- 1
    ## 13 delinquencies_to_credit_lines   1.17       3.23 e+ 0 5.18e- 1
    ## 14 offered_rate                  -69.7        5.48 e-31 2.38e-47

The logistic regression model for loan acceptance reveals that the
offered interest rate is by far the most significant predictor of
whether a borrower will accept a loan. Specifically, higher interest
rates sharply decrease the likelihood of acceptance, indicating that
borrowers are highly sensitive to the cost of borrowing. In addition to
the interest rate, variables such as credit utilization, annual income,
credit score, and loan amount also show measurable, though smaller,
effects on acceptance decisions. For instance, borrowers with higher
credit utilization are slightly more likely to accept the loan,
potentially reflecting greater reliance on credit. Conversely, higher
income slightly reduces the probability of acceptance, suggesting that
wealthier borrowers may be more selective or have alternative financing
options. Variables related to inquiries and delinquencies generally have
weaker impacts, implying that past credit behavior is less influential
than the immediate financial terms of the loan.

``` r
#generate probabilities 
train_predict_acc <- train_clean %>% 
  mutate(prob_logistic = predict(model_logistic_acceptance, type = "response"))

train_predict_acc %>% select(loan_id, annual_income, credit_score, credit_utilization, loan_amount, offer_accepted, offered_rate, prob_logistic) %>% arrange(desc(prob_logistic)) %>% head(50)
```

    ##    loan_id annual_income credit_score credit_utilization loan_amount
    ## 1     2517         53000          796               73.7       22000
    ## 2     4927         45000          801               72.0       22500
    ## 3      263         42000          726               61.1       21000
    ## 4     4419         63000          757               71.9       31500
    ## 5      720         15000          756               26.0        7500
    ## 6     2014         53000          737               21.5       26500
    ## 7     1073         35000          739               33.1       17500
    ## 8      103         19000          723               11.4        9500
    ## 9     3844         59000          776               48.7       10000
    ## 10    4587         50000          735               35.8       25000
    ## 11     508         18000          798               56.2        9000
    ## 12    4161         82000          740               45.3       12000
    ## 13    2064         94000          747               33.3       37000
    ## 14     334         44000          811               30.3       11000
    ## 15    2924         22000          810               16.6       11000
    ## 16    2985         32000          783               44.1       16000
    ## 17    4077        101000          733               24.3       24000
    ## 18    2584         22000          777                7.8        3000
    ## 19    1465         67000          726               63.8        7000
    ## 20    4018         79000          779               48.2       24000
    ## 21    1989         26000          758               19.8       13000
    ## 22     840         76000          760               50.9       37000
    ## 23    2427         44000          823               29.3       22000
    ## 24     810         48000          739               71.0       24000
    ## 25     412         53000          730               16.3       26500
    ## 26    1173         40000          722               10.9       20000
    ## 27    1102         41000          727               33.0       14000
    ## 28    1028         40000          779               74.5       20000
    ## 29    1830         56000          769               37.3       18000
    ## 30    1877        100000          751               63.3       27000
    ## 31    4891         40000          724               36.2        6000
    ## 32    4064         59000          724               52.5       29000
    ## 33       9         37000          757               54.7       18500
    ## 34    4534         18000          744               73.0        9000
    ## 35    2110         33000          759               49.6       16500
    ## 36    3599         40000          750               24.6        2000
    ## 37     289         90000          731               40.5       26000
    ## 38    2559         26000          764               33.8        5000
    ## 39    3414         71000          817               36.2        1000
    ## 40    1991         25000          737                9.4       12500
    ## 41    4132         33000          752               46.9        6000
    ## 42    2220         34000          746               23.5       17000
    ## 43    2084         18000          773                8.8        9000
    ## 44    2012         94000          765               46.1       29000
    ## 45    4737         23000          786               60.8       11500
    ## 46    1659         17000          742               73.5        8500
    ## 47     534         58000          733               70.7       28000
    ## 48    3016         25000          778               42.2       12500
    ## 49    3554         50000          724               61.8       18000
    ## 50    3816         56000          762               30.9       16000
    ##    offer_accepted offered_rate prob_logistic
    ## 1               1   0.02000000     0.9998340
    ## 2               1   0.02000000     0.9997576
    ## 3               1   0.02000000     0.9997286
    ## 4               1   0.03310604     0.9996501
    ## 5               1   0.02000000     0.9996100
    ## 6               1   0.02369931     0.9995719
    ## 7               1   0.04019695     0.9994850
    ## 8               1   0.02000000     0.9994484
    ## 9               1   0.02000000     0.9993394
    ## 10              1   0.03291836     0.9992402
    ## 11              1   0.03483998     0.9992029
    ## 12              1   0.02446730     0.9991990
    ## 13              1   0.02300982     0.9991410
    ## 14              1   0.02000000     0.9991261
    ## 15              1   0.02814422     0.9989836
    ## 16              1   0.03317500     0.9989680
    ## 17              1   0.02000000     0.9989543
    ## 18              1   0.02000000     0.9989197
    ## 19              1   0.04619982     0.9989160
    ## 20              1   0.02847946     0.9989075
    ## 21              1   0.02973535     0.9989001
    ## 22              1   0.04328775     0.9986468
    ## 23              1   0.03173665     0.9986350
    ## 24              1   0.05069384     0.9985920
    ## 25              1   0.04040458     0.9984754
    ## 26              1   0.03897487     0.9984033
    ## 27              1   0.03726515     0.9983873
    ## 28              1   0.04850960     0.9983178
    ## 29              1   0.03968979     0.9980783
    ## 30              1   0.04169688     0.9980383
    ## 31              1   0.03851999     0.9978733
    ## 32              1   0.05258735     0.9978547
    ## 33              1   0.04555872     0.9978475
    ## 34              1   0.05401014     0.9977989
    ## 35              1   0.05255369     0.9977283
    ## 36              1   0.03911240     0.9976629
    ## 37              1   0.03915966     0.9975992
    ## 38              1   0.03861049     0.9975080
    ## 39              1   0.02000000     0.9973937
    ## 40              1   0.04083359     0.9973490
    ## 41              1   0.04342353     0.9973222
    ## 42              1   0.05126997     0.9973074
    ## 43              1   0.04378781     0.9972205
    ## 44              1   0.04273482     0.9970877
    ## 45              1   0.05112767     0.9970797
    ## 46              1   0.06257865     0.9969903
    ## 47              1   0.05597039     0.9969496
    ## 48              1   0.06363009     0.9968598
    ## 49              1   0.05431188     0.9967850
    ## 50              1   0.04492379     0.9967252

``` r
#compute the confusion matrix to classify predictions
train_predict_acc$pred_logistic <- factor(ifelse(train_predict_acc$prob_logistic>0.5,"1","0"), levels=c("0","1")) #since the highest probability is around 1, I set 0.5 as the threshold here

table(Predicted=train_predict_acc$pred_logistic, Actual=train_predict_acc$offer_accepted)
```

    ##          Actual
    ## Predicted   0   1
    ##         0 256  77
    ##         1 163 880

``` r
#compute performance metrics
cm <- table(Predicted = train_predict_acc$pred_logistic,
            Actual = train_predict_acc$offer_accepted)

TN <- cm["0", "0"]
FP <- cm["1", "0"]
FN <- cm["0", "1"]
TP <- cm["1", "1"]

accuracy    <- (TP + TN) / (TP + TN + FP + FN)
precision   <- TP / (TP + FP)
recall      <- TP / (TP + FN) #how many we catch it among who actually defaulted
specificity <- TN / (TN + FP) #how many we mark as safe among who didn't default

metrics_acc <- data.frame(
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  specificity = specificity
)

metrics_acc
```

    ##    accuracy precision    recall specificity
    ## 1 0.8255814   0.84372 0.9195402   0.6109785

The performance of the logistic regression model for predicting loan
acceptance shows that it is generally effective. The model achieves an
overall accuracy of 82.6%, meaning it correctly classifies the majority
of loans. Its precision of 84.4% indicates that most loans predicted as
accepted are indeed accepted, while the recall of 92.0% shows the model
successfully identifies nearly all actual accepted loans. The
specificity of 61.1% is lower, meaning the model is less accurate at
predicting rejections. Overall, the model is particularly strong at
identifying likely borrowers, though it tends to overpredict acceptance
slightly.

``` r
library(purrr)     
library(ggplot2)   

#define the helper
calculate_roc <- function(probabilities, actual_outcomes) {
  thresholds <- seq(0, 1, by = 0.01)
  
  roc_data <- map_dfr(thresholds, function(thresh) {
    predictions <- ifelse(probabilities > thresh, 1, 0)
    
    tp <- sum(predictions == 1 & actual_outcomes == 1)
    fp <- sum(predictions == 1 & actual_outcomes == 0)
    tn <- sum(predictions == 0 & actual_outcomes == 0)
    fn <- sum(predictions == 0 & actual_outcomes == 1)
    
    tpr <- tp / (tp + fn)  
    fpr <- fp / (fp + tn)  
    
    data.frame(threshold = thresh, tpr = tpr, fpr = fpr)
  })
  
  roc_data <- roc_data[order(roc_data$fpr), ]
  auc <- sum(diff(roc_data$fpr) *
               (roc_data$tpr[-1] + roc_data$tpr[-length(roc_data$tpr)])) / 2
  
  list(roc_data = roc_data, auc = auc)
}
```

``` r
# input acceptance data from my dataset
actual_numeric_acc <- ifelse(train_predict_acc$offer_accepted == "1", 1, 0)
roc_result_acc <- calculate_roc(
  probabilities   = train_predict_acc$prob_logistic,  
  actual_outcomes = actual_numeric_acc                
)
roc_data_acc  <- roc_result_acc$roc_data
auc_value_acc <- roc_result_acc$auc

auc_value_acc  
```

    ## [1] 0.8888881

``` r
#plot ROC curve 
ggplot(roc_data_acc, aes(x = fpr, y = tpr)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = paste0("ROC Curve_Acceptance_Logit Regression (AUC = ", round(auc_value_acc, 4), ")"),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal()
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
#select all accepted offers by removing NA rows in default 
train_clean_all_accepted <- train_clean %>%
  drop_na(default)
```

``` r
#build the logistic regression model for default 
model_logistic_def <- glm(
  default ~ age + annual_income + credit_score + num_credit_lines + delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies + delinquencies_to_credit_lines + offered_rate,
  data = train_clean_all_accepted,
  family = binomial
)
summary(model_logistic_def)
```

    ## 
    ## Call:
    ## glm(formula = default ~ age + annual_income + credit_score + 
    ##     num_credit_lines + delinquencies_2yrs + credit_utilization + 
    ##     loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines + 
    ##     inquiries_to_delinquencies + delinquencies_to_credit_lines + 
    ##     offered_rate, family = binomial, data = train_clean_all_accepted)
    ## 
    ## Coefficients:
    ##                                 Estimate Std. Error z value Pr(>|z|)  
    ## (Intercept)                    1.473e-01  1.581e+00   0.093   0.9258  
    ## age                            7.550e-03  6.725e-03   1.123   0.2615  
    ## annual_income                 -4.121e-06  3.491e-06  -1.180   0.2379  
    ## credit_score                  -2.201e-03  1.678e-03  -1.312   0.1896  
    ## num_credit_lines              -3.713e-02  4.117e-02  -0.902   0.3671  
    ## delinquencies_2yrs             1.779e-01  3.461e-01   0.514   0.6073  
    ## credit_utilization             9.464e-03  3.857e-03   2.454   0.0141 *
    ## loan_amount                   -2.665e-05  1.842e-05  -1.447   0.1480  
    ## inquiries_6months             -9.572e-01  5.340e-01  -1.792   0.0731 .
    ## income_to_loan                -2.483e-02  1.590e-02  -1.562   0.1182  
    ## loan_to_credit_lines           3.488e-05  8.686e-05   0.402   0.6880  
    ## inquiries_to_delinquencies     2.838e-01  1.383e-01   2.052   0.0402 *
    ## delinquencies_to_credit_lines -3.736e-01  1.573e+00  -0.238   0.8122  
    ## offered_rate                   2.273e+00  3.874e+00   0.587   0.5574  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## (Dispersion parameter for binomial family taken to be 1)
    ## 
    ##     Null deviance: 1041.5  on 956  degrees of freedom
    ## Residual deviance: 1001.2  on 943  degrees of freedom
    ## AIC: 1029.2
    ## 
    ## Number of Fisher Scoring iterations: 4

The default model predicts the likelihood that a borrower will fail to
repay a loan. The results show that credit score, annual income, and
credit utilization are the most important factors. Borrowers with higher
credit scores and incomes are less likely to default, while those using
a larger portion of their available credit are more likely to default.
The model does well at identifying borrowers who will not default, but
it is less accurate at catching all defaults. These insights can help
guide lending decisions and manage risk.

``` r
tr_clean_summary_def <- tidy(model_logistic_def)
tr_clean_summary_def$odds_ratio <- exp(tr_clean_summary_def$estimate)
tr_clean_summary_def %>% select(term, estimate, odds_ratio, p.value)
```

    ## # A tibble: 14 × 4
    ##    term                             estimate odds_ratio p.value
    ##    <chr>                               <dbl>      <dbl>   <dbl>
    ##  1 (Intercept)                    0.147           1.16   0.926 
    ##  2 age                            0.00755         1.01   0.262 
    ##  3 annual_income                 -0.00000412      1.000  0.238 
    ##  4 credit_score                  -0.00220         0.998  0.190 
    ##  5 num_credit_lines              -0.0371          0.964  0.367 
    ##  6 delinquencies_2yrs             0.178           1.19   0.607 
    ##  7 credit_utilization             0.00946         1.01   0.0141
    ##  8 loan_amount                   -0.0000267       1.000  0.148 
    ##  9 inquiries_6months             -0.957           0.384  0.0731
    ## 10 income_to_loan                -0.0248          0.975  0.118 
    ## 11 loan_to_credit_lines           0.0000349       1.00   0.688 
    ## 12 inquiries_to_delinquencies     0.284           1.33   0.0402
    ## 13 delinquencies_to_credit_lines -0.374           0.688  0.812 
    ## 14 offered_rate                   2.27            9.70   0.557

``` r
train_predict_def <- train_clean_all_accepted %>% 
  mutate(prob_logistic = predict(model_logistic_def, type = "response"))
train_predict_def %>% select(loan_id, credit_utilization, inquiries_to_delinquencies, default, prob_logistic) %>% arrange(desc(prob_logistic)) %>% head(50)
```

    ##    loan_id credit_utilization inquiries_to_delinquencies default prob_logistic
    ## 1     1605               55.5                         16       1     0.5857006
    ## 2     2563               73.4                          4       1     0.5035661
    ## 3     2267               55.9                         12       0     0.4893222
    ## 4     1319               80.8                         12       1     0.4820373
    ## 5     1867               57.8                          4       0     0.4804156
    ## 6     2262               20.0                          8       1     0.4738620
    ## 7      169               68.8                         20       1     0.4642096
    ## 8     3078               29.8                         16       0     0.4611931
    ## 9     1689               53.5                          8       0     0.4601178
    ## 10    3625               42.9                         12       0     0.4572808
    ## 11    1441               79.4                          8       1     0.4523338
    ## 12     170               82.0                          4       0     0.4513518
    ## 13     978               62.8                          4       1     0.4505283
    ## 14    1857               85.2                          8       0     0.4465643
    ## 15    2859               90.9                          4       1     0.4456983
    ## 16    4733               73.9                          0       1     0.4453841
    ## 17     930               74.5                         16       0     0.4412302
    ## 18    3266               67.1                          8       0     0.4394380
    ## 19    3777               58.6                          8       0     0.4383319
    ## 20    2215               92.3                          0       1     0.4381085
    ## 21    4011               40.3                          8       1     0.4374103
    ## 22    2346               66.2                          4       1     0.4358511
    ## 23     458               80.2                          0       0     0.4307945
    ## 24     823               51.8                         12       1     0.4197385
    ## 25    3318               71.7                          4       0     0.4181408
    ## 26    3577               66.8                          8       0     0.4161986
    ## 27    4124               71.1                          8       0     0.4161623
    ## 28    2387               72.7                          0       1     0.4086343
    ## 29    2691               33.1                         12       0     0.4073537
    ## 30    1124               52.0                          8       1     0.4043074
    ## 31    4599               66.0                          0       0     0.4004335
    ## 32    1253               60.9                          4       0     0.3983964
    ## 33    3340               81.1                          0       0     0.3970819
    ## 34    2615               73.5                          8       1     0.3964333
    ## 35    2154               50.6                          4       0     0.3961652
    ## 36    2954               78.6                          0       1     0.3959057
    ## 37    2903               48.4                         12       0     0.3952469
    ## 38    1819               62.1                          8       1     0.3939697
    ## 39    3551                8.7                          8       0     0.3935621
    ## 40    4782               66.1                         12       1     0.3933245
    ## 41    4359               69.2                          8       0     0.3932834
    ## 42    1456               59.5                          8       0     0.3929445
    ## 43    1736               63.0                          4       1     0.3884824
    ## 44    3289               38.6                          8       0     0.3883927
    ## 45    3253               74.8                          8       1     0.3877389
    ## 46    2681               20.6                         12       0     0.3877073
    ## 47     997               55.9                          8       1     0.3866478
    ## 48    1340               65.5                          4       1     0.3866382
    ## 49    2731               74.4                          8       0     0.3842512
    ## 50    4852               74.5                          8       0     0.3835141

``` r
train_predict_def$pred_logistic <- factor(ifelse(train_predict_def$prob_logistic>0.3,"1","0"), levels=c("0","1")) #since the highest probability is around 0.6, I set 0.3 as the threshold here

table(Predicted=train_predict_def$pred_logistic, Actual=train_predict_def$default)
```

    ##          Actual
    ## Predicted   0   1
    ##         0 598 151
    ##         1 135  73

``` r
cm <- table(Predicted = train_predict_def$pred_logistic,
            Actual = train_predict_def$default)

TN <- cm["0", "0"]
FP <- cm["1", "0"]
FN <- cm["0", "1"]
TP <- cm["1", "1"]

accuracy    <- (TP + TN) / (TP + TN + FP + FN)
precision   <- TP / (TP + FP)
recall      <- TP / (TP + FN) #how many we catch it among who actually defaulted
specificity <- TN / (TN + FP) #how many we mark as safe among who didn't default

metrics <- data.frame(
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  specificity = specificity
)

metrics
```

    ##    accuracy precision    recall specificity
    ## 1 0.7011494 0.3509615 0.3258929   0.8158254

``` r
actual_numeric_def <- ifelse(train_predict_def$default == "1", 1, 0)
roc_result_def <- calculate_roc(
  probabilities   = train_predict_def$prob_logistic,  
  actual_outcomes = actual_numeric_def                
)
roc_data_def  <- roc_result_def$roc_data
auc_value_def <- roc_result_def$auc

auc_value_def   
```

    ## [1] 0.6241199

``` r
library(ggplot2)
ggplot(roc_data_def, aes(x = fpr, y = tpr)) +
  geom_line() +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(
    title = paste0("ROC Curve_Default_Logit Regression (AUC = ", round(auc_value_def, 4), ")"),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal()
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-21-1.png)<!-- -->

# Random Forest

``` r
#Acceptance - random forest
set.seed(123)
rf_accept <- randomForest(
  offer_accepted ~ age + annual_income + credit_score + num_credit_lines +
    delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months +
    income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies +
    delinquencies_to_credit_lines,
  data = train_clean,
  ntree = 1000,
  mtry = floor(sqrt(ncol(train_clean))),
  importance = TRUE,
  keep.inbag = TRUE
)

rf_accept
```

    ## 
    ## Call:
    ##  randomForest(formula = offer_accepted ~ age + annual_income +      credit_score + num_credit_lines + delinquencies_2yrs + credit_utilization +      loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines +      inquiries_to_delinquencies + delinquencies_to_credit_lines,      data = train_clean, ntree = 1000, mtry = floor(sqrt(ncol(train_clean))),      importance = TRUE, keep.inbag = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 1000
    ## No. of variables tried at each split: 4
    ## 
    ##         OOB estimate of  error rate: 24.49%
    ## Confusion matrix:
    ##     0   1 class.error
    ## 0 205 214   0.5107399
    ## 1 123 834   0.1285266

``` r
rf_accept$confusion
```

    ##     0   1 class.error
    ## 0 205 214   0.5107399
    ## 1 123 834   0.1285266

``` r
rf_accept$err.rate
```

    ##               OOB         0         1
    ##    [1,] 0.2657480 0.4729730 0.1805556
    ##    [2,] 0.3046117 0.4722222 0.2307692
    ##    [3,] 0.3053659 0.4660194 0.2360335
    ##    [4,] 0.2969697 0.4535211 0.2275000
    ##    [5,] 0.2904264 0.4289474 0.2294322
    ##    [6,] 0.2902477 0.4314721 0.2282851
    ##    [7,] 0.2811791 0.4154229 0.2225841
    ##    [8,] 0.2870992 0.4570025 0.2130621
    ##    [9,] 0.2833948 0.4576271 0.2070064
    ##   [10,] 0.2694567 0.4202899 0.2035865
    ##   [11,] 0.2648569 0.4106280 0.2012645
    ##   [12,] 0.2778993 0.4484412 0.2033543
    ##   [13,] 0.2633115 0.4196643 0.1949686
    ##   [14,] 0.2616618 0.4388489 0.1842932
    ##   [15,] 0.2629279 0.4292566 0.1903766
    ##   [16,] 0.2731245 0.4748201 0.1851464
    ##   [17,] 0.2743814 0.4593301 0.1935146
    ##   [18,] 0.2683636 0.4641148 0.1828631
    ##   [19,] 0.2661818 0.4569378 0.1828631
    ##   [20,] 0.2712727 0.4688995 0.1849530
    ##   [21,] 0.2698182 0.4521531 0.1901776
    ##   [22,] 0.2690909 0.4545455 0.1880878
    ##   [23,] 0.2674419 0.4606205 0.1828631
    ##   [24,] 0.2623547 0.4439141 0.1828631
    ##   [25,] 0.2623547 0.4534606 0.1786834
    ##   [26,] 0.2572674 0.4510740 0.1724138
    ##   [27,] 0.2609012 0.4558473 0.1755486
    ##   [28,] 0.2550872 0.4486874 0.1703239
    ##   [29,] 0.2579942 0.4582339 0.1703239
    ##   [30,] 0.2579942 0.4653938 0.1671891
    ##   [31,] 0.2558140 0.4486874 0.1713689
    ##   [32,] 0.2558140 0.4582339 0.1671891
    ##   [33,] 0.2572674 0.4677804 0.1650993
    ##   [34,] 0.2601744 0.4725537 0.1671891
    ##   [35,] 0.2630814 0.4749403 0.1703239
    ##   [36,] 0.2594477 0.4701671 0.1671891
    ##   [37,] 0.2587209 0.4701671 0.1661442
    ##   [38,] 0.2558140 0.4749403 0.1598746
    ##   [39,] 0.2587209 0.4773270 0.1630094
    ##   [40,] 0.2565407 0.4749403 0.1609195
    ##   [41,] 0.2579942 0.4797136 0.1609195
    ##   [42,] 0.2550872 0.4749403 0.1588297
    ##   [43,] 0.2572674 0.4821002 0.1588297
    ##   [44,] 0.2579942 0.4797136 0.1609195
    ##   [45,] 0.2565407 0.4797136 0.1588297
    ##   [46,] 0.2514535 0.4701671 0.1556949
    ##   [47,] 0.2507267 0.4749403 0.1525601
    ##   [48,] 0.2529070 0.4773270 0.1546499
    ##   [49,] 0.2558140 0.4844869 0.1556949
    ##   [50,] 0.2550872 0.4821002 0.1556949
    ##   [51,] 0.2572674 0.4868735 0.1567398
    ##   [52,] 0.2543605 0.4797136 0.1556949
    ##   [53,] 0.2565407 0.4892601 0.1546499
    ##   [54,] 0.2550872 0.4844869 0.1546499
    ##   [55,] 0.2470930 0.4773270 0.1462905
    ##   [56,] 0.2558140 0.4773270 0.1588297
    ##   [57,] 0.2572674 0.4797136 0.1598746
    ##   [58,] 0.2521802 0.4653938 0.1588297
    ##   [59,] 0.2507267 0.4677804 0.1556949
    ##   [60,] 0.2478198 0.4630072 0.1536050
    ##   [61,] 0.2456395 0.4606205 0.1515152
    ##   [62,] 0.2470930 0.4701671 0.1494253
    ##   [63,] 0.2478198 0.4677804 0.1515152
    ##   [64,] 0.2470930 0.4630072 0.1525601
    ##   [65,] 0.2492733 0.4653938 0.1546499
    ##   [66,] 0.2478198 0.4749403 0.1483804
    ##   [67,] 0.2492733 0.4749403 0.1504702
    ##   [68,] 0.2565407 0.4916468 0.1536050
    ##   [69,] 0.2529070 0.4892601 0.1494253
    ##   [70,] 0.2521802 0.4892601 0.1483804
    ##   [71,] 0.2529070 0.4821002 0.1525601
    ##   [72,] 0.2550872 0.4868735 0.1536050
    ##   [73,] 0.2536337 0.4797136 0.1546499
    ##   [74,] 0.2521802 0.4821002 0.1515152
    ##   [75,] 0.2529070 0.4844869 0.1515152
    ##   [76,] 0.2558140 0.4892601 0.1536050
    ##   [77,] 0.2558140 0.4844869 0.1556949
    ##   [78,] 0.2565407 0.4964200 0.1515152
    ##   [79,] 0.2587209 0.4988067 0.1536050
    ##   [80,] 0.2594477 0.4964200 0.1556949
    ##   [81,] 0.2543605 0.4868735 0.1525601
    ##   [82,] 0.2565407 0.4940334 0.1525601
    ##   [83,] 0.2550872 0.4868735 0.1536050
    ##   [84,] 0.2521802 0.4868735 0.1494253
    ##   [85,] 0.2514535 0.4892601 0.1473354
    ##   [86,] 0.2521802 0.4868735 0.1494253
    ##   [87,] 0.2529070 0.4868735 0.1504702
    ##   [88,] 0.2543605 0.4940334 0.1494253
    ##   [89,] 0.2543605 0.4892601 0.1515152
    ##   [90,] 0.2550872 0.4988067 0.1483804
    ##   [91,] 0.2521802 0.5011933 0.1431557
    ##   [92,] 0.2521802 0.4988067 0.1442006
    ##   [93,] 0.2507267 0.4940334 0.1442006
    ##   [94,] 0.2514535 0.4988067 0.1431557
    ##   [95,] 0.2521802 0.5059666 0.1410658
    ##   [96,] 0.2500000 0.4988067 0.1410658
    ##   [97,] 0.2529070 0.5083532 0.1410658
    ##   [98,] 0.2492733 0.5011933 0.1389760
    ##   [99,] 0.2514535 0.5011933 0.1421108
    ##  [100,] 0.2521802 0.4988067 0.1442006
    ##  [101,] 0.2485465 0.5011933 0.1379310
    ##  [102,] 0.2529070 0.5083532 0.1410658
    ##  [103,] 0.2521802 0.5035800 0.1421108
    ##  [104,] 0.2507267 0.4988067 0.1421108
    ##  [105,] 0.2492733 0.5035800 0.1379310
    ##  [106,] 0.2558140 0.5202864 0.1400209
    ##  [107,] 0.2521802 0.5083532 0.1400209
    ##  [108,] 0.2507267 0.5083532 0.1379310
    ##  [109,] 0.2543605 0.5107399 0.1421108
    ##  [110,] 0.2478198 0.5083532 0.1337513
    ##  [111,] 0.2521802 0.5083532 0.1400209
    ##  [112,] 0.2470930 0.5011933 0.1358412
    ##  [113,] 0.2536337 0.5155131 0.1389760
    ##  [114,] 0.2514535 0.5178998 0.1347962
    ##  [115,] 0.2500000 0.5131265 0.1347962
    ##  [116,] 0.2492733 0.5131265 0.1337513
    ##  [117,] 0.2485465 0.5107399 0.1337513
    ##  [118,] 0.2514535 0.5107399 0.1379310
    ##  [119,] 0.2500000 0.5107399 0.1358412
    ##  [120,] 0.2514535 0.5131265 0.1368861
    ##  [121,] 0.2500000 0.5107399 0.1358412
    ##  [122,] 0.2492733 0.5107399 0.1347962
    ##  [123,] 0.2500000 0.5059666 0.1379310
    ##  [124,] 0.2514535 0.5107399 0.1379310
    ##  [125,] 0.2536337 0.5131265 0.1400209
    ##  [126,] 0.2514535 0.5107399 0.1379310
    ##  [127,] 0.2507267 0.5107399 0.1368861
    ##  [128,] 0.2500000 0.5059666 0.1379310
    ##  [129,] 0.2521802 0.5059666 0.1410658
    ##  [130,] 0.2529070 0.5131265 0.1389760
    ##  [131,] 0.2543605 0.5155131 0.1400209
    ##  [132,] 0.2514535 0.5131265 0.1368861
    ##  [133,] 0.2550872 0.5155131 0.1410658
    ##  [134,] 0.2529070 0.5131265 0.1389760
    ##  [135,] 0.2521802 0.5131265 0.1379310
    ##  [136,] 0.2529070 0.5107399 0.1400209
    ##  [137,] 0.2565407 0.5226730 0.1400209
    ##  [138,] 0.2536337 0.5202864 0.1368861
    ##  [139,] 0.2492733 0.5059666 0.1368861
    ##  [140,] 0.2485465 0.5059666 0.1358412
    ##  [141,] 0.2529070 0.5131265 0.1389760
    ##  [142,] 0.2529070 0.5107399 0.1400209
    ##  [143,] 0.2536337 0.5083532 0.1421108
    ##  [144,] 0.2521802 0.5131265 0.1379310
    ##  [145,] 0.2536337 0.5131265 0.1400209
    ##  [146,] 0.2507267 0.5107399 0.1368861
    ##  [147,] 0.2514535 0.5131265 0.1368861
    ##  [148,] 0.2485465 0.5083532 0.1347962
    ##  [149,] 0.2514535 0.5131265 0.1368861
    ##  [150,] 0.2485465 0.5035800 0.1368861
    ##  [151,] 0.2492733 0.5083532 0.1358412
    ##  [152,] 0.2449128 0.5035800 0.1316614
    ##  [153,] 0.2500000 0.5107399 0.1358412
    ##  [154,] 0.2485465 0.5083532 0.1347962
    ##  [155,] 0.2543605 0.5131265 0.1410658
    ##  [156,] 0.2507267 0.5107399 0.1368861
    ##  [157,] 0.2521802 0.5107399 0.1389760
    ##  [158,] 0.2507267 0.5107399 0.1368861
    ##  [159,] 0.2500000 0.5059666 0.1379310
    ##  [160,] 0.2492733 0.5059666 0.1368861
    ##  [161,] 0.2485465 0.5083532 0.1347962
    ##  [162,] 0.2470930 0.5035800 0.1347962
    ##  [163,] 0.2463663 0.5059666 0.1327064
    ##  [164,] 0.2456395 0.4988067 0.1347962
    ##  [165,] 0.2470930 0.4964200 0.1379310
    ##  [166,] 0.2492733 0.5011933 0.1389760
    ##  [167,] 0.2507267 0.5083532 0.1379310
    ##  [168,] 0.2492733 0.5035800 0.1379310
    ##  [169,] 0.2485465 0.4988067 0.1389760
    ##  [170,] 0.2470930 0.5011933 0.1358412
    ##  [171,] 0.2500000 0.5059666 0.1379310
    ##  [172,] 0.2485465 0.5011933 0.1379310
    ##  [173,] 0.2492733 0.5035800 0.1379310
    ##  [174,] 0.2500000 0.5059666 0.1379310
    ##  [175,] 0.2500000 0.5083532 0.1368861
    ##  [176,] 0.2456395 0.5011933 0.1337513
    ##  [177,] 0.2470930 0.5035800 0.1347962
    ##  [178,] 0.2478198 0.5035800 0.1358412
    ##  [179,] 0.2470930 0.5035800 0.1347962
    ##  [180,] 0.2470930 0.5035800 0.1347962
    ##  [181,] 0.2463663 0.4988067 0.1358412
    ##  [182,] 0.2500000 0.5083532 0.1368861
    ##  [183,] 0.2521802 0.5107399 0.1389760
    ##  [184,] 0.2500000 0.5059666 0.1379310
    ##  [185,] 0.2492733 0.5107399 0.1347962
    ##  [186,] 0.2514535 0.5155131 0.1358412
    ##  [187,] 0.2514535 0.5131265 0.1368861
    ##  [188,] 0.2565407 0.5226730 0.1400209
    ##  [189,] 0.2550872 0.5155131 0.1410658
    ##  [190,] 0.2543605 0.5155131 0.1400209
    ##  [191,] 0.2529070 0.5107399 0.1400209
    ##  [192,] 0.2507267 0.5107399 0.1368861
    ##  [193,] 0.2514535 0.5131265 0.1368861
    ##  [194,] 0.2492733 0.5083532 0.1358412
    ##  [195,] 0.2463663 0.5035800 0.1337513
    ##  [196,] 0.2441860 0.4940334 0.1347962
    ##  [197,] 0.2485465 0.5059666 0.1358412
    ##  [198,] 0.2500000 0.5107399 0.1358412
    ##  [199,] 0.2500000 0.5059666 0.1379310
    ##  [200,] 0.2529070 0.5155131 0.1379310
    ##  [201,] 0.2529070 0.5131265 0.1389760
    ##  [202,] 0.2500000 0.5107399 0.1358412
    ##  [203,] 0.2536337 0.5155131 0.1389760
    ##  [204,] 0.2529070 0.5131265 0.1389760
    ##  [205,] 0.2507267 0.5107399 0.1368861
    ##  [206,] 0.2500000 0.5083532 0.1368861
    ##  [207,] 0.2507267 0.5059666 0.1389760
    ##  [208,] 0.2500000 0.5035800 0.1389760
    ##  [209,] 0.2500000 0.5083532 0.1368861
    ##  [210,] 0.2521802 0.5107399 0.1389760
    ##  [211,] 0.2507267 0.5059666 0.1389760
    ##  [212,] 0.2500000 0.5035800 0.1389760
    ##  [213,] 0.2500000 0.5059666 0.1379310
    ##  [214,] 0.2507267 0.5083532 0.1379310
    ##  [215,] 0.2521802 0.5107399 0.1389760
    ##  [216,] 0.2500000 0.5035800 0.1389760
    ##  [217,] 0.2521802 0.5083532 0.1400209
    ##  [218,] 0.2507267 0.5035800 0.1400209
    ##  [219,] 0.2536337 0.5083532 0.1421108
    ##  [220,] 0.2543605 0.5107399 0.1421108
    ##  [221,] 0.2521802 0.5083532 0.1400209
    ##  [222,] 0.2550872 0.5131265 0.1421108
    ##  [223,] 0.2521802 0.5083532 0.1400209
    ##  [224,] 0.2521802 0.5035800 0.1421108
    ##  [225,] 0.2507267 0.5011933 0.1410658
    ##  [226,] 0.2507267 0.5035800 0.1400209
    ##  [227,] 0.2500000 0.5035800 0.1389760
    ##  [228,] 0.2500000 0.5011933 0.1400209
    ##  [229,] 0.2507267 0.5011933 0.1410658
    ##  [230,] 0.2507267 0.5011933 0.1410658
    ##  [231,] 0.2507267 0.5011933 0.1410658
    ##  [232,] 0.2514535 0.5035800 0.1410658
    ##  [233,] 0.2485465 0.4940334 0.1410658
    ##  [234,] 0.2485465 0.4964200 0.1400209
    ##  [235,] 0.2478198 0.4940334 0.1400209
    ##  [236,] 0.2492733 0.5011933 0.1389760
    ##  [237,] 0.2485465 0.4940334 0.1410658
    ##  [238,] 0.2478198 0.4964200 0.1389760
    ##  [239,] 0.2492733 0.5059666 0.1368861
    ##  [240,] 0.2492733 0.5011933 0.1389760
    ##  [241,] 0.2485465 0.5011933 0.1379310
    ##  [242,] 0.2492733 0.5011933 0.1389760
    ##  [243,] 0.2478198 0.4964200 0.1389760
    ##  [244,] 0.2485465 0.4988067 0.1389760
    ##  [245,] 0.2485465 0.5011933 0.1379310
    ##  [246,] 0.2500000 0.5011933 0.1400209
    ##  [247,] 0.2500000 0.5035800 0.1389760
    ##  [248,] 0.2507267 0.5083532 0.1379310
    ##  [249,] 0.2485465 0.5059666 0.1358412
    ##  [250,] 0.2492733 0.5035800 0.1379310
    ##  [251,] 0.2500000 0.5059666 0.1379310
    ##  [252,] 0.2492733 0.5083532 0.1358412
    ##  [253,] 0.2478198 0.5011933 0.1368861
    ##  [254,] 0.2507267 0.5059666 0.1389760
    ##  [255,] 0.2500000 0.5059666 0.1379310
    ##  [256,] 0.2485465 0.4964200 0.1400209
    ##  [257,] 0.2470930 0.4988067 0.1368861
    ##  [258,] 0.2478198 0.5035800 0.1358412
    ##  [259,] 0.2463663 0.5011933 0.1347962
    ##  [260,] 0.2485465 0.5083532 0.1347962
    ##  [261,] 0.2478198 0.5059666 0.1347962
    ##  [262,] 0.2500000 0.5107399 0.1358412
    ##  [263,] 0.2492733 0.5107399 0.1347962
    ##  [264,] 0.2485465 0.5083532 0.1347962
    ##  [265,] 0.2492733 0.5083532 0.1358412
    ##  [266,] 0.2485465 0.5107399 0.1337513
    ##  [267,] 0.2485465 0.5107399 0.1337513
    ##  [268,] 0.2529070 0.5202864 0.1358412
    ##  [269,] 0.2492733 0.5107399 0.1347962
    ##  [270,] 0.2529070 0.5178998 0.1368861
    ##  [271,] 0.2521802 0.5131265 0.1379310
    ##  [272,] 0.2521802 0.5178998 0.1358412
    ##  [273,] 0.2521802 0.5155131 0.1368861
    ##  [274,] 0.2507267 0.5131265 0.1358412
    ##  [275,] 0.2507267 0.5155131 0.1347962
    ##  [276,] 0.2536337 0.5202864 0.1368861
    ##  [277,] 0.2521802 0.5178998 0.1358412
    ##  [278,] 0.2521802 0.5178998 0.1358412
    ##  [279,] 0.2514535 0.5178998 0.1347962
    ##  [280,] 0.2507267 0.5131265 0.1358412
    ##  [281,] 0.2514535 0.5155131 0.1358412
    ##  [282,] 0.2500000 0.5131265 0.1347962
    ##  [283,] 0.2500000 0.5131265 0.1347962
    ##  [284,] 0.2500000 0.5131265 0.1347962
    ##  [285,] 0.2470930 0.5107399 0.1316614
    ##  [286,] 0.2492733 0.5107399 0.1347962
    ##  [287,] 0.2500000 0.5131265 0.1347962
    ##  [288,] 0.2492733 0.5107399 0.1347962
    ##  [289,] 0.2500000 0.5131265 0.1347962
    ##  [290,] 0.2500000 0.5155131 0.1337513
    ##  [291,] 0.2492733 0.5131265 0.1337513
    ##  [292,] 0.2485465 0.5107399 0.1337513
    ##  [293,] 0.2485465 0.5131265 0.1327064
    ##  [294,] 0.2485465 0.5107399 0.1337513
    ##  [295,] 0.2478198 0.5107399 0.1327064
    ##  [296,] 0.2470930 0.5083532 0.1327064
    ##  [297,] 0.2478198 0.5083532 0.1337513
    ##  [298,] 0.2485465 0.5083532 0.1347962
    ##  [299,] 0.2492733 0.5059666 0.1368861
    ##  [300,] 0.2492733 0.5083532 0.1358412
    ##  [301,] 0.2492733 0.5083532 0.1358412
    ##  [302,] 0.2492733 0.5083532 0.1358412
    ##  [303,] 0.2500000 0.5083532 0.1368861
    ##  [304,] 0.2514535 0.5131265 0.1368861
    ##  [305,] 0.2500000 0.5107399 0.1358412
    ##  [306,] 0.2507267 0.5107399 0.1368861
    ##  [307,] 0.2521802 0.5155131 0.1368861
    ##  [308,] 0.2514535 0.5107399 0.1379310
    ##  [309,] 0.2507267 0.5131265 0.1358412
    ##  [310,] 0.2492733 0.5107399 0.1347962
    ##  [311,] 0.2507267 0.5083532 0.1379310
    ##  [312,] 0.2521802 0.5155131 0.1368861
    ##  [313,] 0.2492733 0.5107399 0.1347962
    ##  [314,] 0.2514535 0.5131265 0.1368861
    ##  [315,] 0.2500000 0.5107399 0.1358412
    ##  [316,] 0.2463663 0.5083532 0.1316614
    ##  [317,] 0.2470930 0.5059666 0.1337513
    ##  [318,] 0.2470930 0.5107399 0.1316614
    ##  [319,] 0.2463663 0.5035800 0.1337513
    ##  [320,] 0.2485465 0.5083532 0.1347962
    ##  [321,] 0.2463663 0.5059666 0.1327064
    ##  [322,] 0.2441860 0.5011933 0.1316614
    ##  [323,] 0.2463663 0.5083532 0.1316614
    ##  [324,] 0.2478198 0.5107399 0.1327064
    ##  [325,] 0.2492733 0.5155131 0.1327064
    ##  [326,] 0.2470930 0.5131265 0.1306165
    ##  [327,] 0.2478198 0.5131265 0.1316614
    ##  [328,] 0.2470930 0.5107399 0.1316614
    ##  [329,] 0.2478198 0.5107399 0.1327064
    ##  [330,] 0.2500000 0.5178998 0.1327064
    ##  [331,] 0.2478198 0.5107399 0.1327064
    ##  [332,] 0.2470930 0.5107399 0.1316614
    ##  [333,] 0.2470930 0.5107399 0.1316614
    ##  [334,] 0.2456395 0.5083532 0.1306165
    ##  [335,] 0.2478198 0.5155131 0.1306165
    ##  [336,] 0.2470930 0.5107399 0.1316614
    ##  [337,] 0.2500000 0.5155131 0.1337513
    ##  [338,] 0.2485465 0.5131265 0.1327064
    ##  [339,] 0.2470930 0.5131265 0.1306165
    ##  [340,] 0.2478198 0.5083532 0.1337513
    ##  [341,] 0.2441860 0.5059666 0.1295716
    ##  [342,] 0.2478198 0.5107399 0.1327064
    ##  [343,] 0.2478198 0.5131265 0.1316614
    ##  [344,] 0.2463663 0.5131265 0.1295716
    ##  [345,] 0.2478198 0.5131265 0.1316614
    ##  [346,] 0.2470930 0.5155131 0.1295716
    ##  [347,] 0.2492733 0.5155131 0.1327064
    ##  [348,] 0.2485465 0.5155131 0.1316614
    ##  [349,] 0.2485465 0.5155131 0.1316614
    ##  [350,] 0.2485465 0.5155131 0.1316614
    ##  [351,] 0.2478198 0.5131265 0.1316614
    ##  [352,] 0.2470930 0.5131265 0.1306165
    ##  [353,] 0.2500000 0.5155131 0.1337513
    ##  [354,] 0.2507267 0.5178998 0.1337513
    ##  [355,] 0.2485465 0.5155131 0.1316614
    ##  [356,] 0.2492733 0.5178998 0.1316614
    ##  [357,] 0.2492733 0.5178998 0.1316614
    ##  [358,] 0.2507267 0.5155131 0.1347962
    ##  [359,] 0.2507267 0.5202864 0.1327064
    ##  [360,] 0.2500000 0.5202864 0.1316614
    ##  [361,] 0.2507267 0.5202864 0.1327064
    ##  [362,] 0.2507267 0.5202864 0.1327064
    ##  [363,] 0.2500000 0.5178998 0.1327064
    ##  [364,] 0.2485465 0.5155131 0.1316614
    ##  [365,] 0.2485465 0.5155131 0.1316614
    ##  [366,] 0.2492733 0.5178998 0.1316614
    ##  [367,] 0.2500000 0.5202864 0.1316614
    ##  [368,] 0.2492733 0.5178998 0.1316614
    ##  [369,] 0.2478198 0.5178998 0.1295716
    ##  [370,] 0.2478198 0.5178998 0.1295716
    ##  [371,] 0.2478198 0.5178998 0.1295716
    ##  [372,] 0.2492733 0.5202864 0.1306165
    ##  [373,] 0.2500000 0.5178998 0.1327064
    ##  [374,] 0.2485465 0.5155131 0.1316614
    ##  [375,] 0.2492733 0.5202864 0.1306165
    ##  [376,] 0.2485465 0.5155131 0.1316614
    ##  [377,] 0.2478198 0.5131265 0.1316614
    ##  [378,] 0.2492733 0.5202864 0.1306165
    ##  [379,] 0.2500000 0.5202864 0.1316614
    ##  [380,] 0.2507267 0.5226730 0.1316614
    ##  [381,] 0.2492733 0.5178998 0.1316614
    ##  [382,] 0.2507267 0.5226730 0.1316614
    ##  [383,] 0.2500000 0.5178998 0.1327064
    ##  [384,] 0.2500000 0.5202864 0.1316614
    ##  [385,] 0.2500000 0.5202864 0.1316614
    ##  [386,] 0.2500000 0.5202864 0.1316614
    ##  [387,] 0.2492733 0.5202864 0.1306165
    ##  [388,] 0.2492733 0.5202864 0.1306165
    ##  [389,] 0.2507267 0.5202864 0.1327064
    ##  [390,] 0.2492733 0.5155131 0.1327064
    ##  [391,] 0.2507267 0.5202864 0.1327064
    ##  [392,] 0.2492733 0.5178998 0.1316614
    ##  [393,] 0.2492733 0.5178998 0.1316614
    ##  [394,] 0.2492733 0.5178998 0.1316614
    ##  [395,] 0.2492733 0.5155131 0.1327064
    ##  [396,] 0.2485465 0.5155131 0.1316614
    ##  [397,] 0.2492733 0.5155131 0.1327064
    ##  [398,] 0.2478198 0.5107399 0.1327064
    ##  [399,] 0.2478198 0.5131265 0.1316614
    ##  [400,] 0.2463663 0.5107399 0.1306165
    ##  [401,] 0.2478198 0.5131265 0.1316614
    ##  [402,] 0.2485465 0.5131265 0.1327064
    ##  [403,] 0.2485465 0.5131265 0.1327064
    ##  [404,] 0.2485465 0.5155131 0.1316614
    ##  [405,] 0.2478198 0.5131265 0.1316614
    ##  [406,] 0.2478198 0.5131265 0.1316614
    ##  [407,] 0.2492733 0.5178998 0.1316614
    ##  [408,] 0.2500000 0.5178998 0.1327064
    ##  [409,] 0.2485465 0.5155131 0.1316614
    ##  [410,] 0.2485465 0.5155131 0.1316614
    ##  [411,] 0.2492733 0.5178998 0.1316614
    ##  [412,] 0.2485465 0.5178998 0.1306165
    ##  [413,] 0.2485465 0.5178998 0.1306165
    ##  [414,] 0.2456395 0.5131265 0.1285266
    ##  [415,] 0.2478198 0.5131265 0.1316614
    ##  [416,] 0.2470930 0.5131265 0.1306165
    ##  [417,] 0.2478198 0.5178998 0.1295716
    ##  [418,] 0.2470930 0.5155131 0.1295716
    ##  [419,] 0.2478198 0.5155131 0.1306165
    ##  [420,] 0.2478198 0.5178998 0.1295716
    ##  [421,] 0.2463663 0.5178998 0.1274817
    ##  [422,] 0.2470930 0.5178998 0.1285266
    ##  [423,] 0.2449128 0.5131265 0.1274817
    ##  [424,] 0.2470930 0.5155131 0.1295716
    ##  [425,] 0.2485465 0.5155131 0.1316614
    ##  [426,] 0.2470930 0.5155131 0.1295716
    ##  [427,] 0.2478198 0.5155131 0.1306165
    ##  [428,] 0.2470930 0.5178998 0.1285266
    ##  [429,] 0.2449128 0.5155131 0.1264368
    ##  [430,] 0.2470930 0.5178998 0.1285266
    ##  [431,] 0.2485465 0.5155131 0.1316614
    ##  [432,] 0.2492733 0.5178998 0.1316614
    ##  [433,] 0.2492733 0.5178998 0.1316614
    ##  [434,] 0.2470930 0.5131265 0.1306165
    ##  [435,] 0.2485465 0.5155131 0.1316614
    ##  [436,] 0.2478198 0.5155131 0.1306165
    ##  [437,] 0.2456395 0.5107399 0.1295716
    ##  [438,] 0.2463663 0.5131265 0.1295716
    ##  [439,] 0.2478198 0.5131265 0.1316614
    ##  [440,] 0.2478198 0.5131265 0.1316614
    ##  [441,] 0.2478198 0.5131265 0.1316614
    ##  [442,] 0.2478198 0.5131265 0.1316614
    ##  [443,] 0.2478198 0.5131265 0.1316614
    ##  [444,] 0.2470930 0.5131265 0.1306165
    ##  [445,] 0.2478198 0.5131265 0.1316614
    ##  [446,] 0.2485465 0.5155131 0.1316614
    ##  [447,] 0.2485465 0.5155131 0.1316614
    ##  [448,] 0.2470930 0.5155131 0.1295716
    ##  [449,] 0.2463663 0.5107399 0.1306165
    ##  [450,] 0.2492733 0.5155131 0.1327064
    ##  [451,] 0.2470930 0.5131265 0.1306165
    ##  [452,] 0.2478198 0.5131265 0.1316614
    ##  [453,] 0.2470930 0.5155131 0.1295716
    ##  [454,] 0.2478198 0.5131265 0.1316614
    ##  [455,] 0.2478198 0.5155131 0.1306165
    ##  [456,] 0.2485465 0.5155131 0.1316614
    ##  [457,] 0.2470930 0.5131265 0.1306165
    ##  [458,] 0.2470930 0.5131265 0.1306165
    ##  [459,] 0.2470930 0.5107399 0.1316614
    ##  [460,] 0.2470930 0.5107399 0.1316614
    ##  [461,] 0.2463663 0.5083532 0.1316614
    ##  [462,] 0.2463663 0.5107399 0.1306165
    ##  [463,] 0.2463663 0.5083532 0.1316614
    ##  [464,] 0.2485465 0.5131265 0.1327064
    ##  [465,] 0.2456395 0.5107399 0.1295716
    ##  [466,] 0.2470930 0.5107399 0.1316614
    ##  [467,] 0.2470930 0.5107399 0.1316614
    ##  [468,] 0.2463663 0.5107399 0.1306165
    ##  [469,] 0.2470930 0.5083532 0.1327064
    ##  [470,] 0.2478198 0.5131265 0.1316614
    ##  [471,] 0.2463663 0.5131265 0.1295716
    ##  [472,] 0.2470930 0.5131265 0.1306165
    ##  [473,] 0.2463663 0.5107399 0.1306165
    ##  [474,] 0.2470930 0.5131265 0.1306165
    ##  [475,] 0.2478198 0.5131265 0.1316614
    ##  [476,] 0.2463663 0.5131265 0.1295716
    ##  [477,] 0.2456395 0.5107399 0.1295716
    ##  [478,] 0.2470930 0.5107399 0.1316614
    ##  [479,] 0.2449128 0.5083532 0.1295716
    ##  [480,] 0.2463663 0.5107399 0.1306165
    ##  [481,] 0.2463663 0.5107399 0.1306165
    ##  [482,] 0.2456395 0.5107399 0.1295716
    ##  [483,] 0.2456395 0.5107399 0.1295716
    ##  [484,] 0.2463663 0.5131265 0.1295716
    ##  [485,] 0.2463663 0.5107399 0.1306165
    ##  [486,] 0.2463663 0.5131265 0.1295716
    ##  [487,] 0.2463663 0.5131265 0.1295716
    ##  [488,] 0.2470930 0.5155131 0.1295716
    ##  [489,] 0.2463663 0.5131265 0.1295716
    ##  [490,] 0.2463663 0.5131265 0.1295716
    ##  [491,] 0.2470930 0.5131265 0.1306165
    ##  [492,] 0.2463663 0.5131265 0.1295716
    ##  [493,] 0.2485465 0.5178998 0.1306165
    ##  [494,] 0.2478198 0.5155131 0.1306165
    ##  [495,] 0.2485465 0.5178998 0.1306165
    ##  [496,] 0.2492733 0.5202864 0.1306165
    ##  [497,] 0.2485465 0.5178998 0.1306165
    ##  [498,] 0.2470930 0.5155131 0.1295716
    ##  [499,] 0.2492733 0.5178998 0.1316614
    ##  [500,] 0.2478198 0.5178998 0.1295716
    ##  [501,] 0.2478198 0.5155131 0.1306165
    ##  [502,] 0.2478198 0.5155131 0.1306165
    ##  [503,] 0.2485465 0.5202864 0.1295716
    ##  [504,] 0.2485465 0.5178998 0.1306165
    ##  [505,] 0.2470930 0.5155131 0.1295716
    ##  [506,] 0.2492733 0.5202864 0.1306165
    ##  [507,] 0.2492733 0.5202864 0.1306165
    ##  [508,] 0.2485465 0.5202864 0.1295716
    ##  [509,] 0.2485465 0.5178998 0.1306165
    ##  [510,] 0.2492733 0.5202864 0.1306165
    ##  [511,] 0.2492733 0.5202864 0.1306165
    ##  [512,] 0.2485465 0.5178998 0.1306165
    ##  [513,] 0.2492733 0.5178998 0.1316614
    ##  [514,] 0.2500000 0.5202864 0.1316614
    ##  [515,] 0.2500000 0.5202864 0.1316614
    ##  [516,] 0.2500000 0.5202864 0.1316614
    ##  [517,] 0.2492733 0.5178998 0.1316614
    ##  [518,] 0.2485465 0.5155131 0.1316614
    ##  [519,] 0.2500000 0.5178998 0.1327064
    ##  [520,] 0.2492733 0.5178998 0.1316614
    ##  [521,] 0.2492733 0.5178998 0.1316614
    ##  [522,] 0.2500000 0.5178998 0.1327064
    ##  [523,] 0.2500000 0.5178998 0.1327064
    ##  [524,] 0.2485465 0.5178998 0.1306165
    ##  [525,] 0.2492733 0.5178998 0.1316614
    ##  [526,] 0.2470930 0.5131265 0.1306165
    ##  [527,] 0.2485465 0.5107399 0.1337513
    ##  [528,] 0.2478198 0.5107399 0.1327064
    ##  [529,] 0.2478198 0.5131265 0.1316614
    ##  [530,] 0.2478198 0.5131265 0.1316614
    ##  [531,] 0.2478198 0.5131265 0.1316614
    ##  [532,] 0.2478198 0.5131265 0.1316614
    ##  [533,] 0.2485465 0.5155131 0.1316614
    ##  [534,] 0.2485465 0.5155131 0.1316614
    ##  [535,] 0.2478198 0.5155131 0.1306165
    ##  [536,] 0.2478198 0.5155131 0.1306165
    ##  [537,] 0.2478198 0.5155131 0.1306165
    ##  [538,] 0.2478198 0.5155131 0.1306165
    ##  [539,] 0.2478198 0.5155131 0.1306165
    ##  [540,] 0.2492733 0.5178998 0.1316614
    ##  [541,] 0.2485465 0.5155131 0.1316614
    ##  [542,] 0.2478198 0.5131265 0.1316614
    ##  [543,] 0.2485465 0.5155131 0.1316614
    ##  [544,] 0.2478198 0.5155131 0.1306165
    ##  [545,] 0.2492733 0.5178998 0.1316614
    ##  [546,] 0.2500000 0.5155131 0.1337513
    ##  [547,] 0.2492733 0.5131265 0.1337513
    ##  [548,] 0.2478198 0.5131265 0.1316614
    ##  [549,] 0.2470930 0.5131265 0.1306165
    ##  [550,] 0.2500000 0.5178998 0.1327064
    ##  [551,] 0.2478198 0.5131265 0.1316614
    ##  [552,] 0.2478198 0.5131265 0.1316614
    ##  [553,] 0.2470930 0.5131265 0.1306165
    ##  [554,] 0.2463663 0.5131265 0.1295716
    ##  [555,] 0.2456395 0.5131265 0.1285266
    ##  [556,] 0.2470930 0.5155131 0.1295716
    ##  [557,] 0.2470930 0.5155131 0.1295716
    ##  [558,] 0.2470930 0.5131265 0.1306165
    ##  [559,] 0.2470930 0.5131265 0.1306165
    ##  [560,] 0.2456395 0.5107399 0.1295716
    ##  [561,] 0.2470930 0.5131265 0.1306165
    ##  [562,] 0.2478198 0.5155131 0.1306165
    ##  [563,] 0.2463663 0.5107399 0.1306165
    ##  [564,] 0.2456395 0.5107399 0.1295716
    ##  [565,] 0.2478198 0.5155131 0.1306165
    ##  [566,] 0.2485465 0.5178998 0.1306165
    ##  [567,] 0.2492733 0.5178998 0.1316614
    ##  [568,] 0.2492733 0.5202864 0.1306165
    ##  [569,] 0.2485465 0.5178998 0.1306165
    ##  [570,] 0.2485465 0.5155131 0.1316614
    ##  [571,] 0.2485465 0.5155131 0.1316614
    ##  [572,] 0.2478198 0.5155131 0.1306165
    ##  [573,] 0.2478198 0.5155131 0.1306165
    ##  [574,] 0.2470930 0.5155131 0.1295716
    ##  [575,] 0.2478198 0.5178998 0.1295716
    ##  [576,] 0.2485465 0.5155131 0.1316614
    ##  [577,] 0.2485465 0.5178998 0.1306165
    ##  [578,] 0.2492733 0.5178998 0.1316614
    ##  [579,] 0.2492733 0.5178998 0.1316614
    ##  [580,] 0.2492733 0.5178998 0.1316614
    ##  [581,] 0.2485465 0.5178998 0.1306165
    ##  [582,] 0.2492733 0.5155131 0.1327064
    ##  [583,] 0.2492733 0.5155131 0.1327064
    ##  [584,] 0.2500000 0.5178998 0.1327064
    ##  [585,] 0.2500000 0.5155131 0.1337513
    ##  [586,] 0.2500000 0.5178998 0.1327064
    ##  [587,] 0.2492733 0.5178998 0.1316614
    ##  [588,] 0.2500000 0.5178998 0.1327064
    ##  [589,] 0.2492733 0.5155131 0.1327064
    ##  [590,] 0.2500000 0.5202864 0.1316614
    ##  [591,] 0.2507267 0.5202864 0.1327064
    ##  [592,] 0.2492733 0.5178998 0.1316614
    ##  [593,] 0.2485465 0.5178998 0.1306165
    ##  [594,] 0.2478198 0.5155131 0.1306165
    ##  [595,] 0.2492733 0.5178998 0.1316614
    ##  [596,] 0.2492733 0.5178998 0.1316614
    ##  [597,] 0.2492733 0.5178998 0.1316614
    ##  [598,] 0.2500000 0.5202864 0.1316614
    ##  [599,] 0.2485465 0.5202864 0.1295716
    ##  [600,] 0.2485465 0.5178998 0.1306165
    ##  [601,] 0.2485465 0.5178998 0.1306165
    ##  [602,] 0.2507267 0.5226730 0.1316614
    ##  [603,] 0.2485465 0.5178998 0.1306165
    ##  [604,] 0.2500000 0.5202864 0.1316614
    ##  [605,] 0.2492733 0.5178998 0.1316614
    ##  [606,] 0.2470930 0.5155131 0.1295716
    ##  [607,] 0.2478198 0.5155131 0.1306165
    ##  [608,] 0.2500000 0.5202864 0.1316614
    ##  [609,] 0.2485465 0.5178998 0.1306165
    ##  [610,] 0.2485465 0.5178998 0.1306165
    ##  [611,] 0.2500000 0.5226730 0.1306165
    ##  [612,] 0.2478198 0.5155131 0.1306165
    ##  [613,] 0.2500000 0.5202864 0.1316614
    ##  [614,] 0.2492733 0.5178998 0.1316614
    ##  [615,] 0.2514535 0.5226730 0.1327064
    ##  [616,] 0.2514535 0.5226730 0.1327064
    ##  [617,] 0.2514535 0.5226730 0.1327064
    ##  [618,] 0.2500000 0.5202864 0.1316614
    ##  [619,] 0.2514535 0.5250597 0.1316614
    ##  [620,] 0.2507267 0.5202864 0.1327064
    ##  [621,] 0.2514535 0.5226730 0.1327064
    ##  [622,] 0.2500000 0.5202864 0.1316614
    ##  [623,] 0.2492733 0.5202864 0.1306165
    ##  [624,] 0.2500000 0.5202864 0.1316614
    ##  [625,] 0.2492733 0.5202864 0.1306165
    ##  [626,] 0.2492733 0.5202864 0.1306165
    ##  [627,] 0.2492733 0.5202864 0.1306165
    ##  [628,] 0.2492733 0.5202864 0.1306165
    ##  [629,] 0.2492733 0.5202864 0.1306165
    ##  [630,] 0.2492733 0.5202864 0.1306165
    ##  [631,] 0.2492733 0.5202864 0.1306165
    ##  [632,] 0.2492733 0.5202864 0.1306165
    ##  [633,] 0.2478198 0.5178998 0.1295716
    ##  [634,] 0.2463663 0.5155131 0.1285266
    ##  [635,] 0.2470930 0.5178998 0.1285266
    ##  [636,] 0.2478198 0.5155131 0.1306165
    ##  [637,] 0.2456395 0.5083532 0.1306165
    ##  [638,] 0.2470930 0.5155131 0.1295716
    ##  [639,] 0.2463663 0.5155131 0.1285266
    ##  [640,] 0.2456395 0.5131265 0.1285266
    ##  [641,] 0.2470930 0.5155131 0.1295716
    ##  [642,] 0.2456395 0.5107399 0.1295716
    ##  [643,] 0.2463663 0.5155131 0.1285266
    ##  [644,] 0.2470930 0.5131265 0.1306165
    ##  [645,] 0.2463663 0.5131265 0.1295716
    ##  [646,] 0.2456395 0.5131265 0.1285266
    ##  [647,] 0.2456395 0.5131265 0.1285266
    ##  [648,] 0.2463663 0.5131265 0.1295716
    ##  [649,] 0.2470930 0.5131265 0.1306165
    ##  [650,] 0.2463663 0.5131265 0.1295716
    ##  [651,] 0.2449128 0.5131265 0.1274817
    ##  [652,] 0.2463663 0.5155131 0.1285266
    ##  [653,] 0.2456395 0.5155131 0.1274817
    ##  [654,] 0.2449128 0.5155131 0.1264368
    ##  [655,] 0.2456395 0.5178998 0.1264368
    ##  [656,] 0.2449128 0.5155131 0.1264368
    ##  [657,] 0.2456395 0.5178998 0.1264368
    ##  [658,] 0.2470930 0.5202864 0.1274817
    ##  [659,] 0.2456395 0.5178998 0.1264368
    ##  [660,] 0.2449128 0.5155131 0.1264368
    ##  [661,] 0.2456395 0.5178998 0.1264368
    ##  [662,] 0.2463663 0.5178998 0.1274817
    ##  [663,] 0.2456395 0.5178998 0.1264368
    ##  [664,] 0.2456395 0.5178998 0.1264368
    ##  [665,] 0.2463663 0.5178998 0.1274817
    ##  [666,] 0.2456395 0.5155131 0.1274817
    ##  [667,] 0.2463663 0.5178998 0.1274817
    ##  [668,] 0.2470930 0.5178998 0.1285266
    ##  [669,] 0.2463663 0.5131265 0.1295716
    ##  [670,] 0.2463663 0.5155131 0.1285266
    ##  [671,] 0.2470930 0.5178998 0.1285266
    ##  [672,] 0.2478198 0.5178998 0.1295716
    ##  [673,] 0.2449128 0.5107399 0.1285266
    ##  [674,] 0.2463663 0.5155131 0.1285266
    ##  [675,] 0.2456395 0.5131265 0.1285266
    ##  [676,] 0.2456395 0.5131265 0.1285266
    ##  [677,] 0.2449128 0.5107399 0.1285266
    ##  [678,] 0.2441860 0.5083532 0.1285266
    ##  [679,] 0.2463663 0.5155131 0.1285266
    ##  [680,] 0.2456395 0.5107399 0.1295716
    ##  [681,] 0.2456395 0.5131265 0.1285266
    ##  [682,] 0.2463663 0.5155131 0.1285266
    ##  [683,] 0.2463663 0.5131265 0.1295716
    ##  [684,] 0.2470930 0.5155131 0.1295716
    ##  [685,] 0.2456395 0.5131265 0.1285266
    ##  [686,] 0.2470930 0.5155131 0.1295716
    ##  [687,] 0.2463663 0.5131265 0.1295716
    ##  [688,] 0.2456395 0.5131265 0.1285266
    ##  [689,] 0.2456395 0.5131265 0.1285266
    ##  [690,] 0.2456395 0.5131265 0.1285266
    ##  [691,] 0.2456395 0.5131265 0.1285266
    ##  [692,] 0.2470930 0.5155131 0.1295716
    ##  [693,] 0.2463663 0.5107399 0.1306165
    ##  [694,] 0.2456395 0.5131265 0.1285266
    ##  [695,] 0.2456395 0.5131265 0.1285266
    ##  [696,] 0.2456395 0.5107399 0.1295716
    ##  [697,] 0.2441860 0.5107399 0.1274817
    ##  [698,] 0.2449128 0.5131265 0.1274817
    ##  [699,] 0.2449128 0.5107399 0.1285266
    ##  [700,] 0.2449128 0.5107399 0.1285266
    ##  [701,] 0.2441860 0.5083532 0.1285266
    ##  [702,] 0.2441860 0.5107399 0.1274817
    ##  [703,] 0.2456395 0.5107399 0.1295716
    ##  [704,] 0.2456395 0.5131265 0.1285266
    ##  [705,] 0.2463663 0.5131265 0.1295716
    ##  [706,] 0.2456395 0.5107399 0.1295716
    ##  [707,] 0.2470930 0.5131265 0.1306165
    ##  [708,] 0.2463663 0.5131265 0.1295716
    ##  [709,] 0.2478198 0.5178998 0.1295716
    ##  [710,] 0.2470930 0.5155131 0.1295716
    ##  [711,] 0.2470930 0.5155131 0.1295716
    ##  [712,] 0.2485465 0.5155131 0.1316614
    ##  [713,] 0.2492733 0.5178998 0.1316614
    ##  [714,] 0.2492733 0.5178998 0.1316614
    ##  [715,] 0.2478198 0.5131265 0.1316614
    ##  [716,] 0.2470930 0.5107399 0.1316614
    ##  [717,] 0.2478198 0.5131265 0.1316614
    ##  [718,] 0.2492733 0.5178998 0.1316614
    ##  [719,] 0.2485465 0.5178998 0.1306165
    ##  [720,] 0.2492733 0.5178998 0.1316614
    ##  [721,] 0.2492733 0.5155131 0.1327064
    ##  [722,] 0.2478198 0.5155131 0.1306165
    ##  [723,] 0.2456395 0.5083532 0.1306165
    ##  [724,] 0.2463663 0.5107399 0.1306165
    ##  [725,] 0.2463663 0.5107399 0.1306165
    ##  [726,] 0.2463663 0.5107399 0.1306165
    ##  [727,] 0.2463663 0.5131265 0.1295716
    ##  [728,] 0.2470930 0.5131265 0.1306165
    ##  [729,] 0.2470930 0.5131265 0.1306165
    ##  [730,] 0.2463663 0.5107399 0.1306165
    ##  [731,] 0.2463663 0.5131265 0.1295716
    ##  [732,] 0.2470930 0.5131265 0.1306165
    ##  [733,] 0.2441860 0.5107399 0.1274817
    ##  [734,] 0.2456395 0.5107399 0.1295716
    ##  [735,] 0.2456395 0.5107399 0.1295716
    ##  [736,] 0.2456395 0.5107399 0.1295716
    ##  [737,] 0.2463663 0.5083532 0.1316614
    ##  [738,] 0.2470930 0.5107399 0.1316614
    ##  [739,] 0.2470930 0.5107399 0.1316614
    ##  [740,] 0.2463663 0.5107399 0.1306165
    ##  [741,] 0.2470930 0.5131265 0.1306165
    ##  [742,] 0.2463663 0.5107399 0.1306165
    ##  [743,] 0.2478198 0.5155131 0.1306165
    ##  [744,] 0.2485465 0.5155131 0.1316614
    ##  [745,] 0.2470930 0.5155131 0.1295716
    ##  [746,] 0.2478198 0.5155131 0.1306165
    ##  [747,] 0.2478198 0.5155131 0.1306165
    ##  [748,] 0.2470930 0.5155131 0.1295716
    ##  [749,] 0.2478198 0.5155131 0.1306165
    ##  [750,] 0.2485465 0.5178998 0.1306165
    ##  [751,] 0.2470930 0.5131265 0.1306165
    ##  [752,] 0.2463663 0.5131265 0.1295716
    ##  [753,] 0.2478198 0.5155131 0.1306165
    ##  [754,] 0.2478198 0.5155131 0.1306165
    ##  [755,] 0.2470930 0.5131265 0.1306165
    ##  [756,] 0.2470930 0.5131265 0.1306165
    ##  [757,] 0.2492733 0.5202864 0.1306165
    ##  [758,] 0.2478198 0.5155131 0.1306165
    ##  [759,] 0.2485465 0.5202864 0.1295716
    ##  [760,] 0.2485465 0.5178998 0.1306165
    ##  [761,] 0.2463663 0.5131265 0.1295716
    ##  [762,] 0.2485465 0.5155131 0.1316614
    ##  [763,] 0.2478198 0.5131265 0.1316614
    ##  [764,] 0.2478198 0.5131265 0.1316614
    ##  [765,] 0.2463663 0.5107399 0.1306165
    ##  [766,] 0.2463663 0.5107399 0.1306165
    ##  [767,] 0.2463663 0.5107399 0.1306165
    ##  [768,] 0.2485465 0.5131265 0.1327064
    ##  [769,] 0.2485465 0.5131265 0.1327064
    ##  [770,] 0.2470930 0.5107399 0.1316614
    ##  [771,] 0.2485465 0.5155131 0.1316614
    ##  [772,] 0.2478198 0.5131265 0.1316614
    ##  [773,] 0.2456395 0.5131265 0.1285266
    ##  [774,] 0.2463663 0.5131265 0.1295716
    ##  [775,] 0.2463663 0.5131265 0.1295716
    ##  [776,] 0.2470930 0.5131265 0.1306165
    ##  [777,] 0.2478198 0.5155131 0.1306165
    ##  [778,] 0.2470930 0.5131265 0.1306165
    ##  [779,] 0.2478198 0.5155131 0.1306165
    ##  [780,] 0.2492733 0.5178998 0.1316614
    ##  [781,] 0.2478198 0.5202864 0.1285266
    ##  [782,] 0.2463663 0.5155131 0.1285266
    ##  [783,] 0.2470930 0.5155131 0.1295716
    ##  [784,] 0.2470930 0.5155131 0.1295716
    ##  [785,] 0.2478198 0.5155131 0.1306165
    ##  [786,] 0.2478198 0.5155131 0.1306165
    ##  [787,] 0.2470930 0.5155131 0.1295716
    ##  [788,] 0.2463663 0.5131265 0.1295716
    ##  [789,] 0.2456395 0.5131265 0.1285266
    ##  [790,] 0.2456395 0.5131265 0.1285266
    ##  [791,] 0.2463663 0.5131265 0.1295716
    ##  [792,] 0.2463663 0.5155131 0.1285266
    ##  [793,] 0.2470930 0.5178998 0.1285266
    ##  [794,] 0.2463663 0.5155131 0.1285266
    ##  [795,] 0.2470930 0.5178998 0.1285266
    ##  [796,] 0.2470930 0.5178998 0.1285266
    ##  [797,] 0.2478198 0.5178998 0.1295716
    ##  [798,] 0.2470930 0.5155131 0.1295716
    ##  [799,] 0.2470930 0.5155131 0.1295716
    ##  [800,] 0.2470930 0.5131265 0.1306165
    ##  [801,] 0.2463663 0.5131265 0.1295716
    ##  [802,] 0.2463663 0.5107399 0.1306165
    ##  [803,] 0.2478198 0.5155131 0.1306165
    ##  [804,] 0.2463663 0.5107399 0.1306165
    ##  [805,] 0.2470930 0.5131265 0.1306165
    ##  [806,] 0.2463663 0.5107399 0.1306165
    ##  [807,] 0.2470930 0.5131265 0.1306165
    ##  [808,] 0.2456395 0.5083532 0.1306165
    ##  [809,] 0.2463663 0.5131265 0.1295716
    ##  [810,] 0.2463663 0.5107399 0.1306165
    ##  [811,] 0.2456395 0.5107399 0.1295716
    ##  [812,] 0.2456395 0.5107399 0.1295716
    ##  [813,] 0.2463663 0.5107399 0.1306165
    ##  [814,] 0.2456395 0.5083532 0.1306165
    ##  [815,] 0.2456395 0.5083532 0.1306165
    ##  [816,] 0.2456395 0.5083532 0.1306165
    ##  [817,] 0.2449128 0.5059666 0.1306165
    ##  [818,] 0.2441860 0.5035800 0.1306165
    ##  [819,] 0.2449128 0.5059666 0.1306165
    ##  [820,] 0.2470930 0.5131265 0.1306165
    ##  [821,] 0.2456395 0.5083532 0.1306165
    ##  [822,] 0.2478198 0.5155131 0.1306165
    ##  [823,] 0.2470930 0.5131265 0.1306165
    ##  [824,] 0.2478198 0.5155131 0.1306165
    ##  [825,] 0.2470930 0.5131265 0.1306165
    ##  [826,] 0.2485465 0.5178998 0.1306165
    ##  [827,] 0.2507267 0.5202864 0.1327064
    ##  [828,] 0.2492733 0.5178998 0.1316614
    ##  [829,] 0.2492733 0.5178998 0.1316614
    ##  [830,] 0.2478198 0.5131265 0.1316614
    ##  [831,] 0.2470930 0.5131265 0.1306165
    ##  [832,] 0.2485465 0.5155131 0.1316614
    ##  [833,] 0.2500000 0.5178998 0.1327064
    ##  [834,] 0.2500000 0.5178998 0.1327064
    ##  [835,] 0.2492733 0.5155131 0.1327064
    ##  [836,] 0.2485465 0.5131265 0.1327064
    ##  [837,] 0.2485465 0.5155131 0.1316614
    ##  [838,] 0.2485465 0.5155131 0.1316614
    ##  [839,] 0.2500000 0.5178998 0.1327064
    ##  [840,] 0.2500000 0.5178998 0.1327064
    ##  [841,] 0.2485465 0.5155131 0.1316614
    ##  [842,] 0.2485465 0.5155131 0.1316614
    ##  [843,] 0.2492733 0.5178998 0.1316614
    ##  [844,] 0.2478198 0.5155131 0.1306165
    ##  [845,] 0.2485465 0.5131265 0.1327064
    ##  [846,] 0.2492733 0.5178998 0.1316614
    ##  [847,] 0.2492733 0.5155131 0.1327064
    ##  [848,] 0.2485465 0.5155131 0.1316614
    ##  [849,] 0.2500000 0.5202864 0.1316614
    ##  [850,] 0.2500000 0.5178998 0.1327064
    ##  [851,] 0.2500000 0.5178998 0.1327064
    ##  [852,] 0.2492733 0.5178998 0.1316614
    ##  [853,] 0.2492733 0.5178998 0.1316614
    ##  [854,] 0.2500000 0.5178998 0.1327064
    ##  [855,] 0.2485465 0.5155131 0.1316614
    ##  [856,] 0.2485465 0.5155131 0.1316614
    ##  [857,] 0.2485465 0.5155131 0.1316614
    ##  [858,] 0.2485465 0.5178998 0.1306165
    ##  [859,] 0.2478198 0.5178998 0.1295716
    ##  [860,] 0.2470930 0.5178998 0.1285266
    ##  [861,] 0.2470930 0.5178998 0.1285266
    ##  [862,] 0.2478198 0.5178998 0.1295716
    ##  [863,] 0.2470930 0.5178998 0.1285266
    ##  [864,] 0.2463663 0.5155131 0.1285266
    ##  [865,] 0.2463663 0.5155131 0.1285266
    ##  [866,] 0.2463663 0.5155131 0.1285266
    ##  [867,] 0.2456395 0.5131265 0.1285266
    ##  [868,] 0.2470930 0.5155131 0.1295716
    ##  [869,] 0.2463663 0.5131265 0.1295716
    ##  [870,] 0.2478198 0.5178998 0.1295716
    ##  [871,] 0.2470930 0.5155131 0.1295716
    ##  [872,] 0.2478198 0.5178998 0.1295716
    ##  [873,] 0.2470930 0.5155131 0.1295716
    ##  [874,] 0.2470930 0.5155131 0.1295716
    ##  [875,] 0.2470930 0.5155131 0.1295716
    ##  [876,] 0.2470930 0.5155131 0.1295716
    ##  [877,] 0.2463663 0.5131265 0.1295716
    ##  [878,] 0.2463663 0.5131265 0.1295716
    ##  [879,] 0.2478198 0.5178998 0.1295716
    ##  [880,] 0.2456395 0.5107399 0.1295716
    ##  [881,] 0.2463663 0.5131265 0.1295716
    ##  [882,] 0.2463663 0.5131265 0.1295716
    ##  [883,] 0.2470930 0.5155131 0.1295716
    ##  [884,] 0.2463663 0.5131265 0.1295716
    ##  [885,] 0.2463663 0.5107399 0.1306165
    ##  [886,] 0.2456395 0.5107399 0.1295716
    ##  [887,] 0.2463663 0.5107399 0.1306165
    ##  [888,] 0.2470930 0.5155131 0.1295716
    ##  [889,] 0.2463663 0.5131265 0.1295716
    ##  [890,] 0.2463663 0.5155131 0.1285266
    ##  [891,] 0.2463663 0.5155131 0.1285266
    ##  [892,] 0.2470930 0.5178998 0.1285266
    ##  [893,] 0.2463663 0.5155131 0.1285266
    ##  [894,] 0.2470930 0.5155131 0.1295716
    ##  [895,] 0.2463663 0.5131265 0.1295716
    ##  [896,] 0.2463663 0.5131265 0.1295716
    ##  [897,] 0.2463663 0.5131265 0.1295716
    ##  [898,] 0.2470930 0.5131265 0.1306165
    ##  [899,] 0.2478198 0.5131265 0.1316614
    ##  [900,] 0.2470930 0.5131265 0.1306165
    ##  [901,] 0.2463663 0.5107399 0.1306165
    ##  [902,] 0.2463663 0.5131265 0.1295716
    ##  [903,] 0.2470930 0.5107399 0.1316614
    ##  [904,] 0.2470930 0.5155131 0.1295716
    ##  [905,] 0.2485465 0.5155131 0.1316614
    ##  [906,] 0.2485465 0.5155131 0.1316614
    ##  [907,] 0.2478198 0.5131265 0.1316614
    ##  [908,] 0.2478198 0.5155131 0.1306165
    ##  [909,] 0.2470930 0.5131265 0.1306165
    ##  [910,] 0.2470930 0.5131265 0.1306165
    ##  [911,] 0.2470930 0.5131265 0.1306165
    ##  [912,] 0.2441860 0.5107399 0.1274817
    ##  [913,] 0.2449128 0.5107399 0.1285266
    ##  [914,] 0.2456395 0.5131265 0.1285266
    ##  [915,] 0.2456395 0.5131265 0.1285266
    ##  [916,] 0.2463663 0.5155131 0.1285266
    ##  [917,] 0.2478198 0.5155131 0.1306165
    ##  [918,] 0.2485465 0.5178998 0.1306165
    ##  [919,] 0.2463663 0.5107399 0.1306165
    ##  [920,] 0.2478198 0.5131265 0.1316614
    ##  [921,] 0.2478198 0.5131265 0.1316614
    ##  [922,] 0.2463663 0.5107399 0.1306165
    ##  [923,] 0.2449128 0.5059666 0.1306165
    ##  [924,] 0.2463663 0.5107399 0.1306165
    ##  [925,] 0.2441860 0.5059666 0.1295716
    ##  [926,] 0.2441860 0.5059666 0.1295716
    ##  [927,] 0.2449128 0.5083532 0.1295716
    ##  [928,] 0.2456395 0.5083532 0.1306165
    ##  [929,] 0.2456395 0.5083532 0.1306165
    ##  [930,] 0.2456395 0.5107399 0.1295716
    ##  [931,] 0.2456395 0.5107399 0.1295716
    ##  [932,] 0.2449128 0.5083532 0.1295716
    ##  [933,] 0.2449128 0.5083532 0.1295716
    ##  [934,] 0.2463663 0.5131265 0.1295716
    ##  [935,] 0.2449128 0.5083532 0.1295716
    ##  [936,] 0.2449128 0.5083532 0.1295716
    ##  [937,] 0.2456395 0.5107399 0.1295716
    ##  [938,] 0.2463663 0.5107399 0.1306165
    ##  [939,] 0.2441860 0.5059666 0.1295716
    ##  [940,] 0.2456395 0.5083532 0.1306165
    ##  [941,] 0.2449128 0.5059666 0.1306165
    ##  [942,] 0.2441860 0.5059666 0.1295716
    ##  [943,] 0.2456395 0.5083532 0.1306165
    ##  [944,] 0.2441860 0.5059666 0.1295716
    ##  [945,] 0.2463663 0.5107399 0.1306165
    ##  [946,] 0.2463663 0.5107399 0.1306165
    ##  [947,] 0.2463663 0.5107399 0.1306165
    ##  [948,] 0.2449128 0.5083532 0.1295716
    ##  [949,] 0.2449128 0.5083532 0.1295716
    ##  [950,] 0.2456395 0.5083532 0.1306165
    ##  [951,] 0.2449128 0.5059666 0.1306165
    ##  [952,] 0.2441860 0.5059666 0.1295716
    ##  [953,] 0.2441860 0.5059666 0.1295716
    ##  [954,] 0.2456395 0.5083532 0.1306165
    ##  [955,] 0.2441860 0.5083532 0.1285266
    ##  [956,] 0.2441860 0.5083532 0.1285266
    ##  [957,] 0.2463663 0.5107399 0.1306165
    ##  [958,] 0.2449128 0.5083532 0.1295716
    ##  [959,] 0.2449128 0.5083532 0.1295716
    ##  [960,] 0.2449128 0.5083532 0.1295716
    ##  [961,] 0.2449128 0.5083532 0.1295716
    ##  [962,] 0.2463663 0.5107399 0.1306165
    ##  [963,] 0.2463663 0.5107399 0.1306165
    ##  [964,] 0.2463663 0.5107399 0.1306165
    ##  [965,] 0.2456395 0.5107399 0.1295716
    ##  [966,] 0.2456395 0.5107399 0.1295716
    ##  [967,] 0.2456395 0.5107399 0.1295716
    ##  [968,] 0.2449128 0.5083532 0.1295716
    ##  [969,] 0.2441860 0.5059666 0.1295716
    ##  [970,] 0.2434593 0.5059666 0.1285266
    ##  [971,] 0.2449128 0.5083532 0.1295716
    ##  [972,] 0.2441860 0.5059666 0.1295716
    ##  [973,] 0.2449128 0.5059666 0.1306165
    ##  [974,] 0.2427326 0.5035800 0.1285266
    ##  [975,] 0.2449128 0.5059666 0.1306165
    ##  [976,] 0.2449128 0.5083532 0.1295716
    ##  [977,] 0.2412791 0.5011933 0.1274817
    ##  [978,] 0.2441860 0.5083532 0.1285266
    ##  [979,] 0.2441860 0.5059666 0.1295716
    ##  [980,] 0.2449128 0.5083532 0.1295716
    ##  [981,] 0.2434593 0.5035800 0.1295716
    ##  [982,] 0.2441860 0.5059666 0.1295716
    ##  [983,] 0.2449128 0.5083532 0.1295716
    ##  [984,] 0.2434593 0.5035800 0.1295716
    ##  [985,] 0.2441860 0.5059666 0.1295716
    ##  [986,] 0.2434593 0.5059666 0.1285266
    ##  [987,] 0.2456395 0.5107399 0.1295716
    ##  [988,] 0.2434593 0.5059666 0.1285266
    ##  [989,] 0.2449128 0.5083532 0.1295716
    ##  [990,] 0.2441860 0.5059666 0.1295716
    ##  [991,] 0.2441860 0.5059666 0.1295716
    ##  [992,] 0.2456395 0.5107399 0.1295716
    ##  [993,] 0.2441860 0.5059666 0.1295716
    ##  [994,] 0.2449128 0.5083532 0.1295716
    ##  [995,] 0.2441860 0.5083532 0.1285266
    ##  [996,] 0.2441860 0.5083532 0.1285266
    ##  [997,] 0.2441860 0.5083532 0.1285266
    ##  [998,] 0.2456395 0.5107399 0.1295716
    ##  [999,] 0.2456395 0.5107399 0.1295716
    ## [1000,] 0.2449128 0.5107399 0.1285266

The Random Forest model for predicting loan acceptance has an OOB error
rate of about 24.5%, meaning it correctly predicts whether a customer
will accept a loan around three-quarters of the time. Looking at the
confusion matrix, the model does a good job identifying people who
accept offers, but it sometimes mistakes people who would reject the
offer as likely to accept. This shows the model is fairly reliable for
spotting likely acceptances, which can help guide loan approvals, but
some adjustments might be needed to avoid giving offers to customers who
would actually say no.

``` r
#Default - random forest
set.seed(123)
rf_default <- randomForest(
  default ~ age + annual_income + credit_score + num_credit_lines +
    delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months +
    income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies +
    delinquencies_to_credit_lines,
  data = train_clean_all_accepted,
  ntree = 1000,
  mtry = floor(sqrt(ncol(train_clean_all_accepted))),
  importance = TRUE,
  keep.inbag = TRUE
)

rf_default
```

    ## 
    ## Call:
    ##  randomForest(formula = default ~ age + annual_income + credit_score +      num_credit_lines + delinquencies_2yrs + credit_utilization +      loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines +      inquiries_to_delinquencies + delinquencies_to_credit_lines,      data = train_clean_all_accepted, ntree = 1000, mtry = floor(sqrt(ncol(train_clean_all_accepted))),      importance = TRUE, keep.inbag = TRUE) 
    ##                Type of random forest: classification
    ##                      Number of trees: 1000
    ## No. of variables tried at each split: 4
    ## 
    ##         OOB estimate of  error rate: 25.08%
    ## Confusion matrix:
    ##     0  1 class.error
    ## 0 707 26  0.03547067
    ## 1 214 10  0.95535714

``` r
rf_default$confusion
```

    ##     0  1 class.error
    ## 0 707 26  0.03547067
    ## 1 214 10  0.95535714

``` r
rf_default$err.rate
```

    ##               OOB          0         1
    ##    [1,] 0.3874644 0.26394052 0.7926829
    ##    [2,] 0.3722127 0.24000000 0.8195489
    ##    [3,] 0.3636364 0.22893773 0.7988166
    ##    [4,] 0.3353960 0.20993590 0.7608696
    ##    [5,] 0.3244496 0.18674699 0.7839196
    ##    [6,] 0.3215078 0.18965517 0.7669903
    ##    [7,] 0.3214286 0.17489422 0.8046512
    ##    [8,] 0.3180363 0.16828929 0.8119266
    ##    [9,] 0.3138918 0.15193370 0.8493151
    ##   [10,] 0.3136220 0.15405777 0.8409091
    ##   [11,] 0.3052632 0.14010989 0.8468468
    ##   [12,] 0.3053515 0.13698630 0.8565022
    ##   [13,] 0.2910995 0.12431694 0.8385650
    ##   [14,] 0.3141361 0.14617486 0.8654709
    ##   [15,] 0.3057592 0.13797814 0.8565022
    ##   [16,] 0.3106695 0.13642565 0.8834081
    ##   [17,] 0.3054393 0.12824011 0.8878924
    ##   [18,] 0.3103448 0.12960437 0.9017857
    ##   [19,] 0.3019854 0.12687585 0.8750000
    ##   [20,] 0.3051202 0.12278308 0.9017857
    ##   [21,] 0.2998955 0.11732606 0.8973214
    ##   [22,] 0.2988506 0.11459754 0.9017857
    ##   [23,] 0.2978056 0.11186903 0.9062500
    ##   [24,] 0.2946708 0.11323329 0.8883929
    ##   [25,] 0.2957158 0.11186903 0.8973214
    ##   [26,] 0.2884013 0.10641201 0.8839286
    ##   [27,] 0.2925810 0.10914052 0.8928571
    ##   [28,] 0.2852665 0.09959072 0.8928571
    ##   [29,] 0.2821317 0.09686221 0.8883929
    ##   [30,] 0.2831766 0.09549795 0.8973214
    ##   [31,] 0.2842215 0.09959072 0.8883929
    ##   [32,] 0.2821317 0.09822647 0.8839286
    ##   [33,] 0.2842215 0.09959072 0.8883929
    ##   [34,] 0.2821317 0.09276944 0.9017857
    ##   [35,] 0.2800418 0.09140518 0.8973214
    ##   [36,] 0.2800418 0.09276944 0.8928571
    ##   [37,] 0.2737722 0.08321965 0.8973214
    ##   [38,] 0.2758621 0.08594816 0.8973214
    ##   [39,] 0.2727273 0.08049113 0.9017857
    ##   [40,] 0.2748171 0.08594816 0.8928571
    ##   [41,] 0.2748171 0.08321965 0.9017857
    ##   [42,] 0.2727273 0.08185539 0.8973214
    ##   [43,] 0.2779519 0.08458390 0.9107143
    ##   [44,] 0.2758621 0.08321965 0.9062500
    ##   [45,] 0.2716823 0.08049113 0.8973214
    ##   [46,] 0.2727273 0.07639836 0.9151786
    ##   [47,] 0.2695925 0.07366985 0.9107143
    ##   [48,] 0.2695925 0.07230559 0.9151786
    ##   [49,] 0.2748171 0.07639836 0.9241071
    ##   [50,] 0.2706374 0.07366985 0.9151786
    ##   [51,] 0.2695925 0.07230559 0.9151786
    ##   [52,] 0.2695925 0.07094134 0.9196429
    ##   [53,] 0.2737722 0.06957708 0.9419643
    ##   [54,] 0.2622780 0.06002729 0.9241071
    ##   [55,] 0.2633229 0.05729877 0.9375000
    ##   [56,] 0.2622780 0.05866303 0.9285714
    ##   [57,] 0.2633229 0.06139154 0.9241071
    ##   [58,] 0.2591432 0.05866303 0.9151786
    ##   [59,] 0.2580982 0.06002729 0.9062500
    ##   [60,] 0.2549634 0.05457026 0.9107143
    ##   [61,] 0.2539185 0.05184175 0.9151786
    ##   [62,] 0.2622780 0.05866303 0.9285714
    ##   [63,] 0.2570533 0.05320600 0.9241071
    ##   [64,] 0.2622780 0.05729877 0.9330357
    ##   [65,] 0.2539185 0.05047749 0.9196429
    ##   [66,] 0.2518286 0.04638472 0.9241071
    ##   [67,] 0.2549634 0.04774898 0.9330357
    ##   [68,] 0.2570533 0.05047749 0.9330357
    ##   [69,] 0.2539185 0.04911323 0.9241071
    ##   [70,] 0.2528736 0.04638472 0.9285714
    ##   [71,] 0.2518286 0.04774898 0.9196429
    ##   [72,] 0.2486938 0.04502046 0.9151786
    ##   [73,] 0.2539185 0.05184175 0.9151786
    ##   [74,] 0.2528736 0.05047749 0.9151786
    ##   [75,] 0.2560084 0.05457026 0.9151786
    ##   [76,] 0.2497388 0.04911323 0.9062500
    ##   [77,] 0.2507837 0.04774898 0.9151786
    ##   [78,] 0.2528736 0.04911323 0.9196429
    ##   [79,] 0.2497388 0.04365621 0.9241071
    ##   [80,] 0.2476489 0.04229195 0.9196429
    ##   [81,] 0.2507837 0.04638472 0.9196429
    ##   [82,] 0.2445141 0.03956344 0.9151786
    ##   [83,] 0.2497388 0.04092769 0.9330357
    ##   [84,] 0.2507837 0.04638472 0.9196429
    ##   [85,] 0.2466040 0.04229195 0.9151786
    ##   [86,] 0.2486938 0.04502046 0.9151786
    ##   [87,] 0.2518286 0.04911323 0.9151786
    ##   [88,] 0.2497388 0.04638472 0.9151786
    ##   [89,] 0.2507837 0.04911323 0.9107143
    ##   [90,] 0.2507837 0.04774898 0.9151786
    ##   [91,] 0.2507837 0.04638472 0.9196429
    ##   [92,] 0.2486938 0.04229195 0.9241071
    ##   [93,] 0.2497388 0.04229195 0.9285714
    ##   [94,] 0.2539185 0.04774898 0.9285714
    ##   [95,] 0.2518286 0.04638472 0.9241071
    ##   [96,] 0.2528736 0.04502046 0.9330357
    ##   [97,] 0.2560084 0.05047749 0.9285714
    ##   [98,] 0.2539185 0.04911323 0.9241071
    ##   [99,] 0.2539185 0.04911323 0.9241071
    ##  [100,] 0.2539185 0.04774898 0.9285714
    ##  [101,] 0.2518286 0.04365621 0.9330357
    ##  [102,] 0.2518286 0.04774898 0.9196429
    ##  [103,] 0.2486938 0.04502046 0.9151786
    ##  [104,] 0.2528736 0.04774898 0.9241071
    ##  [105,] 0.2518286 0.04502046 0.9285714
    ##  [106,] 0.2497388 0.04365621 0.9241071
    ##  [107,] 0.2549634 0.04774898 0.9330357
    ##  [108,] 0.2518286 0.04638472 0.9241071
    ##  [109,] 0.2507837 0.04365621 0.9285714
    ##  [110,] 0.2497388 0.04502046 0.9196429
    ##  [111,] 0.2518286 0.04774898 0.9196429
    ##  [112,] 0.2528736 0.04774898 0.9241071
    ##  [113,] 0.2507837 0.04502046 0.9241071
    ##  [114,] 0.2539185 0.04638472 0.9330357
    ##  [115,] 0.2497388 0.04229195 0.9285714
    ##  [116,] 0.2497388 0.04365621 0.9241071
    ##  [117,] 0.2507837 0.04365621 0.9285714
    ##  [118,] 0.2486938 0.04092769 0.9285714
    ##  [119,] 0.2507837 0.04365621 0.9285714
    ##  [120,] 0.2455590 0.03956344 0.9196429
    ##  [121,] 0.2466040 0.03956344 0.9241071
    ##  [122,] 0.2486938 0.04092769 0.9285714
    ##  [123,] 0.2466040 0.04092769 0.9196429
    ##  [124,] 0.2486938 0.04365621 0.9196429
    ##  [125,] 0.2497388 0.04365621 0.9241071
    ##  [126,] 0.2528736 0.04502046 0.9330357
    ##  [127,] 0.2486938 0.04229195 0.9241071
    ##  [128,] 0.2486938 0.04092769 0.9285714
    ##  [129,] 0.2497388 0.04365621 0.9241071
    ##  [130,] 0.2497388 0.04502046 0.9196429
    ##  [131,] 0.2507837 0.04229195 0.9330357
    ##  [132,] 0.2486938 0.03819918 0.9375000
    ##  [133,] 0.2497388 0.04229195 0.9285714
    ##  [134,] 0.2497388 0.03956344 0.9375000
    ##  [135,] 0.2507837 0.04092769 0.9375000
    ##  [136,] 0.2507837 0.04092769 0.9375000
    ##  [137,] 0.2518286 0.03956344 0.9464286
    ##  [138,] 0.2507837 0.03956344 0.9419643
    ##  [139,] 0.2539185 0.04092769 0.9508929
    ##  [140,] 0.2539185 0.04092769 0.9508929
    ##  [141,] 0.2507837 0.03956344 0.9419643
    ##  [142,] 0.2518286 0.03819918 0.9508929
    ##  [143,] 0.2528736 0.04092769 0.9464286
    ##  [144,] 0.2539185 0.04092769 0.9508929
    ##  [145,] 0.2528736 0.03956344 0.9508929
    ##  [146,] 0.2539185 0.03819918 0.9598214
    ##  [147,] 0.2518286 0.03956344 0.9464286
    ##  [148,] 0.2486938 0.03683492 0.9419643
    ##  [149,] 0.2497388 0.03956344 0.9375000
    ##  [150,] 0.2476489 0.03819918 0.9330357
    ##  [151,] 0.2476489 0.03956344 0.9285714
    ##  [152,] 0.2497388 0.03956344 0.9375000
    ##  [153,] 0.2486938 0.03683492 0.9419643
    ##  [154,] 0.2455590 0.03547067 0.9330357
    ##  [155,] 0.2518286 0.04092769 0.9419643
    ##  [156,] 0.2497388 0.03956344 0.9375000
    ##  [157,] 0.2486938 0.03819918 0.9375000
    ##  [158,] 0.2497388 0.03956344 0.9375000
    ##  [159,] 0.2507837 0.03819918 0.9464286
    ##  [160,] 0.2507837 0.03956344 0.9419643
    ##  [161,] 0.2507837 0.03819918 0.9464286
    ##  [162,] 0.2497388 0.03819918 0.9419643
    ##  [163,] 0.2486938 0.03547067 0.9464286
    ##  [164,] 0.2476489 0.03410641 0.9464286
    ##  [165,] 0.2476489 0.03547067 0.9419643
    ##  [166,] 0.2497388 0.03683492 0.9464286
    ##  [167,] 0.2518286 0.03956344 0.9464286
    ##  [168,] 0.2507837 0.03819918 0.9464286
    ##  [169,] 0.2507837 0.03956344 0.9419643
    ##  [170,] 0.2518286 0.04092769 0.9419643
    ##  [171,] 0.2486938 0.03683492 0.9419643
    ##  [172,] 0.2497388 0.03819918 0.9419643
    ##  [173,] 0.2507837 0.03956344 0.9419643
    ##  [174,] 0.2507837 0.03956344 0.9419643
    ##  [175,] 0.2507837 0.03956344 0.9419643
    ##  [176,] 0.2507837 0.04092769 0.9375000
    ##  [177,] 0.2476489 0.03683492 0.9375000
    ##  [178,] 0.2486938 0.04092769 0.9285714
    ##  [179,] 0.2476489 0.03956344 0.9285714
    ##  [180,] 0.2476489 0.03819918 0.9330357
    ##  [181,] 0.2466040 0.03819918 0.9285714
    ##  [182,] 0.2466040 0.03683492 0.9330357
    ##  [183,] 0.2476489 0.04092769 0.9241071
    ##  [184,] 0.2486938 0.04092769 0.9285714
    ##  [185,] 0.2455590 0.03547067 0.9330357
    ##  [186,] 0.2486938 0.03819918 0.9375000
    ##  [187,] 0.2476489 0.03683492 0.9375000
    ##  [188,] 0.2466040 0.03683492 0.9330357
    ##  [189,] 0.2476489 0.03956344 0.9285714
    ##  [190,] 0.2507837 0.04092769 0.9375000
    ##  [191,] 0.2497388 0.03819918 0.9419643
    ##  [192,] 0.2497388 0.03819918 0.9419643
    ##  [193,] 0.2486938 0.03683492 0.9419643
    ##  [194,] 0.2486938 0.03819918 0.9375000
    ##  [195,] 0.2497388 0.03683492 0.9464286
    ##  [196,] 0.2497388 0.03683492 0.9464286
    ##  [197,] 0.2486938 0.03410641 0.9508929
    ##  [198,] 0.2486938 0.03410641 0.9508929
    ##  [199,] 0.2486938 0.03410641 0.9508929
    ##  [200,] 0.2486938 0.03547067 0.9464286
    ##  [201,] 0.2486938 0.03547067 0.9464286
    ##  [202,] 0.2486938 0.03547067 0.9464286
    ##  [203,] 0.2497388 0.03683492 0.9464286
    ##  [204,] 0.2497388 0.03819918 0.9419643
    ##  [205,] 0.2497388 0.03819918 0.9419643
    ##  [206,] 0.2507837 0.03819918 0.9464286
    ##  [207,] 0.2497388 0.03819918 0.9419643
    ##  [208,] 0.2507837 0.03819918 0.9464286
    ##  [209,] 0.2518286 0.03956344 0.9464286
    ##  [210,] 0.2507837 0.03819918 0.9464286
    ##  [211,] 0.2497388 0.03819918 0.9419643
    ##  [212,] 0.2486938 0.03683492 0.9419643
    ##  [213,] 0.2507837 0.03683492 0.9508929
    ##  [214,] 0.2507837 0.03683492 0.9508929
    ##  [215,] 0.2518286 0.03819918 0.9508929
    ##  [216,] 0.2528736 0.03956344 0.9508929
    ##  [217,] 0.2528736 0.03956344 0.9508929
    ##  [218,] 0.2518286 0.03819918 0.9508929
    ##  [219,] 0.2497388 0.03547067 0.9508929
    ##  [220,] 0.2507837 0.03683492 0.9508929
    ##  [221,] 0.2507837 0.03683492 0.9508929
    ##  [222,] 0.2507837 0.03819918 0.9464286
    ##  [223,] 0.2518286 0.03819918 0.9508929
    ##  [224,] 0.2507837 0.03819918 0.9464286
    ##  [225,] 0.2497388 0.03683492 0.9464286
    ##  [226,] 0.2486938 0.03547067 0.9464286
    ##  [227,] 0.2486938 0.03410641 0.9508929
    ##  [228,] 0.2486938 0.03547067 0.9464286
    ##  [229,] 0.2476489 0.03410641 0.9464286
    ##  [230,] 0.2476489 0.03410641 0.9464286
    ##  [231,] 0.2507837 0.03683492 0.9508929
    ##  [232,] 0.2486938 0.03410641 0.9508929
    ##  [233,] 0.2466040 0.03274216 0.9464286
    ##  [234,] 0.2497388 0.03547067 0.9508929
    ##  [235,] 0.2486938 0.03410641 0.9508929
    ##  [236,] 0.2486938 0.03410641 0.9508929
    ##  [237,] 0.2486938 0.03547067 0.9464286
    ##  [238,] 0.2507837 0.03683492 0.9508929
    ##  [239,] 0.2507837 0.03683492 0.9508929
    ##  [240,] 0.2497388 0.03683492 0.9464286
    ##  [241,] 0.2507837 0.03683492 0.9508929
    ##  [242,] 0.2497388 0.03683492 0.9464286
    ##  [243,] 0.2497388 0.03819918 0.9419643
    ##  [244,] 0.2497388 0.03683492 0.9464286
    ##  [245,] 0.2507837 0.03819918 0.9464286
    ##  [246,] 0.2518286 0.03956344 0.9464286
    ##  [247,] 0.2497388 0.03683492 0.9464286
    ##  [248,] 0.2507837 0.03819918 0.9464286
    ##  [249,] 0.2497388 0.03819918 0.9419643
    ##  [250,] 0.2497388 0.03819918 0.9419643
    ##  [251,] 0.2486938 0.03683492 0.9419643
    ##  [252,] 0.2486938 0.03683492 0.9419643
    ##  [253,] 0.2497388 0.03683492 0.9464286
    ##  [254,] 0.2518286 0.03956344 0.9464286
    ##  [255,] 0.2507837 0.03819918 0.9464286
    ##  [256,] 0.2518286 0.03819918 0.9508929
    ##  [257,] 0.2497388 0.03683492 0.9464286
    ##  [258,] 0.2507837 0.03956344 0.9419643
    ##  [259,] 0.2497388 0.03819918 0.9419643
    ##  [260,] 0.2507837 0.03956344 0.9419643
    ##  [261,] 0.2518286 0.03819918 0.9508929
    ##  [262,] 0.2497388 0.03547067 0.9508929
    ##  [263,] 0.2507837 0.03819918 0.9464286
    ##  [264,] 0.2497388 0.03547067 0.9508929
    ##  [265,] 0.2507837 0.03683492 0.9508929
    ##  [266,] 0.2507837 0.03956344 0.9419643
    ##  [267,] 0.2539185 0.04092769 0.9508929
    ##  [268,] 0.2528736 0.03956344 0.9508929
    ##  [269,] 0.2528736 0.03956344 0.9508929
    ##  [270,] 0.2549634 0.04229195 0.9508929
    ##  [271,] 0.2549634 0.04229195 0.9508929
    ##  [272,] 0.2549634 0.04092769 0.9553571
    ##  [273,] 0.2549634 0.03956344 0.9598214
    ##  [274,] 0.2570533 0.04229195 0.9598214
    ##  [275,] 0.2549634 0.03956344 0.9598214
    ##  [276,] 0.2528736 0.03819918 0.9553571
    ##  [277,] 0.2518286 0.03819918 0.9508929
    ##  [278,] 0.2518286 0.03956344 0.9464286
    ##  [279,] 0.2507837 0.03683492 0.9508929
    ##  [280,] 0.2518286 0.03683492 0.9553571
    ##  [281,] 0.2518286 0.03683492 0.9553571
    ##  [282,] 0.2518286 0.03683492 0.9553571
    ##  [283,] 0.2518286 0.03819918 0.9508929
    ##  [284,] 0.2518286 0.03819918 0.9508929
    ##  [285,] 0.2539185 0.03819918 0.9598214
    ##  [286,] 0.2518286 0.03683492 0.9553571
    ##  [287,] 0.2539185 0.03819918 0.9598214
    ##  [288,] 0.2539185 0.03819918 0.9598214
    ##  [289,] 0.2518286 0.03683492 0.9553571
    ##  [290,] 0.2518286 0.03683492 0.9553571
    ##  [291,] 0.2518286 0.03683492 0.9553571
    ##  [292,] 0.2528736 0.03683492 0.9598214
    ##  [293,] 0.2528736 0.03683492 0.9598214
    ##  [294,] 0.2528736 0.03683492 0.9598214
    ##  [295,] 0.2518286 0.03683492 0.9553571
    ##  [296,] 0.2528736 0.03683492 0.9598214
    ##  [297,] 0.2528736 0.03683492 0.9598214
    ##  [298,] 0.2528736 0.03683492 0.9598214
    ##  [299,] 0.2528736 0.03683492 0.9598214
    ##  [300,] 0.2528736 0.03683492 0.9598214
    ##  [301,] 0.2528736 0.03683492 0.9598214
    ##  [302,] 0.2518286 0.03683492 0.9553571
    ##  [303,] 0.2518286 0.03683492 0.9553571
    ##  [304,] 0.2539185 0.03819918 0.9598214
    ##  [305,] 0.2539185 0.03956344 0.9553571
    ##  [306,] 0.2518286 0.03683492 0.9553571
    ##  [307,] 0.2518286 0.03683492 0.9553571
    ##  [308,] 0.2518286 0.03683492 0.9553571
    ##  [309,] 0.2539185 0.03956344 0.9553571
    ##  [310,] 0.2528736 0.03956344 0.9508929
    ##  [311,] 0.2518286 0.03683492 0.9553571
    ##  [312,] 0.2539185 0.03819918 0.9598214
    ##  [313,] 0.2528736 0.03956344 0.9508929
    ##  [314,] 0.2549634 0.03956344 0.9598214
    ##  [315,] 0.2528736 0.03819918 0.9553571
    ##  [316,] 0.2528736 0.03683492 0.9598214
    ##  [317,] 0.2549634 0.03819918 0.9642857
    ##  [318,] 0.2528736 0.03819918 0.9553571
    ##  [319,] 0.2539185 0.03819918 0.9598214
    ##  [320,] 0.2518286 0.03683492 0.9553571
    ##  [321,] 0.2528736 0.03683492 0.9598214
    ##  [322,] 0.2518286 0.03683492 0.9553571
    ##  [323,] 0.2528736 0.03819918 0.9553571
    ##  [324,] 0.2507837 0.03547067 0.9553571
    ##  [325,] 0.2528736 0.03683492 0.9598214
    ##  [326,] 0.2528736 0.03683492 0.9598214
    ##  [327,] 0.2518286 0.03683492 0.9553571
    ##  [328,] 0.2528736 0.03819918 0.9553571
    ##  [329,] 0.2518286 0.03683492 0.9553571
    ##  [330,] 0.2528736 0.03819918 0.9553571
    ##  [331,] 0.2539185 0.03956344 0.9553571
    ##  [332,] 0.2528736 0.03819918 0.9553571
    ##  [333,] 0.2539185 0.03956344 0.9553571
    ##  [334,] 0.2528736 0.03956344 0.9508929
    ##  [335,] 0.2539185 0.03819918 0.9598214
    ##  [336,] 0.2539185 0.03819918 0.9598214
    ##  [337,] 0.2539185 0.03956344 0.9553571
    ##  [338,] 0.2539185 0.03956344 0.9553571
    ##  [339,] 0.2539185 0.03956344 0.9553571
    ##  [340,] 0.2549634 0.03956344 0.9598214
    ##  [341,] 0.2528736 0.03819918 0.9553571
    ##  [342,] 0.2539185 0.03956344 0.9553571
    ##  [343,] 0.2539185 0.04092769 0.9508929
    ##  [344,] 0.2539185 0.04092769 0.9508929
    ##  [345,] 0.2528736 0.03956344 0.9508929
    ##  [346,] 0.2539185 0.04092769 0.9508929
    ##  [347,] 0.2528736 0.03956344 0.9508929
    ##  [348,] 0.2528736 0.03956344 0.9508929
    ##  [349,] 0.2528736 0.03956344 0.9508929
    ##  [350,] 0.2528736 0.03956344 0.9508929
    ##  [351,] 0.2528736 0.03819918 0.9553571
    ##  [352,] 0.2539185 0.03956344 0.9553571
    ##  [353,] 0.2528736 0.03956344 0.9508929
    ##  [354,] 0.2539185 0.03956344 0.9553571
    ##  [355,] 0.2528736 0.03956344 0.9508929
    ##  [356,] 0.2528736 0.03956344 0.9508929
    ##  [357,] 0.2539185 0.03956344 0.9553571
    ##  [358,] 0.2528736 0.03819918 0.9553571
    ##  [359,] 0.2507837 0.03683492 0.9508929
    ##  [360,] 0.2518286 0.03819918 0.9508929
    ##  [361,] 0.2528736 0.03956344 0.9508929
    ##  [362,] 0.2528736 0.03956344 0.9508929
    ##  [363,] 0.2539185 0.04092769 0.9508929
    ##  [364,] 0.2528736 0.03956344 0.9508929
    ##  [365,] 0.2539185 0.04092769 0.9508929
    ##  [366,] 0.2560084 0.04229195 0.9553571
    ##  [367,] 0.2549634 0.04229195 0.9508929
    ##  [368,] 0.2539185 0.04092769 0.9508929
    ##  [369,] 0.2549634 0.04092769 0.9553571
    ##  [370,] 0.2549634 0.04092769 0.9553571
    ##  [371,] 0.2549634 0.04229195 0.9508929
    ##  [372,] 0.2549634 0.04092769 0.9553571
    ##  [373,] 0.2570533 0.04365621 0.9553571
    ##  [374,] 0.2560084 0.04365621 0.9508929
    ##  [375,] 0.2539185 0.04229195 0.9464286
    ##  [376,] 0.2528736 0.04092769 0.9464286
    ##  [377,] 0.2539185 0.04092769 0.9508929
    ##  [378,] 0.2528736 0.04092769 0.9464286
    ##  [379,] 0.2528736 0.03956344 0.9508929
    ##  [380,] 0.2528736 0.03956344 0.9508929
    ##  [381,] 0.2518286 0.03956344 0.9464286
    ##  [382,] 0.2539185 0.04229195 0.9464286
    ##  [383,] 0.2549634 0.04229195 0.9508929
    ##  [384,] 0.2539185 0.04092769 0.9508929
    ##  [385,] 0.2539185 0.04092769 0.9508929
    ##  [386,] 0.2528736 0.03956344 0.9508929
    ##  [387,] 0.2528736 0.03956344 0.9508929
    ##  [388,] 0.2528736 0.03956344 0.9508929
    ##  [389,] 0.2528736 0.03956344 0.9508929
    ##  [390,] 0.2549634 0.04092769 0.9553571
    ##  [391,] 0.2560084 0.04229195 0.9553571
    ##  [392,] 0.2560084 0.04229195 0.9553571
    ##  [393,] 0.2549634 0.04092769 0.9553571
    ##  [394,] 0.2539185 0.04092769 0.9508929
    ##  [395,] 0.2549634 0.04229195 0.9508929
    ##  [396,] 0.2549634 0.04229195 0.9508929
    ##  [397,] 0.2549634 0.04229195 0.9508929
    ##  [398,] 0.2549634 0.04229195 0.9508929
    ##  [399,] 0.2549634 0.04229195 0.9508929
    ##  [400,] 0.2549634 0.04092769 0.9553571
    ##  [401,] 0.2528736 0.03956344 0.9508929
    ##  [402,] 0.2539185 0.03956344 0.9553571
    ##  [403,] 0.2539185 0.03956344 0.9553571
    ##  [404,] 0.2539185 0.03956344 0.9553571
    ##  [405,] 0.2539185 0.03956344 0.9553571
    ##  [406,] 0.2539185 0.03956344 0.9553571
    ##  [407,] 0.2560084 0.04229195 0.9553571
    ##  [408,] 0.2539185 0.04092769 0.9508929
    ##  [409,] 0.2549634 0.04092769 0.9553571
    ##  [410,] 0.2549634 0.04092769 0.9553571
    ##  [411,] 0.2539185 0.03956344 0.9553571
    ##  [412,] 0.2539185 0.03956344 0.9553571
    ##  [413,] 0.2549634 0.04092769 0.9553571
    ##  [414,] 0.2539185 0.03956344 0.9553571
    ##  [415,] 0.2539185 0.03956344 0.9553571
    ##  [416,] 0.2539185 0.03956344 0.9553571
    ##  [417,] 0.2528736 0.03819918 0.9553571
    ##  [418,] 0.2539185 0.03956344 0.9553571
    ##  [419,] 0.2528736 0.03819918 0.9553571
    ##  [420,] 0.2528736 0.03819918 0.9553571
    ##  [421,] 0.2528736 0.03819918 0.9553571
    ##  [422,] 0.2528736 0.03819918 0.9553571
    ##  [423,] 0.2528736 0.03819918 0.9553571
    ##  [424,] 0.2539185 0.03819918 0.9598214
    ##  [425,] 0.2528736 0.03819918 0.9553571
    ##  [426,] 0.2528736 0.03819918 0.9553571
    ##  [427,] 0.2539185 0.03819918 0.9598214
    ##  [428,] 0.2528736 0.03819918 0.9553571
    ##  [429,] 0.2549634 0.03956344 0.9598214
    ##  [430,] 0.2539185 0.03819918 0.9598214
    ##  [431,] 0.2528736 0.03819918 0.9553571
    ##  [432,] 0.2528736 0.03819918 0.9553571
    ##  [433,] 0.2528736 0.03819918 0.9553571
    ##  [434,] 0.2528736 0.03819918 0.9553571
    ##  [435,] 0.2539185 0.03819918 0.9598214
    ##  [436,] 0.2539185 0.03819918 0.9598214
    ##  [437,] 0.2518286 0.03819918 0.9508929
    ##  [438,] 0.2549634 0.03956344 0.9598214
    ##  [439,] 0.2539185 0.03956344 0.9553571
    ##  [440,] 0.2549634 0.03956344 0.9598214
    ##  [441,] 0.2539185 0.03819918 0.9598214
    ##  [442,] 0.2528736 0.03819918 0.9553571
    ##  [443,] 0.2528736 0.03819918 0.9553571
    ##  [444,] 0.2507837 0.03683492 0.9508929
    ##  [445,] 0.2528736 0.03819918 0.9553571
    ##  [446,] 0.2518286 0.03683492 0.9553571
    ##  [447,] 0.2518286 0.03683492 0.9553571
    ##  [448,] 0.2518286 0.03683492 0.9553571
    ##  [449,] 0.2518286 0.03683492 0.9553571
    ##  [450,] 0.2518286 0.03819918 0.9508929
    ##  [451,] 0.2518286 0.03683492 0.9553571
    ##  [452,] 0.2528736 0.03819918 0.9553571
    ##  [453,] 0.2528736 0.03819918 0.9553571
    ##  [454,] 0.2528736 0.03819918 0.9553571
    ##  [455,] 0.2528736 0.03819918 0.9553571
    ##  [456,] 0.2528736 0.03819918 0.9553571
    ##  [457,] 0.2528736 0.03819918 0.9553571
    ##  [458,] 0.2528736 0.03819918 0.9553571
    ##  [459,] 0.2539185 0.03956344 0.9553571
    ##  [460,] 0.2528736 0.03819918 0.9553571
    ##  [461,] 0.2528736 0.03819918 0.9553571
    ##  [462,] 0.2528736 0.03819918 0.9553571
    ##  [463,] 0.2539185 0.03956344 0.9553571
    ##  [464,] 0.2539185 0.03956344 0.9553571
    ##  [465,] 0.2539185 0.03956344 0.9553571
    ##  [466,] 0.2549634 0.04092769 0.9553571
    ##  [467,] 0.2549634 0.04092769 0.9553571
    ##  [468,] 0.2528736 0.03956344 0.9508929
    ##  [469,] 0.2539185 0.03956344 0.9553571
    ##  [470,] 0.2528736 0.03956344 0.9508929
    ##  [471,] 0.2539185 0.03956344 0.9553571
    ##  [472,] 0.2539185 0.03956344 0.9553571
    ##  [473,] 0.2539185 0.03956344 0.9553571
    ##  [474,] 0.2539185 0.03956344 0.9553571
    ##  [475,] 0.2539185 0.03956344 0.9553571
    ##  [476,] 0.2539185 0.03956344 0.9553571
    ##  [477,] 0.2539185 0.03956344 0.9553571
    ##  [478,] 0.2539185 0.03956344 0.9553571
    ##  [479,] 0.2539185 0.03956344 0.9553571
    ##  [480,] 0.2539185 0.03956344 0.9553571
    ##  [481,] 0.2539185 0.03956344 0.9553571
    ##  [482,] 0.2528736 0.03956344 0.9508929
    ##  [483,] 0.2528736 0.03819918 0.9553571
    ##  [484,] 0.2539185 0.03956344 0.9553571
    ##  [485,] 0.2518286 0.03819918 0.9508929
    ##  [486,] 0.2518286 0.03956344 0.9464286
    ##  [487,] 0.2539185 0.03956344 0.9553571
    ##  [488,] 0.2518286 0.03819918 0.9508929
    ##  [489,] 0.2518286 0.03819918 0.9508929
    ##  [490,] 0.2539185 0.04092769 0.9508929
    ##  [491,] 0.2539185 0.04092769 0.9508929
    ##  [492,] 0.2549634 0.04092769 0.9553571
    ##  [493,] 0.2539185 0.03956344 0.9553571
    ##  [494,] 0.2518286 0.03819918 0.9508929
    ##  [495,] 0.2528736 0.03956344 0.9508929
    ##  [496,] 0.2528736 0.03819918 0.9553571
    ##  [497,] 0.2539185 0.03956344 0.9553571
    ##  [498,] 0.2528736 0.03819918 0.9553571
    ##  [499,] 0.2518286 0.03683492 0.9553571
    ##  [500,] 0.2518286 0.03683492 0.9553571
    ##  [501,] 0.2518286 0.03683492 0.9553571
    ##  [502,] 0.2518286 0.03683492 0.9553571
    ##  [503,] 0.2518286 0.03683492 0.9553571
    ##  [504,] 0.2528736 0.03819918 0.9553571
    ##  [505,] 0.2528736 0.03819918 0.9553571
    ##  [506,] 0.2518286 0.03683492 0.9553571
    ##  [507,] 0.2518286 0.03683492 0.9553571
    ##  [508,] 0.2528736 0.03819918 0.9553571
    ##  [509,] 0.2518286 0.03683492 0.9553571
    ##  [510,] 0.2518286 0.03683492 0.9553571
    ##  [511,] 0.2518286 0.03683492 0.9553571
    ##  [512,] 0.2528736 0.03819918 0.9553571
    ##  [513,] 0.2539185 0.03956344 0.9553571
    ##  [514,] 0.2528736 0.03819918 0.9553571
    ##  [515,] 0.2528736 0.03819918 0.9553571
    ##  [516,] 0.2539185 0.03956344 0.9553571
    ##  [517,] 0.2549634 0.04092769 0.9553571
    ##  [518,] 0.2539185 0.04092769 0.9508929
    ##  [519,] 0.2539185 0.04092769 0.9508929
    ##  [520,] 0.2539185 0.04092769 0.9508929
    ##  [521,] 0.2549634 0.04229195 0.9508929
    ##  [522,] 0.2539185 0.04092769 0.9508929
    ##  [523,] 0.2539185 0.04092769 0.9508929
    ##  [524,] 0.2539185 0.04092769 0.9508929
    ##  [525,] 0.2539185 0.04092769 0.9508929
    ##  [526,] 0.2560084 0.04229195 0.9553571
    ##  [527,] 0.2549634 0.04092769 0.9553571
    ##  [528,] 0.2549634 0.04092769 0.9553571
    ##  [529,] 0.2539185 0.04092769 0.9508929
    ##  [530,] 0.2528736 0.03956344 0.9508929
    ##  [531,] 0.2518286 0.03819918 0.9508929
    ##  [532,] 0.2528736 0.03956344 0.9508929
    ##  [533,] 0.2528736 0.03956344 0.9508929
    ##  [534,] 0.2528736 0.03956344 0.9508929
    ##  [535,] 0.2528736 0.03956344 0.9508929
    ##  [536,] 0.2528736 0.03956344 0.9508929
    ##  [537,] 0.2528736 0.03956344 0.9508929
    ##  [538,] 0.2539185 0.04092769 0.9508929
    ##  [539,] 0.2528736 0.03956344 0.9508929
    ##  [540,] 0.2549634 0.04229195 0.9508929
    ##  [541,] 0.2549634 0.04229195 0.9508929
    ##  [542,] 0.2528736 0.03956344 0.9508929
    ##  [543,] 0.2539185 0.04092769 0.9508929
    ##  [544,] 0.2539185 0.04092769 0.9508929
    ##  [545,] 0.2539185 0.04092769 0.9508929
    ##  [546,] 0.2539185 0.04092769 0.9508929
    ##  [547,] 0.2549634 0.04229195 0.9508929
    ##  [548,] 0.2539185 0.04092769 0.9508929
    ##  [549,] 0.2539185 0.04092769 0.9508929
    ##  [550,] 0.2549634 0.04092769 0.9553571
    ##  [551,] 0.2560084 0.04229195 0.9553571
    ##  [552,] 0.2539185 0.03956344 0.9553571
    ##  [553,] 0.2549634 0.04092769 0.9553571
    ##  [554,] 0.2560084 0.04229195 0.9553571
    ##  [555,] 0.2549634 0.04092769 0.9553571
    ##  [556,] 0.2560084 0.04365621 0.9508929
    ##  [557,] 0.2560084 0.04229195 0.9553571
    ##  [558,] 0.2539185 0.04092769 0.9508929
    ##  [559,] 0.2560084 0.04229195 0.9553571
    ##  [560,] 0.2560084 0.04229195 0.9553571
    ##  [561,] 0.2560084 0.04229195 0.9553571
    ##  [562,] 0.2549634 0.04092769 0.9553571
    ##  [563,] 0.2560084 0.04229195 0.9553571
    ##  [564,] 0.2528736 0.03956344 0.9508929
    ##  [565,] 0.2560084 0.04229195 0.9553571
    ##  [566,] 0.2539185 0.03956344 0.9553571
    ##  [567,] 0.2560084 0.04229195 0.9553571
    ##  [568,] 0.2539185 0.03956344 0.9553571
    ##  [569,] 0.2528736 0.03819918 0.9553571
    ##  [570,] 0.2528736 0.03819918 0.9553571
    ##  [571,] 0.2528736 0.03819918 0.9553571
    ##  [572,] 0.2549634 0.04092769 0.9553571
    ##  [573,] 0.2560084 0.04229195 0.9553571
    ##  [574,] 0.2528736 0.03819918 0.9553571
    ##  [575,] 0.2507837 0.03547067 0.9553571
    ##  [576,] 0.2507837 0.03683492 0.9508929
    ##  [577,] 0.2518286 0.03819918 0.9508929
    ##  [578,] 0.2518286 0.03819918 0.9508929
    ##  [579,] 0.2497388 0.03547067 0.9508929
    ##  [580,] 0.2518286 0.03819918 0.9508929
    ##  [581,] 0.2518286 0.03683492 0.9553571
    ##  [582,] 0.2518286 0.03683492 0.9553571
    ##  [583,] 0.2507837 0.03683492 0.9508929
    ##  [584,] 0.2528736 0.03819918 0.9553571
    ##  [585,] 0.2507837 0.03683492 0.9508929
    ##  [586,] 0.2497388 0.03547067 0.9508929
    ##  [587,] 0.2518286 0.03819918 0.9508929
    ##  [588,] 0.2507837 0.03683492 0.9508929
    ##  [589,] 0.2507837 0.03683492 0.9508929
    ##  [590,] 0.2507837 0.03683492 0.9508929
    ##  [591,] 0.2518286 0.03819918 0.9508929
    ##  [592,] 0.2518286 0.03819918 0.9508929
    ##  [593,] 0.2528736 0.03956344 0.9508929
    ##  [594,] 0.2518286 0.03819918 0.9508929
    ##  [595,] 0.2518286 0.03819918 0.9508929
    ##  [596,] 0.2507837 0.03683492 0.9508929
    ##  [597,] 0.2507837 0.03683492 0.9508929
    ##  [598,] 0.2507837 0.03683492 0.9508929
    ##  [599,] 0.2518286 0.03819918 0.9508929
    ##  [600,] 0.2507837 0.03683492 0.9508929
    ##  [601,] 0.2497388 0.03547067 0.9508929
    ##  [602,] 0.2507837 0.03683492 0.9508929
    ##  [603,] 0.2497388 0.03547067 0.9508929
    ##  [604,] 0.2497388 0.03547067 0.9508929
    ##  [605,] 0.2497388 0.03547067 0.9508929
    ##  [606,] 0.2497388 0.03547067 0.9508929
    ##  [607,] 0.2497388 0.03547067 0.9508929
    ##  [608,] 0.2497388 0.03547067 0.9508929
    ##  [609,] 0.2497388 0.03547067 0.9508929
    ##  [610,] 0.2507837 0.03683492 0.9508929
    ##  [611,] 0.2528736 0.03819918 0.9553571
    ##  [612,] 0.2528736 0.03819918 0.9553571
    ##  [613,] 0.2528736 0.03819918 0.9553571
    ##  [614,] 0.2528736 0.03819918 0.9553571
    ##  [615,] 0.2507837 0.03683492 0.9508929
    ##  [616,] 0.2518286 0.03683492 0.9553571
    ##  [617,] 0.2518286 0.03683492 0.9553571
    ##  [618,] 0.2518286 0.03683492 0.9553571
    ##  [619,] 0.2518286 0.03819918 0.9508929
    ##  [620,] 0.2539185 0.03956344 0.9553571
    ##  [621,] 0.2518286 0.03683492 0.9553571
    ##  [622,] 0.2539185 0.03956344 0.9553571
    ##  [623,] 0.2518286 0.03683492 0.9553571
    ##  [624,] 0.2507837 0.03547067 0.9553571
    ##  [625,] 0.2507837 0.03683492 0.9508929
    ##  [626,] 0.2507837 0.03683492 0.9508929
    ##  [627,] 0.2507837 0.03683492 0.9508929
    ##  [628,] 0.2507837 0.03683492 0.9508929
    ##  [629,] 0.2507837 0.03683492 0.9508929
    ##  [630,] 0.2507837 0.03683492 0.9508929
    ##  [631,] 0.2497388 0.03683492 0.9464286
    ##  [632,] 0.2497388 0.03683492 0.9464286
    ##  [633,] 0.2507837 0.03819918 0.9464286
    ##  [634,] 0.2518286 0.03819918 0.9508929
    ##  [635,] 0.2518286 0.03819918 0.9508929
    ##  [636,] 0.2518286 0.03819918 0.9508929
    ##  [637,] 0.2518286 0.03819918 0.9508929
    ##  [638,] 0.2518286 0.03819918 0.9508929
    ##  [639,] 0.2518286 0.03819918 0.9508929
    ##  [640,] 0.2507837 0.03683492 0.9508929
    ##  [641,] 0.2507837 0.03683492 0.9508929
    ##  [642,] 0.2507837 0.03683492 0.9508929
    ##  [643,] 0.2507837 0.03683492 0.9508929
    ##  [644,] 0.2507837 0.03683492 0.9508929
    ##  [645,] 0.2507837 0.03683492 0.9508929
    ##  [646,] 0.2518286 0.03683492 0.9553571
    ##  [647,] 0.2518286 0.03683492 0.9553571
    ##  [648,] 0.2518286 0.03683492 0.9553571
    ##  [649,] 0.2507837 0.03683492 0.9508929
    ##  [650,] 0.2518286 0.03819918 0.9508929
    ##  [651,] 0.2507837 0.03683492 0.9508929
    ##  [652,] 0.2518286 0.03819918 0.9508929
    ##  [653,] 0.2518286 0.03819918 0.9508929
    ##  [654,] 0.2518286 0.03819918 0.9508929
    ##  [655,] 0.2507837 0.03819918 0.9464286
    ##  [656,] 0.2518286 0.03819918 0.9508929
    ##  [657,] 0.2507837 0.03819918 0.9464286
    ##  [658,] 0.2518286 0.03819918 0.9508929
    ##  [659,] 0.2507837 0.03819918 0.9464286
    ##  [660,] 0.2497388 0.03683492 0.9464286
    ##  [661,] 0.2507837 0.03819918 0.9464286
    ##  [662,] 0.2486938 0.03547067 0.9464286
    ##  [663,] 0.2497388 0.03683492 0.9464286
    ##  [664,] 0.2497388 0.03547067 0.9508929
    ##  [665,] 0.2497388 0.03547067 0.9508929
    ##  [666,] 0.2518286 0.03683492 0.9553571
    ##  [667,] 0.2518286 0.03683492 0.9553571
    ##  [668,] 0.2497388 0.03547067 0.9508929
    ##  [669,] 0.2518286 0.03683492 0.9553571
    ##  [670,] 0.2497388 0.03547067 0.9508929
    ##  [671,] 0.2476489 0.03410641 0.9464286
    ##  [672,] 0.2486938 0.03410641 0.9508929
    ##  [673,] 0.2486938 0.03410641 0.9508929
    ##  [674,] 0.2486938 0.03547067 0.9464286
    ##  [675,] 0.2476489 0.03410641 0.9464286
    ##  [676,] 0.2486938 0.03410641 0.9508929
    ##  [677,] 0.2476489 0.03410641 0.9464286
    ##  [678,] 0.2476489 0.03410641 0.9464286
    ##  [679,] 0.2497388 0.03547067 0.9508929
    ##  [680,] 0.2497388 0.03547067 0.9508929
    ##  [681,] 0.2486938 0.03410641 0.9508929
    ##  [682,] 0.2486938 0.03547067 0.9464286
    ##  [683,] 0.2486938 0.03410641 0.9508929
    ##  [684,] 0.2486938 0.03547067 0.9464286
    ##  [685,] 0.2476489 0.03410641 0.9464286
    ##  [686,] 0.2497388 0.03683492 0.9464286
    ##  [687,] 0.2486938 0.03547067 0.9464286
    ##  [688,] 0.2497388 0.03683492 0.9464286
    ##  [689,] 0.2507837 0.03819918 0.9464286
    ##  [690,] 0.2497388 0.03683492 0.9464286
    ##  [691,] 0.2497388 0.03683492 0.9464286
    ##  [692,] 0.2497388 0.03683492 0.9464286
    ##  [693,] 0.2507837 0.03819918 0.9464286
    ##  [694,] 0.2507837 0.03819918 0.9464286
    ##  [695,] 0.2507837 0.03819918 0.9464286
    ##  [696,] 0.2497388 0.03683492 0.9464286
    ##  [697,] 0.2518286 0.03819918 0.9508929
    ##  [698,] 0.2518286 0.03819918 0.9508929
    ##  [699,] 0.2497388 0.03547067 0.9508929
    ##  [700,] 0.2507837 0.03683492 0.9508929
    ##  [701,] 0.2507837 0.03683492 0.9508929
    ##  [702,] 0.2518286 0.03819918 0.9508929
    ##  [703,] 0.2518286 0.03819918 0.9508929
    ##  [704,] 0.2518286 0.03819918 0.9508929
    ##  [705,] 0.2518286 0.03819918 0.9508929
    ##  [706,] 0.2518286 0.03819918 0.9508929
    ##  [707,] 0.2518286 0.03819918 0.9508929
    ##  [708,] 0.2507837 0.03683492 0.9508929
    ##  [709,] 0.2507837 0.03683492 0.9508929
    ##  [710,] 0.2518286 0.03819918 0.9508929
    ##  [711,] 0.2507837 0.03683492 0.9508929
    ##  [712,] 0.2528736 0.03956344 0.9508929
    ##  [713,] 0.2518286 0.03819918 0.9508929
    ##  [714,] 0.2507837 0.03819918 0.9464286
    ##  [715,] 0.2518286 0.03819918 0.9508929
    ##  [716,] 0.2497388 0.03683492 0.9464286
    ##  [717,] 0.2486938 0.03547067 0.9464286
    ##  [718,] 0.2507837 0.03819918 0.9464286
    ##  [719,] 0.2507837 0.03683492 0.9508929
    ##  [720,] 0.2497388 0.03547067 0.9508929
    ##  [721,] 0.2518286 0.03819918 0.9508929
    ##  [722,] 0.2497388 0.03547067 0.9508929
    ##  [723,] 0.2476489 0.03410641 0.9464286
    ##  [724,] 0.2486938 0.03547067 0.9464286
    ##  [725,] 0.2476489 0.03410641 0.9464286
    ##  [726,] 0.2486938 0.03547067 0.9464286
    ##  [727,] 0.2466040 0.03274216 0.9464286
    ##  [728,] 0.2507837 0.03683492 0.9508929
    ##  [729,] 0.2486938 0.03547067 0.9464286
    ##  [730,] 0.2486938 0.03547067 0.9464286
    ##  [731,] 0.2476489 0.03410641 0.9464286
    ##  [732,] 0.2486938 0.03683492 0.9419643
    ##  [733,] 0.2486938 0.03683492 0.9419643
    ##  [734,] 0.2497388 0.03683492 0.9464286
    ##  [735,] 0.2476489 0.03410641 0.9464286
    ##  [736,] 0.2476489 0.03410641 0.9464286
    ##  [737,] 0.2476489 0.03410641 0.9464286
    ##  [738,] 0.2486938 0.03547067 0.9464286
    ##  [739,] 0.2476489 0.03547067 0.9419643
    ##  [740,] 0.2486938 0.03547067 0.9464286
    ##  [741,] 0.2486938 0.03547067 0.9464286
    ##  [742,] 0.2486938 0.03547067 0.9464286
    ##  [743,] 0.2476489 0.03547067 0.9419643
    ##  [744,] 0.2497388 0.03547067 0.9508929
    ##  [745,] 0.2497388 0.03547067 0.9508929
    ##  [746,] 0.2486938 0.03547067 0.9464286
    ##  [747,] 0.2497388 0.03547067 0.9508929
    ##  [748,] 0.2497388 0.03547067 0.9508929
    ##  [749,] 0.2507837 0.03683492 0.9508929
    ##  [750,] 0.2486938 0.03410641 0.9508929
    ##  [751,] 0.2497388 0.03547067 0.9508929
    ##  [752,] 0.2497388 0.03547067 0.9508929
    ##  [753,] 0.2486938 0.03410641 0.9508929
    ##  [754,] 0.2497388 0.03547067 0.9508929
    ##  [755,] 0.2497388 0.03547067 0.9508929
    ##  [756,] 0.2486938 0.03410641 0.9508929
    ##  [757,] 0.2486938 0.03547067 0.9464286
    ##  [758,] 0.2486938 0.03547067 0.9464286
    ##  [759,] 0.2486938 0.03547067 0.9464286
    ##  [760,] 0.2486938 0.03547067 0.9464286
    ##  [761,] 0.2486938 0.03547067 0.9464286
    ##  [762,] 0.2476489 0.03410641 0.9464286
    ##  [763,] 0.2476489 0.03410641 0.9464286
    ##  [764,] 0.2486938 0.03547067 0.9464286
    ##  [765,] 0.2486938 0.03547067 0.9464286
    ##  [766,] 0.2486938 0.03547067 0.9464286
    ##  [767,] 0.2486938 0.03547067 0.9464286
    ##  [768,] 0.2497388 0.03547067 0.9508929
    ##  [769,] 0.2497388 0.03547067 0.9508929
    ##  [770,] 0.2497388 0.03547067 0.9508929
    ##  [771,] 0.2497388 0.03547067 0.9508929
    ##  [772,] 0.2497388 0.03547067 0.9508929
    ##  [773,] 0.2497388 0.03547067 0.9508929
    ##  [774,] 0.2497388 0.03547067 0.9508929
    ##  [775,] 0.2507837 0.03547067 0.9553571
    ##  [776,] 0.2497388 0.03547067 0.9508929
    ##  [777,] 0.2497388 0.03547067 0.9508929
    ##  [778,] 0.2497388 0.03547067 0.9508929
    ##  [779,] 0.2497388 0.03547067 0.9508929
    ##  [780,] 0.2518286 0.03819918 0.9508929
    ##  [781,] 0.2497388 0.03547067 0.9508929
    ##  [782,] 0.2497388 0.03547067 0.9508929
    ##  [783,] 0.2507837 0.03683492 0.9508929
    ##  [784,] 0.2507837 0.03683492 0.9508929
    ##  [785,] 0.2507837 0.03683492 0.9508929
    ##  [786,] 0.2507837 0.03683492 0.9508929
    ##  [787,] 0.2507837 0.03683492 0.9508929
    ##  [788,] 0.2507837 0.03683492 0.9508929
    ##  [789,] 0.2507837 0.03683492 0.9508929
    ##  [790,] 0.2507837 0.03683492 0.9508929
    ##  [791,] 0.2507837 0.03683492 0.9508929
    ##  [792,] 0.2507837 0.03683492 0.9508929
    ##  [793,] 0.2497388 0.03547067 0.9508929
    ##  [794,] 0.2497388 0.03547067 0.9508929
    ##  [795,] 0.2497388 0.03547067 0.9508929
    ##  [796,] 0.2497388 0.03547067 0.9508929
    ##  [797,] 0.2497388 0.03547067 0.9508929
    ##  [798,] 0.2507837 0.03683492 0.9508929
    ##  [799,] 0.2507837 0.03683492 0.9508929
    ##  [800,] 0.2497388 0.03547067 0.9508929
    ##  [801,] 0.2507837 0.03683492 0.9508929
    ##  [802,] 0.2507837 0.03683492 0.9508929
    ##  [803,] 0.2507837 0.03683492 0.9508929
    ##  [804,] 0.2497388 0.03547067 0.9508929
    ##  [805,] 0.2497388 0.03547067 0.9508929
    ##  [806,] 0.2507837 0.03683492 0.9508929
    ##  [807,] 0.2497388 0.03547067 0.9508929
    ##  [808,] 0.2507837 0.03683492 0.9508929
    ##  [809,] 0.2497388 0.03547067 0.9508929
    ##  [810,] 0.2507837 0.03683492 0.9508929
    ##  [811,] 0.2497388 0.03547067 0.9508929
    ##  [812,] 0.2507837 0.03683492 0.9508929
    ##  [813,] 0.2518286 0.03819918 0.9508929
    ##  [814,] 0.2518286 0.03819918 0.9508929
    ##  [815,] 0.2518286 0.03819918 0.9508929
    ##  [816,] 0.2518286 0.03819918 0.9508929
    ##  [817,] 0.2518286 0.03819918 0.9508929
    ##  [818,] 0.2507837 0.03683492 0.9508929
    ##  [819,] 0.2518286 0.03819918 0.9508929
    ##  [820,] 0.2507837 0.03683492 0.9508929
    ##  [821,] 0.2507837 0.03683492 0.9508929
    ##  [822,] 0.2497388 0.03547067 0.9508929
    ##  [823,] 0.2486938 0.03410641 0.9508929
    ##  [824,] 0.2486938 0.03410641 0.9508929
    ##  [825,] 0.2486938 0.03410641 0.9508929
    ##  [826,] 0.2486938 0.03410641 0.9508929
    ##  [827,] 0.2497388 0.03547067 0.9508929
    ##  [828,] 0.2486938 0.03410641 0.9508929
    ##  [829,] 0.2486938 0.03410641 0.9508929
    ##  [830,] 0.2507837 0.03683492 0.9508929
    ##  [831,] 0.2497388 0.03547067 0.9508929
    ##  [832,] 0.2497388 0.03547067 0.9508929
    ##  [833,] 0.2497388 0.03547067 0.9508929
    ##  [834,] 0.2497388 0.03547067 0.9508929
    ##  [835,] 0.2497388 0.03547067 0.9508929
    ##  [836,] 0.2497388 0.03547067 0.9508929
    ##  [837,] 0.2507837 0.03683492 0.9508929
    ##  [838,] 0.2497388 0.03547067 0.9508929
    ##  [839,] 0.2497388 0.03547067 0.9508929
    ##  [840,] 0.2497388 0.03547067 0.9508929
    ##  [841,] 0.2486938 0.03410641 0.9508929
    ##  [842,] 0.2486938 0.03410641 0.9508929
    ##  [843,] 0.2486938 0.03410641 0.9508929
    ##  [844,] 0.2486938 0.03410641 0.9508929
    ##  [845,] 0.2486938 0.03410641 0.9508929
    ##  [846,] 0.2486938 0.03410641 0.9508929
    ##  [847,] 0.2497388 0.03547067 0.9508929
    ##  [848,] 0.2497388 0.03547067 0.9508929
    ##  [849,] 0.2486938 0.03410641 0.9508929
    ##  [850,] 0.2486938 0.03410641 0.9508929
    ##  [851,] 0.2486938 0.03410641 0.9508929
    ##  [852,] 0.2486938 0.03410641 0.9508929
    ##  [853,] 0.2486938 0.03410641 0.9508929
    ##  [854,] 0.2486938 0.03410641 0.9508929
    ##  [855,] 0.2497388 0.03547067 0.9508929
    ##  [856,] 0.2486938 0.03410641 0.9508929
    ##  [857,] 0.2497388 0.03547067 0.9508929
    ##  [858,] 0.2486938 0.03410641 0.9508929
    ##  [859,] 0.2486938 0.03410641 0.9508929
    ##  [860,] 0.2486938 0.03410641 0.9508929
    ##  [861,] 0.2486938 0.03410641 0.9508929
    ##  [862,] 0.2486938 0.03410641 0.9508929
    ##  [863,] 0.2497388 0.03547067 0.9508929
    ##  [864,] 0.2486938 0.03410641 0.9508929
    ##  [865,] 0.2486938 0.03410641 0.9508929
    ##  [866,] 0.2486938 0.03410641 0.9508929
    ##  [867,] 0.2486938 0.03410641 0.9508929
    ##  [868,] 0.2497388 0.03547067 0.9508929
    ##  [869,] 0.2486938 0.03410641 0.9508929
    ##  [870,] 0.2486938 0.03410641 0.9508929
    ##  [871,] 0.2497388 0.03547067 0.9508929
    ##  [872,] 0.2497388 0.03547067 0.9508929
    ##  [873,] 0.2497388 0.03547067 0.9508929
    ##  [874,] 0.2497388 0.03547067 0.9508929
    ##  [875,] 0.2497388 0.03547067 0.9508929
    ##  [876,] 0.2497388 0.03547067 0.9508929
    ##  [877,] 0.2497388 0.03547067 0.9508929
    ##  [878,] 0.2486938 0.03410641 0.9508929
    ##  [879,] 0.2486938 0.03410641 0.9508929
    ##  [880,] 0.2497388 0.03547067 0.9508929
    ##  [881,] 0.2497388 0.03547067 0.9508929
    ##  [882,] 0.2497388 0.03547067 0.9508929
    ##  [883,] 0.2497388 0.03547067 0.9508929
    ##  [884,] 0.2497388 0.03547067 0.9508929
    ##  [885,] 0.2497388 0.03547067 0.9508929
    ##  [886,] 0.2497388 0.03547067 0.9508929
    ##  [887,] 0.2497388 0.03547067 0.9508929
    ##  [888,] 0.2497388 0.03547067 0.9508929
    ##  [889,] 0.2497388 0.03547067 0.9508929
    ##  [890,] 0.2497388 0.03547067 0.9508929
    ##  [891,] 0.2497388 0.03547067 0.9508929
    ##  [892,] 0.2497388 0.03547067 0.9508929
    ##  [893,] 0.2497388 0.03547067 0.9508929
    ##  [894,] 0.2497388 0.03547067 0.9508929
    ##  [895,] 0.2497388 0.03547067 0.9508929
    ##  [896,] 0.2497388 0.03547067 0.9508929
    ##  [897,] 0.2497388 0.03547067 0.9508929
    ##  [898,] 0.2486938 0.03410641 0.9508929
    ##  [899,] 0.2486938 0.03410641 0.9508929
    ##  [900,] 0.2486938 0.03410641 0.9508929
    ##  [901,] 0.2486938 0.03410641 0.9508929
    ##  [902,] 0.2486938 0.03410641 0.9508929
    ##  [903,] 0.2476489 0.03274216 0.9508929
    ##  [904,] 0.2486938 0.03410641 0.9508929
    ##  [905,] 0.2486938 0.03274216 0.9553571
    ##  [906,] 0.2486938 0.03274216 0.9553571
    ##  [907,] 0.2486938 0.03274216 0.9553571
    ##  [908,] 0.2486938 0.03274216 0.9553571
    ##  [909,] 0.2486938 0.03410641 0.9508929
    ##  [910,] 0.2476489 0.03274216 0.9508929
    ##  [911,] 0.2486938 0.03410641 0.9508929
    ##  [912,] 0.2476489 0.03274216 0.9508929
    ##  [913,] 0.2486938 0.03410641 0.9508929
    ##  [914,] 0.2476489 0.03137790 0.9553571
    ##  [915,] 0.2486938 0.03274216 0.9553571
    ##  [916,] 0.2486938 0.03274216 0.9553571
    ##  [917,] 0.2476489 0.03274216 0.9508929
    ##  [918,] 0.2476489 0.03137790 0.9553571
    ##  [919,] 0.2486938 0.03410641 0.9508929
    ##  [920,] 0.2476489 0.03274216 0.9508929
    ##  [921,] 0.2476489 0.03274216 0.9508929
    ##  [922,] 0.2476489 0.03274216 0.9508929
    ##  [923,] 0.2476489 0.03274216 0.9508929
    ##  [924,] 0.2476489 0.03274216 0.9508929
    ##  [925,] 0.2486938 0.03410641 0.9508929
    ##  [926,] 0.2476489 0.03274216 0.9508929
    ##  [927,] 0.2486938 0.03410641 0.9508929
    ##  [928,] 0.2476489 0.03274216 0.9508929
    ##  [929,] 0.2486938 0.03410641 0.9508929
    ##  [930,] 0.2486938 0.03410641 0.9508929
    ##  [931,] 0.2486938 0.03410641 0.9508929
    ##  [932,] 0.2486938 0.03410641 0.9508929
    ##  [933,] 0.2476489 0.03274216 0.9508929
    ##  [934,] 0.2476489 0.03274216 0.9508929
    ##  [935,] 0.2486938 0.03410641 0.9508929
    ##  [936,] 0.2476489 0.03274216 0.9508929
    ##  [937,] 0.2476489 0.03274216 0.9508929
    ##  [938,] 0.2486938 0.03410641 0.9508929
    ##  [939,] 0.2486938 0.03410641 0.9508929
    ##  [940,] 0.2476489 0.03274216 0.9508929
    ##  [941,] 0.2486938 0.03410641 0.9508929
    ##  [942,] 0.2476489 0.03274216 0.9508929
    ##  [943,] 0.2486938 0.03410641 0.9508929
    ##  [944,] 0.2476489 0.03274216 0.9508929
    ##  [945,] 0.2476489 0.03274216 0.9508929
    ##  [946,] 0.2476489 0.03274216 0.9508929
    ##  [947,] 0.2476489 0.03274216 0.9508929
    ##  [948,] 0.2476489 0.03274216 0.9508929
    ##  [949,] 0.2486938 0.03410641 0.9508929
    ##  [950,] 0.2476489 0.03274216 0.9508929
    ##  [951,] 0.2486938 0.03274216 0.9553571
    ##  [952,] 0.2486938 0.03410641 0.9508929
    ##  [953,] 0.2476489 0.03274216 0.9508929
    ##  [954,] 0.2476489 0.03274216 0.9508929
    ##  [955,] 0.2476489 0.03274216 0.9508929
    ##  [956,] 0.2486938 0.03410641 0.9508929
    ##  [957,] 0.2476489 0.03274216 0.9508929
    ##  [958,] 0.2486938 0.03410641 0.9508929
    ##  [959,] 0.2497388 0.03410641 0.9553571
    ##  [960,] 0.2486938 0.03274216 0.9553571
    ##  [961,] 0.2497388 0.03410641 0.9553571
    ##  [962,] 0.2497388 0.03410641 0.9553571
    ##  [963,] 0.2497388 0.03410641 0.9553571
    ##  [964,] 0.2486938 0.03274216 0.9553571
    ##  [965,] 0.2486938 0.03274216 0.9553571
    ##  [966,] 0.2497388 0.03410641 0.9553571
    ##  [967,] 0.2486938 0.03274216 0.9553571
    ##  [968,] 0.2486938 0.03274216 0.9553571
    ##  [969,] 0.2497388 0.03410641 0.9553571
    ##  [970,] 0.2497388 0.03410641 0.9553571
    ##  [971,] 0.2497388 0.03410641 0.9553571
    ##  [972,] 0.2497388 0.03410641 0.9553571
    ##  [973,] 0.2497388 0.03410641 0.9553571
    ##  [974,] 0.2497388 0.03410641 0.9553571
    ##  [975,] 0.2486938 0.03274216 0.9553571
    ##  [976,] 0.2486938 0.03274216 0.9553571
    ##  [977,] 0.2497388 0.03410641 0.9553571
    ##  [978,] 0.2497388 0.03410641 0.9553571
    ##  [979,] 0.2497388 0.03410641 0.9553571
    ##  [980,] 0.2497388 0.03410641 0.9553571
    ##  [981,] 0.2497388 0.03410641 0.9553571
    ##  [982,] 0.2497388 0.03410641 0.9553571
    ##  [983,] 0.2497388 0.03410641 0.9553571
    ##  [984,] 0.2497388 0.03410641 0.9553571
    ##  [985,] 0.2497388 0.03410641 0.9553571
    ##  [986,] 0.2497388 0.03410641 0.9553571
    ##  [987,] 0.2497388 0.03410641 0.9553571
    ##  [988,] 0.2497388 0.03410641 0.9553571
    ##  [989,] 0.2497388 0.03410641 0.9553571
    ##  [990,] 0.2497388 0.03410641 0.9553571
    ##  [991,] 0.2497388 0.03410641 0.9553571
    ##  [992,] 0.2497388 0.03410641 0.9553571
    ##  [993,] 0.2497388 0.03410641 0.9553571
    ##  [994,] 0.2497388 0.03410641 0.9553571
    ##  [995,] 0.2497388 0.03410641 0.9553571
    ##  [996,] 0.2497388 0.03410641 0.9553571
    ##  [997,] 0.2497388 0.03410641 0.9553571
    ##  [998,] 0.2497388 0.03410641 0.9553571
    ##  [999,] 0.2497388 0.03410641 0.9553571
    ## [1000,] 0.2507837 0.03547067 0.9553571

The Random Forest model for predicting loan default has an OOB error
rate of about 25%, meaning it correctly predicts defaults and
non-defaults around three-quarters of the time. The confusion matrix
shows the model is very good at identifying loans that won’t default,
but it struggles to correctly identify loans that will default—it
misclassifies most of them. This suggests the model is strong for
spotting safe loans, but less reliable for predicting risky ones, so
caution is needed when using it to manage credit risk.

``` r
imp_acc <- importance(rf_accept)

importance_df_acc <- data.frame(
  Variable = rownames(imp_acc),
  MeanDecreaseAccuracy = imp_acc[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini = imp_acc[, "MeanDecreaseGini"]
) %>% arrange(desc(MeanDecreaseAccuracy))

ggplot(importance_df_acc,
       aes(x = reorder(Variable, MeanDecreaseAccuracy),
           y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Variable Importance — Acceptance Model",
    x = "Variable",
    y = "Mean Decrease Accuracy"
  )
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-24-1.png)<!-- -->

To understand which factors most influence whether a customer accepts a
loan, we used the Random Forest model’s built-in variable importance
measures. Specifically, we extracted the Mean Decrease in Accuracy and
Mean Decrease in Gini for each predictor. The Mean Decrease Accuracy
shows how much the model’s prediction accuracy drops if a variable is
removed, so higher values indicate more important variables. We then
arranged the variables by their Mean Decrease Accuracy and visualized
them using a horizontal bar chart. This plot clearly highlights which
features, such as credit score, income-to-loan ratio, and annual income
are the strongest predictors of loan acceptance, providing actionable
insights for targeting potential borrowers.

``` r
imp_def <- importance(rf_default)

importance_df_def <- data.frame(
  Variable = rownames(imp_def),
  MeanDecreaseAccuracy = imp_def[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini = imp_def[, "MeanDecreaseGini"]
) %>% arrange(desc(MeanDecreaseAccuracy))

ggplot(importance_df_def,
       aes(x = reorder(Variable, MeanDecreaseAccuracy),
           y = MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "tomato") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Variable Importance — Default Model",
    x = "Variable",
    y = "Mean Decrease Accuracy"
  )
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

For the default prediction model, we similarly used the Random Forest’s
variable importance measures to identify the key drivers of loan
default. We extracted both the Mean Decrease in Accuracy and Mean
Decrease in Gini for each predictor, with higher values indicating
variables that are more critical for accurate predictions. By arranging
the predictors by Mean Decrease Accuracy and visualizing them in a bar
chart, we could clearly see which factors most influence default risk.
Key variables such as loan amount, income to loan ratio and annual
income emerged as the strongest predictors, highlighting the features
most relevant for assessing borrower risk.

``` r
mtry_values <- 1:10

mtry_results_accept <- data.frame(
  mtry = mtry_values,
  oob_error = numeric(length(mtry_values))
)

for (i in seq_along(mtry_values)) {
  set.seed(2024)
  rf_temp <- randomForest(
    offer_accepted ~ age + annual_income + credit_score + num_credit_lines +
      delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months +
      income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies +
      delinquencies_to_credit_lines,
    data = train_clean,
    mtry = mtry_values[i],
    ntree = 1000
  )
  
  mtry_results_accept$oob_error[i] <- tail(rf_temp$err.rate[,"OOB"], 1)
}

mtry_results_accept
```

    ##    mtry oob_error
    ## 1     1 0.2572674
    ## 2     2 0.2492733
    ## 3     3 0.2405523
    ## 4     4 0.2427326
    ## 5     5 0.2398256
    ## 6     6 0.2405523
    ## 7     7 0.2478198
    ## 8     8 0.2456395
    ## 9     9 0.2500000
    ## 10   10 0.2485465

``` r
mtry_values <- 1:10

mtry_results_default <- data.frame(
  mtry = mtry_values,
  oob_error = numeric(length(mtry_values))
)

for (i in seq_along(mtry_values)) {
  set.seed(2024)
  rf_temp <- randomForest(
    default ~ age + annual_income + credit_score + num_credit_lines +
      delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months +
      income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies +
      delinquencies_to_credit_lines,
    data = train_clean_all_accepted,
    mtry = mtry_values[i],
    ntree = 1000
  )
  
  mtry_results_default$oob_error[i] <- tail(rf_temp$err.rate[,"OOB"], 1)
}

mtry_results_default
```

    ##    mtry oob_error
    ## 1     1 0.2361546
    ## 2     2 0.2382445
    ## 3     3 0.2466040
    ## 4     4 0.2424242
    ## 5     5 0.2445141
    ## 6     6 0.2466040
    ## 7     7 0.2455590
    ## 8     8 0.2539185
    ## 9     9 0.2455590
    ## 10   10 0.2549634

To optimize the Random Forest model for predicting offer acceptance, we
tested different values of the mtry parameter, which controls the number
of predictor variables randomly considered at each split in a tree. We
looped through mtry values from 1 to 10, fitting a Random Forest model
for each and recording the OOB error.The resulting tables shows the OOB
error for each tested mtry for acceptance and default, allowing us to
select the best configuration for the final model.From the OOB error
results, we observed that the error decreased as mtry increased up to a
certain point, after which it either stabilized or slightly increased.
This indicates that using too few variables at each split reduces model
accuracy, while too many adds little benefit and may increase variance.
The mtry value with the lowest OOB error was selected for the final
Random Forest model, ensuring a good balance between bias and variance.

``` r
rf_accept_probs <- rf_accept$votes[, "1"]
actual_accept   <- train_clean$offer_accepted

roc_accept_rf <- roc(
  response = actual_accept,
  predictor = rf_accept_probs,
  levels = c("0", "1")
)
```

    ## Setting direction: controls < cases

``` r
auc_accept_rf <- auc(roc_accept_rf)

# Plot 
plot(roc_accept_rf,
     main = paste("ROC Curve - Acceptance Model\nAUC =", round(auc_accept_rf, 4)),
     col = "blue",
     lwd = 2)
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-28-1.png)<!-- -->

``` r
rf_default_probs <- rf_default$votes[, "1"]
actual_default   <- train_clean_all_accepted$default

roc_default_rf <- roc(
  response = actual_default,
  predictor = rf_default_probs,
  levels = c("0", "1")
)
```

    ## Setting direction: controls < cases

``` r
auc_default_rf <- auc(roc_default_rf)

# Plot
plot(roc_default_rf,
     main = paste("ROC Curve - Default Model\nAUC =", round(auc_default_rf, 4)),
     col = "red",
     lwd = 2)
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-29-1.png)<!-- -->

``` r
auc_accept_rf
```

    ## Area under the curve: 0.812

``` r
auc_default_rf
```

    ## Area under the curve: 0.5552

# Boosting

``` r
boost_data <- read.csv("/Users/lllllnx/Downloads/loan_training_data.csv")
```

``` r
boost_clean <- boost_data %>% mutate(
  income_to_loan = annual_income / (loan_amount * (1 + offered_rate)),
  loan_to_credit_lines = loan_amount / num_credit_lines,
  inquiries_to_delinquencies = inquiries_6months * 4 / delinquencies_2yrs,
  delinquencies_to_credit_lines = delinquencies_2yrs / num_credit_lines
) %>%
  drop_na(income_to_loan, loan_to_credit_lines, 
          inquiries_to_delinquencies, delinquencies_to_credit_lines) %>%
  filter(if_all(everything(), ~ !is.infinite(.)))
```

``` r
# Tune shrinkage and number of trees
train_idx <- sample(nrow(boost_clean), nrow(boost_clean)/2)
boost_train <- boost_clean[train_idx, ]
boost_test <- boost_clean[-train_idx, ]

library(gbm)
shrinkage_grid <- c(0.01, 0.10, 0.20, 0.001)

tune_res <- data.frame(
  shrinkage     = shrinkage_grid,
  mse           = NA,
  optimal_trees = NA
)

for (i in seq_along(shrinkage_grid)) {
  lam <- shrinkage_grid[i]

  fit <- gbm(
    offer_accepted ~ age + annual_income + credit_score + num_credit_lines +
      delinquencies_2yrs + credit_utilization + loan_amount +
      inquiries_6months + income_to_loan + loan_to_credit_lines +
      inquiries_to_delinquencies + delinquencies_to_credit_lines +
      offered_rate,
    data = boost_train,
    distribution = "bernoulli",
    n.trees = 2000,          # big upper bound
    interaction.depth = 4,
    shrinkage = lam,
    bag.fraction = 0.5,
    cv.folds = 5,
    verbose = FALSE
  )

  best_trees <- gbm.perf(fit, method = "cv", plot.it = FALSE)

  pred <- predict(fit, boost_test, n.trees = best_trees, type = "response")
  tune_res$mse[i] <- mean((pred - boost_test$offer_accepted)^2)
  tune_res$optimal_trees[i] <- best_trees
}

tune_res
```

    ##   shrinkage       mse optimal_trees
    ## 1     0.010 0.1324729           527
    ## 2     0.100 0.1304018            31
    ## 3     0.200 0.1386247            21
    ## 4     0.001 0.1302638          2000

During the gradient boosting model tuning, we tested different learning
rates (shrinkage values) to find the optimal trade-off between model
complexity and predictive accuracy. For each shrinkage value, we allowed
up to 2,000 trees and used 5-fold cross-validation to determine the
optimal number of trees. The results showed that with a smaller learning
rate of 0.001, the model required all 2,000 trees to achieve the best
performance, while larger learning rates converged faster—e.g., a
shrinkage of 0.10 only needed 31 trees. This illustrates the typical
trade-off in boosting: smaller learning rates often require more trees
but can produce more stable and accurate predictions, whereas larger
learning rates converge quickly but may risk overfitting or underfitting
if too aggressive.

``` r
set.seed(1)

boost_clean$acc_binary <- ifelse(boost_clean$offer_accepted == "1", 1, 0)

boost_acc <- gbm(offer_accepted ~ age + annual_income + credit_score + num_credit_lines + delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies + delinquencies_to_credit_lines + offered_rate, data = boost_train,
                   distribution = "bernoulli",  # For regression
                   n.trees = 2000,           # Number of trees
                   interaction.depth = 3,     # Tree depth
                   shrinkage = 0.001)         # Learning rate

boost_prob <- predict(boost_acc, boost_test, 
                     n.trees = 2000, type = "response")

boost_pred_class <- ifelse(boost_prob > 0.5, "Yes", "No")

# Make predictions
boost_mse <- mean((boost_prob - boost_test$offer_accepted)^2)

boost_mse
```

    ## [1] 0.1310116

``` r
boost_cv <- gbm(offer_accepted ~ age + annual_income + credit_score + num_credit_lines + delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies + delinquencies_to_credit_lines + offered_rate, data = boost_train,
               distribution = "bernoulli",
               n.trees = 2000,
               interaction.depth = 4,     # Fixed - not tuned by CV
               shrinkage = 0.01,          # Fixed - not tuned by CV
               cv.folds = 5)              # Enables CV for finding optimal trees
# Find optimal number of trees using CV error
best_iter <- gbm.perf(boost_cv, method = "cv")
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-35-1.png)<!-- -->

``` r
# Make predictions using optimal number of trees
pred_cv <- predict(boost_cv, boost_test, n.trees = best_iter, type = "response")
```

This graph shows how well the boosting model predicts the outcome as
more trees are added. At first, both lines drop quickly, meaning the
model is improving. The green line (cross-validation error) reaches its
lowest point around 350–400 trees—the blue dashed line marks this spot.
This is the point where the model performs best on new, unseen data.
After that, the green line starts to rise again, meaning the model
begins to overfit: it keeps getting better on the training data (black
line keeps going down) but worse on new data. So, the best number of
trees to use is the one at the lowest point of the green line, around
350–400.

``` r
par(mar = c(5, 12, 4, 2))

summary(boost_acc, cBars = 10, main = "Boosting Variable Importance for Acceptance",las = 2, cex.names = 0.7, cex.axis  = 0.7)
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-36-1.png)<!-- -->

    ##                                                         var     rel.inf
    ## offered_rate                                   offered_rate 71.16795507
    ## income_to_loan                               income_to_loan  9.07154635
    ## annual_income                                 annual_income  7.05749358
    ## credit_utilization                       credit_utilization  6.36237389
    ## credit_score                                   credit_score  3.22721435
    ## loan_to_credit_lines                   loan_to_credit_lines  1.22819873
    ## age                                                     age  0.72628578
    ## delinquencies_to_credit_lines delinquencies_to_credit_lines  0.51575503
    ## loan_amount                                     loan_amount  0.23299132
    ## num_credit_lines                           num_credit_lines  0.20163431
    ## inquiries_6months                         inquiries_6months  0.18255913
    ## inquiries_to_delinquencies       inquiries_to_delinquencies  0.01503232
    ## delinquencies_2yrs                       delinquencies_2yrs  0.01096015

This plot clearly highlights which features, such as offered rate,
income-to-loan ratio, and annual income are the strongest predictors of
loan acceptance, providing actionable insights for targeting potential
borrowers.

``` r
plot(boost_acc, i.var = "offered_rate",  main = "Partial Dependence: Offered Rate")
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-37-1.png)<!-- -->

``` r
boost_clean_def <- boost_clean %>%
  drop_na(default)

set.seed(1)

boost_clean_def$default_binary <- ifelse(boost_clean_def$default == "1", 1, 0)

train_idx1 <- sample(nrow(boost_clean_def), nrow(boost_clean_def)/2)
boost_train_def <- boost_clean_def[train_idx1, ]
boost_test_def <- boost_clean_def[-train_idx1, ]

shrinkage_grid <- c(0.01, 0.10, 0.20, 0.001)

tune_res_def <- data.frame(
  shrinkage     = shrinkage_grid,
  mse           = NA,
  optimal_trees = NA
)

for (i in seq_along(shrinkage_grid)) {
  lam <- shrinkage_grid[i]

  fit <- gbm(
    default ~ age + annual_income + credit_score + num_credit_lines +
      delinquencies_2yrs + credit_utilization + loan_amount +
      inquiries_6months + income_to_loan + loan_to_credit_lines +
      inquiries_to_delinquencies + delinquencies_to_credit_lines +
      offered_rate,
    data = boost_clean_def,
    distribution = "bernoulli",
    n.trees = 2000,          
    interaction.depth = 4,
    shrinkage = lam,
    bag.fraction = 0.5,
    cv.folds = 5,
    verbose = FALSE
  )

  best_trees_def <- gbm.perf(fit, method = "cv", plot.it = FALSE)

  pred_def <- predict(fit, boost_test_def, n.trees = best_trees_def, type = "response")
  tune_res_def$mse[i] <- mean((pred_def - boost_test_def$default)^2)
  tune_res_def$optimal_trees[i] <- best_trees_def
}

tune_res_def
```

    ##   shrinkage       mse optimal_trees
    ## 1     0.010 0.1681419            95
    ## 2     0.100 0.1768085             3
    ## 3     0.200 0.1754108             2
    ## 4     0.001 0.1658569          1217

This table compares different shrinkage values for the boosting model
and shows how they affect performance and the number of trees needed.
Smaller shrinkage values (like 0.001 and 0.01) require many more trees
to reach their best performance, because the model learns more slowly.
Larger shrinkage values (0.1 and 0.2) need only a few trees, but they
also produce slightly higher MSE, meaning their predictions are a bit
less accurate. Overall, the shrinkage value of 0.001 gives the lowest
MSE, but it needs over 1200 trees, while 0.01 gives almost as good
accuracy with far fewer trees (about 95), making it a more efficient
choice.

``` r
boost_cv_def <- gbm(default ~ age + annual_income + credit_score + num_credit_lines + delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies + delinquencies_to_credit_lines + offered_rate, data = boost_clean_def,
               distribution = "bernoulli",
               n.trees = 2000,
               interaction.depth = 4,     # Fixed - not tuned by CV
               shrinkage = 0.01,          # Fixed - not tuned by CV
               cv.folds = 5)              # Enables CV for finding optimal trees
# Find optimal number of trees using CV error
best_iter_def <- gbm.perf(boost_cv_def, method = "cv")
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-39-1.png)<!-- -->

``` r
# Make predictions using optimal number of trees
pred_cv_def <- predict(boost_cv_def, boost_test_def, n.trees = best_iter_def, type = "response")
```

This graph shows how the model’s error changes as more trees are added
during boosting. The black line is the training error, which keeps going
down because the model fits the training data better with each
iteration. The green line is the cross-validation error, which tells us
how well the model generalizes to new data. It decreases at first,
reaching its lowest point around the blue dashed line (about 250–300
trees), meaning the model performs best here. After that point, the
green line slowly rises, showing that adding more trees leads to
overfitting.

``` r
#build the model
library(gbm)
boost_def <- gbm(default ~ age + annual_income + credit_score + num_credit_lines + delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months + income_to_loan + loan_to_credit_lines + inquiries_to_delinquencies + delinquencies_to_credit_lines + offered_rate, data = boost_clean_def,
                   distribution = "bernoulli",  # For regression
                   n.trees = 2000,           # Number of trees
                   interaction.depth = 4,     # Tree depth
                   shrinkage = 0.01)         # Learning rate

boost_prob_def <- predict(boost_def, boost_test_def, 
                     n.trees = 95, type = "response")

boost_pred_class_def <- ifelse(boost_prob_def > 0.5, "Yes", "No")

# Make predictions
boost_mse_def <- mean((boost_prob_def - boost_test_def$default)^2)

boost_mse_def
```

    ## [1] 0.1684163

``` r
par(mar = c(5, 12, 4, 2))

summary(boost_def, cBars = 10, main = "Boosting Variable Importance for Default",las = 2, cex.names = 0.7, cex.axis  = 0.7)
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-41-1.png)<!-- -->

    ##                                                         var    rel.inf
    ## credit_utilization                       credit_utilization 14.4178854
    ## income_to_loan                               income_to_loan 12.6897918
    ## credit_score                                   credit_score 12.6644326
    ## annual_income                                 annual_income 12.5261424
    ## offered_rate                                   offered_rate 11.2525871
    ## age                                                     age  9.1341072
    ## loan_to_credit_lines                   loan_to_credit_lines  9.0135029
    ## loan_amount                                     loan_amount  6.8331926
    ## num_credit_lines                           num_credit_lines  3.8992318
    ## delinquencies_to_credit_lines delinquencies_to_credit_lines  3.8800588
    ## inquiries_to_delinquencies       inquiries_to_delinquencies  2.5609370
    ## inquiries_6months                         inquiries_6months  0.9033082
    ## delinquencies_2yrs                       delinquencies_2yrs  0.2248222

Visualizing them in a bar chart, we could clearly see which factors most
influence default risk. Key variables such as credit utilization, income
to loan ratio and credit score emerged as the strongest predictors,
highlighting the features most relevant for assessing borrower risk.

``` r
plot(boost_def, i.var = "credit_utilization",  main = "Partial Dependence: Credit Utilization")
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-42-1.png)<!-- -->

``` r
#compute AUC for acceptance model
boost_test$acc_binary <- ifelse(boost_test$offer_accepted == "1", 1, 0)
boost_test$acc_binary <- as.numeric(boost_test$acc_binary)

roc_accept <- roc(boost_test$acc_binary, boost_prob)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

``` r
auc_accept <- auc(roc_accept)
auc_accept   
```

    ## Area under the curve: 0.8817

``` r
plot(roc_accept, legacy.axes = TRUE, main = paste0("ROC Curve: Boosting for Acceptance (AUC = ", round(auc_accept, 4), ")"))
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-43-1.png)<!-- -->

``` r
boost_test_def$default_binary <- ifelse(boost_test_def$default == "1", 1, 0)
boost_test_def$default_binary <- as.numeric(boost_test_def$default_binary)

roc_default <- roc(boost_test_def$default_binary, boost_prob_def)
```

    ## Setting levels: control = 0, case = 1

    ## Setting direction: controls < cases

``` r
auc_default <- auc(roc_default)
auc_default   
```

    ## Area under the curve: 0.7192

``` r
plot(roc_accept, legacy.axes = TRUE, main = paste0("ROC Curve: Boosting for Default (AUC = ", round(auc_default, 4), ")"))
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-44-1.png)<!-- -->

# Linear Regression Model

``` r
## ACCEPTANCE - linear regression model 
train_clean$default <- as.numeric(train_clean$default)
train_clean$offer_accepted <- as.numeric(train_clean$offer_accepted)

model_linear_accept <- lm(
  offer_accepted ~ age + annual_income + credit_score + num_credit_lines +
    delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months +
    income_to_loan + loan_to_credit_lines +
    inquiries_to_delinquencies + delinquencies_to_credit_lines,
  data = train_clean
)

train_clean$pred_accept_linear <- predict(model_linear_accept)

roc_accept_linear <- roc(
  response = train_clean$offer_accepted,
  predictor = train_clean$pred_accept_linear
)
```

    ## Setting levels: control = 1, case = 2

    ## Setting direction: controls < cases

``` r
auc_accept_linear <- auc(roc_accept_linear)
print(paste("AUC - Acceptance:", round(auc_accept_linear, 4)))
```

    ## [1] "AUC - Acceptance: 0.7959"

``` r
plot(roc_accept_linear,
     legacy.axes = TRUE,
     main = "ROC Curve: Linear Model for Acceptance",
     print.auc = TRUE)
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-45-1.png)<!-- -->

``` r
RMSE_accept <- sqrt(mean((train_clean$offer_accepted - train_clean$pred_accept_linear)^2))
summary_accept <- summary(model_linear_accept)

metrics_accept_linear <- data.frame(
  R_squared = summary_accept$r.squared,
  Adj_R_squared = summary_accept$adj.r.squared,
  RMSE = RMSE_accept
)

metrics_accept_linear
```

    ##   R_squared Adj_R_squared     RMSE
    ## 1 0.2062786     0.1992906 0.409995

The linear model does not perform very well at predicting whether an
offer is accepted. The R-squared (0.206) and Adjusted R-squared (0.199)
show that the model explains only about 20% of the variance in offer
acceptance, which means most of the important patterns are not captured
by a linear relationship. The RMSE of about 0.41 indicates a fairly
large average prediction error, especially considering the outcome is
binary (0 or 1). Overall, these metrics indicate that a linear model is
not a strong predictor for this problem, and more flexible models (like
logistic regression, decision trees, or boosting) will likely perform
much better.

``` r
#DEFAULT - linear regression model 
# Convert outcomes to numeric (0/1)
train_clean_all_accepted$default <- as.numeric(train_clean_all_accepted$default)
train_clean_all_accepted$offer_accepted <- as.numeric(train_clean_all_accepted$offer_accepted)

model_linear_default <- lm(
  default ~ age + annual_income + credit_score + num_credit_lines +
    delinquencies_2yrs + credit_utilization + loan_amount + inquiries_6months +
    income_to_loan + loan_to_credit_lines +
    inquiries_to_delinquencies + delinquencies_to_credit_lines,
  data = train_clean_all_accepted
)

train_clean_all_accepted$pred_default_linear <- predict(model_linear_default)

roc_default_linear <- roc(
  response = train_clean_all_accepted$default,
  predictor = train_clean_all_accepted$pred_default_linear
)
```

    ## Setting levels: control = 1, case = 2

    ## Setting direction: controls < cases

``` r
auc_default_linear <- auc(roc_default_linear)
print(paste("AUC - Default:", round(auc_default_linear, 4)))
```

    ## [1] "AUC - Default: 0.6218"

``` r
plot(roc_default_linear,
     legacy.axes = TRUE,
     main = "ROC Curve: Linear Model for Default",
     print.auc = TRUE)
```

![](christy-nancy-nitya-report_files/figure-gfm/unnamed-chunk-46-1.png)<!-- -->

``` r
RMSE_default <- sqrt(mean((train_clean_all_accepted$default - train_clean_all_accepted$pred_default_linear)^2))
summary_default_linear <- summary(model_linear_default)

metrics_default_linear <- data.frame(
  R_squared = summary_default_linear$r.squared,
  Adj_R_squared = summary_default_linear$adj.r.squared,
  RMSE = RMSE_default
)

metrics_default_linear
```

    ##    R_squared Adj_R_squared      RMSE
    ## 1 0.03936843    0.02715701 0.4149946

For default, the linear model for predicting default performs very
poorly. The R-squared is only 0.039, meaning the model explains less
than 4% of the variation in whether someone defaults. The Adjusted
R-squared (0.027) is even lower, confirming that the predictors
contribute very little once model complexity is accounted for. The RMSE
is about 0.415, which is almost the same as the error you would get from
predicting the average default rate for everyone, showing that the model
does not improve much over a naïve baseline. Overall, these numbers
indicate that a linear model is not effective for predicting default,
likely because the relationship between the predictors and the outcome
is nonlinear and requires a more flexible modeling approach.

# Running Boosting on Test Data & Profit Maximization

``` r
test_data <- read.csv("/Users/lllllnx/Downloads/loan_test_data.csv")

library(dplyr)

test_clean <- test_data %>%
  mutate(
    loan_to_credit_lines = loan_amount / num_credit_lines,
    inquiries_to_delinquencies = inquiries_6months * 4 / delinquencies_2yrs,
    delinquencies_to_credit_lines = delinquencies_2yrs / num_credit_lines
  )

rate_min <- min(boost_clean$offered_rate, na.rm = TRUE)
rate_max <- max(boost_clean$offered_rate, na.rm = TRUE)

rate_grid <- seq(rate_min, rate_max, by = 0.005)

rate_min; rate_max
```

    ## [1] 0.02

    ## [1] 0.2354469

``` r
n_loans <- nrow(test_clean)
n_rates <- length(rate_grid)

profit_mat <- matrix(NA_real_, nrow = n_loans, ncol = n_rates)

for (j in seq_along(rate_grid)) {
  r <- rate_grid[j]
  
#creating a new df with new columns 
  df_rate <- test_clean %>%
    mutate(
      offered_rate = r,
      income_to_loan = annual_income / (loan_amount * (1 + offered_rate))
    )
  
  # 1) probability of acceptance (boost model)
  df_acc <- predict(
    boost_acc,
    newdata = df_rate,
    type = "response"
  )
  
  # 2) probability of default (boost model)
  df_def <- predict(
    boost_def,
    newdata = df_rate,
    type = "response"
  )
  
  # 3) expected profit for this rate r
  profit_if_paid   <- df_rate$loan_amount * r * 3 - 200
  loss_if_default  <- -(df_rate$loan_amount + 700)
  
  expected_profit_r <- df_acc * (
    (1 - df_def) * profit_if_paid +
      df_def * loss_if_default
  )
  
  profit_mat[, j] <- expected_profit_r
}
```

    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...
    ## 
    ## Using 2000 trees...

``` r
best_idx <- max.col(profit_mat, ties.method = "first")
test_clean$optimal_rate <- rate_grid[best_idx]
test_clean$expected_profit <- profit_mat[cbind(seq_len(n_loans), best_idx)]

test_clean$decision <- ifelse(test_clean$expected_profit > 0, "approve", "deny")
test_clean$expected_profit <- pmax(test_clean$expected_profit, 0)

total_expected_profit <- sum(test_clean$expected_profit)
total_expected_profit
```

    ## [1] 1335189

The procedure evaluates a set of possible interest rates for each loan
and calculates the expected profit associated with each rate. For every
loan, the model identifies the interest rate that produces the highest
expected profit, and this becomes the loan’s optimal rate. The model
then determines whether issuing the loan is financially worthwhile: if
the optimal expected profit is positive, the loan is approved; if the
expected profit is negative, the loan is denied and the profit is set to
zero. After applying this strategy across all 1,000 loans in the test
dataset, the total expected profit is \$1,335,189. This total represents
the profit that the portfolio is expected to generate under a policy
that selects the most profitable rate for each loan and approves only
those loans predicted to yield positive returns.

``` r
library(dplyr)
test_clean <- test_clean %>%
  rename(
    interest_rate   = optimal_rate
  )
df_profit_max <- test_clean %>% select(loan_id, decision, interest_rate)
```

``` r
write.csv(df_profit_max, "Christy_Nancy_Nitya_predictions.csv", row.names = FALSE)
```
