# Credit Risk Predictive Modelling

**TL;DR**  
**Business problem:** Peer-to-peer lenders risk funding loans likely to default; even a small number of bad approvals can wipe out interest gains.  
**Method:** Logistic Regression baseline with class weighting/SMOTE and threshold tuning to prioritize recall on high-risk loans; reproducible pipeline with tests and CI.  
**Impact:** On sample data, ~0.95 balanced accuracy with ~0.90+ recall for the High-Risk class, enabling policy-driven cutoffs for manual review and reduced expected loss.



Overview of the Analysis
The primary objective of this analysis was to develop machine learning models to predict loan risk based on financial (i.e. lending) data.

This model was made to predict and classify loans into two categories: Healthy Loans (Class 0) and High-Risk Loans (Class 1) based on the relevant financial risk factors or repayment potential (cf. "lending_data.csv").

Ultimately, this was to predict the credit worthiness of potential borrowers from peer-to-peer lending services, allowing lenders to better assess their own risk exposure when deciding to grant a loan.

The following financial data was utilized: loan size, interest rate, borrower income, debt-to-income ratio, number of credit accounts, derogatory marks, and total debt as predictive factors for "loan status".

The lending data (lending_data.csv) was first loaded into a DataFrame and the "loan_status" column was designated as the target variable (y). The remaining columns were assigned as the features (X) (loan_size, interest_rate, borrower_income, debt_to_income, num_of_accounts, derogatory_marks, total_debt, loan_status).

*Machine Learning Process

The data was split into training and testing sets for model training, fit, and subsequent evaluation.

In the analysis, the scikit-learn library's LogisticRegression method was used for its suitability in binary classification tasks (in this case, Healthy versus High-Risk loans) and was trained using the training data from the CSV file. A logistic regression model was applied and fitted by using the training data (X_train and y_train). This method provided predictions for the test portion of the data, and the following evaluation metrics, including balanced accuracy score, confusion matrix, and classification report, were computed to assess the model's performance.

A confusion matrix was then generated.

The classification report was subsequently printed.

These evaluation metrics were used to test the model for both labels of loans: The accuracy scores indicate how well the model classifies instances correctly overall and in a balanced manner. Precision scores indicate the proportion of correctly identified instances among the instances classified as belonging to a particular class. Finally, recall scores indicate the proportion of correctly identified instances of a class among all instances that truly belong to that class.

Results
Accuracy scores and the precision and recall scores of all machine learning models:

Results Accuracy: 0.99185

Balanced Accuracy: 0.95205

Precision Healthy: 1.00 High-Risk: 0.85

Recall Healthy: 0.99 High-Risk: 0.91

Analysis of Results and Summary
The Logistic Regression model overall does an excellent job in its ability to accurately predict loan creditworthiness with its loan classification instances.

Accuracy: The overall accuracy of the model is very high at 99.185%. Balanced accuracy, which takes into account class imbalances, is also high at 95%. This suggests that the model performs well across both classes, even if they have different proportions in the dataset--and looking at the raw data, one can see that Healthy loans compose a far larger class than High-Risk do.

Precision: In a similar dynamic per the model's accuracy measurements, the Precision is also higher in the "Healthy" class, the precision of the model's predictions is 100%; it is a bit lower for the "High-Risk" class at 87%. Still, despite this 13% false-positive rate, the model has a reliable classification performance in correctly identifying data belonging in the "High-Risk" loan class.

Recall: The recall for the "Healthy" class is very high at over 99%, showing alow false negative rate for the "Healthy" class. The recall for the "High-Risk" class is also high at 91%. This all indicates good performance in capturing "High-Risk" instances, although slightly lower than the recall for the "Healthy" class.

Analysis of "Best Fit" The predictionss for the Healthy loans (=0) perform the better of the two classes. The prediction accuracy is near perfect and there are many more Healthy loans data points amongst the dataset to classify. On the other hand, High-Risk loans (=1) are disproportionately less in number so the model has less data to work with in its classification and results in nearly 13% of High-Risk loans being misclassifie as Healthy.

Prioritization of predictions (i.e. which class is the focus?)

While none of the loans used for training are for very large amounts of money, a loan that defaults could cost more than the interest earned from many healthy loans.

As mentioned, there are also very few High-Risk loans for the model to learn from when compared to the number of Healthy loans in the data set. If more of these types of loans could be added to the data to train from, providing more balance, there may be improvement to the model. For real world implications, preventing High-Risk loans is more of a priority than giving out a Healthy loan as more money can be lost from a single loan that defaults than the interest earned from a loan.

Further Recommendations If this model is only used for loans less than $24,000 and historically the bank has had such a high ratio of Healthy to High-Risk loans, this logistic regression may may be adequate; however, to better prevent the poor credit worthiness of potential risky borrowers, a more complex model is needed-- a model that gives out more than binary answers (healthy/risky), and perhaps more indepth predictive analysis for loans which are on the edge of Healthy versus High-Risk.
