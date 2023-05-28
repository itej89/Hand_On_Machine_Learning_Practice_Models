# Types of Machine learning
## Supervised
1. K-nearest neighbours
    * multilabel classification
    * multi output classification
2. Linear Regression
    * Error fucntion : mse,mae(mean absolute)
    * Evaluations : cross validations
    * Tuning: Grid CV, RandomforestCV
    * R-Value
    * F-score
    * P-Value
3. Logistic Regression
    * Binary Classifier
    * Cross Validation
    * accuracy-precision-recall(threshold)
    * roc curve
4. Support Vector Machines
5. Decision Trees
    * Error fucntion : mse
    * Evaluations : cross validations
    * Tuning: Grid CV, RandomforestCV
6. RandomForest Regression: trains multiple decision trees
    * Error fucntion : mse
    * Evaluations : cross validations
    * Tuning: Grid CV, RandomforestCV
7. Binary Classification
    * Accuracy
    * Precision
    * Recall
    * f1 score
    * Precision recall curve
    * roc curve
    * roc area under the curve score
8. Neural Networks
    * Regression
        * Activation : 
            * Relu/Softplus for positive values
            * hyperbolic tangent and logistic function for bounded output
                * hyperbolic tangent : scale out from [-1 1] to bounding limits
                * logistic function : scale output from [0, 1] to bounding limits
            * Loss functions: MSE, MAE or Huber loss(combination of MAE and MSE)
                
    * Classification

## Unsupervised
1. CLustering
    * K-means
    * DBSCAN
    * Hierarchial Cluster Analysis(HCA)
2. Anamoly Detection
    * One Class SVM
    * Isolation Forest
3. Visualization and Dimentionality reduction
    * Principal Component Analysis (PCA)
    * Kernel PCA
    * Locally Linear Embedding (LLE)
    * t-distributed Stochastic Neighbour Embedding
4. Association Rule learning
    * Apriori


## Semisupervised
1. Some labeled Data & lot of unlabeled data (Google photos)
2. Ex: Deep Belief networks
        * These are Restricted Boltzmann machienes stacked on top of one another
        * Trained sequentially in an unsuperviced manner
        * Then fine tuned using supervised learning techniques

            
## Reinforcement
* Learnign system called agent
* Observes environment
* Selects and performs action
* get rewards
* Update policy to get most reward overtime
* repeat


## When are they trained
1. Online learning
    1. System is trained incrementally using mini-batches
    2. Great for systems taht recieves continuous flow of data
    3. where fast adaptation is required (ex:stock price)
    4. Limited computing resources: No need to keep old data
    5. Can be used to train huge data that can not be fit into systems main memory (out-of-core learning)
    Problems:
        Tuning learning Rate
        Bad data could degrate the system overtime
        Need to keep monitoring incoming data for anomalies

2. Batch learning
    1. Offline learning
    2. Incapable of learning incrementally
    3. Done offline
    4. Trained using all available data
    5. New version of system need to be created when new data is available
    6. Can be automated, but takes lot of time and computing resources


## How they predict the output
1. Instance based (by heart)
    1. Uses a measure of similarity between known data and input data

2. Pattern detection (Mathematical Model)
