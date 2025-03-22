# Comparative Study Between Ensemble Learning and Evolutionary Learning to Solve the Higgs Boson Detection

## Authors:
- Ahmed MAALOUL
- Aksel YILMAZ

## Environment:
- Google Colab with GPU enabled (T4) for accelerated computation.

## Dataset Description and Aim

### Dataset Description
**Title**: Dataset from the ATLAS Higgs Boson Machine Learning Challenge 2014
**Source**: ATLAS Collaboration, CERN Open Data Portal

This dataset was created for the ATLAS Higgs Boson Machine Learning Challenge and is derived from official ATLAS full-detector simulations. It contains events representing "Higgs to tau-tau" signals mixed with various background processes. The events are simulated to mimic real detector data, leveraging our knowledge of particle physics. Each event includes detailed properties such as transverse momentum, invariant mass, and pseudorapidity, among others, which are crucial for classification tasks.

#### Key Characteristics:
- **Number of events**: 818,238
- **File size**: 186.5 MiB
- **Evaluation Metric**: Approximate Median Significance (AMS)

#### Simulation Process:
1. Random proton-proton collisions are simulated based on existing particle physics knowledge.
2. Resulting particles are tracked through a virtual model of the detector, yielding simulated events with properties mimicking real-world collisions.

#### Signal and Background:
- **Signal Events**: Events in which Higgs bosons (with a fixed mass of 125 GeV) are produced.
- **Background Events**: Other known processes producing similar signatures, including:
  - Decay of Z bosons into two taus.
  - Events with top quark pairs producing leptons and hadronic taus.
  - W boson decays producing electrons or muons with hadronic taus (due to particle identification imperfections).

#### Weights:
Each simulated event has a weight proportional to its conditional density. These weights are crucial for accurate evaluation of the AMS metric but are excluded as inputs to the classifier.

---

### Aim of the Analysis
The goal is to develop and compare machine learning models for detecting Higgs boson events based on the dataset’s features. Specifically, the objectives are:
1. **Preprocessing**: Handle missing values, normalize features, and prepare the data for modeling.
2. **Model Implementation**:
   - Apply ensemble learning methods (e.g., Random Forest, Gradient Boosting).
   - Use evolutionary learning techniques to optimize performance.
3. **Performance Evaluation**:
   - Assess models using metrics such as AMS, accuracy, precision, recall, and F1-score.
   - Compare ensemble and evolutionary learning approaches in terms of performance and computational efficiency.

---

## Feature Descriptions

| **Variable**                 | **Description**                                                                                  |
|------------------------------|--------------------------------------------------------------------------------------------------|
| EventId                     | Unique identifier for the event.                                                               |
| DER_mass_MMC                | Estimated mass of the Higgs boson candidate.                                                   |
| DER_mass_transverse_met_lep | Transverse mass between missing transverse energy and the lepton.                              |
| DER_mass_vis                | Invariant mass of the hadronic tau and the lepton.                                             |
| DER_pt_h                    | Modulus of the vector sum of transverse momenta (hadronic tau, lepton, missing energy).        |
| DER_deltaeta_jet_jet        | Pseudorapidity separation between the two jets. Undefined if PRI_jet_num ≤ 1.                 |
| DER_mass_jet_jet            | Invariant mass of the two jets. Undefined if PRI_jet_num ≤ 1.                               |
| DER_prodeta_jet_jet         | Product of pseudorapidities of the two jets. Undefined if PRI_jet_num ≤ 1.                  |
| DER_deltar_tau_lep          | R separation between the hadronic tau and the lepton.                                          |
| DER_pt_tot                  | Modulus of the vector sum of transverse momenta of key particles.                              |
| DER_sum_pt                  | Sum of transverse momenta of all key particles and additional jets.                            |
| DER_pt_ratio_lep_tau        | Ratio of transverse momenta of the lepton and the hadronic tau.                                |
| DER_met_phi_centrality      | Centrality of the azimuthal angle of missing transverse energy relative to the tau and lepton. |
| DER_lep_eta_centrality      | Centrality of the pseudorapidity of the lepton relative to the two jets. Undefined if PRI_jet_num ≤ 1. |
| PRI_tau_pt                  | Transverse momentum of the hadronic tau.                                                       |
| PRI_tau_eta                 | Pseudorapidity of the hadronic tau.                                                            |
| PRI_tau_phi                 | Azimuth angle of the hadronic tau.                                                             |
| PRI_lep_pt                  | Transverse momentum of the lepton.                                                             |
| PRI_lep_eta                 | Pseudorapidity of the lepton.                                                                  |
| PRI_lep_phi                 | Azimuth angle of the lepton.                                                                   |
| PRI_met                     | Missing transverse energy.                                                                     |
| PRI_met_phi                 | Azimuth angle of the missing transverse energy.                                                |
| PRI_met_sumet               | Total transverse energy in the detector.                                                      |
| PRI_jet_num                 | Number of jets. Possible values: 0, 1, 2, or 3.                                                |
| PRI_jet_leading_pt          | Transverse momentum of the leading jet. Undefined if PRI_jet_num = 0.                         |
| PRI_jet_leading_eta         | Pseudorapidity of the leading jet. Undefined if PRI_jet_num = 0.                               |
| PRI_jet_leading_phi         | Azimuth angle of the leading jet. Undefined if PRI_jet_num = 0.                                |
| PRI_jet_subleading_pt       | Transverse momentum of the subleading jet. Undefined if PRI_jet_num ≤ 1.                    |
| PRI_jet_subleading_eta      | Pseudorapidity of the subleading jet. Undefined if PRI_jet_num ≤ 1.                         |
| PRI_jet_subleading_phi       | Azimuth angle of the subleading jet. Undefined if PRI_jet_num ≤ 1.                          |
| PRI_jet_all_pt              | Scalar sum of transverse momentum of all jets in the event.                                    |
| Weight                      | Event weight proportional to the conditional density.                                          |
| Label                       | Event label: ‘s’ for signal and ‘b’ for background.                                        |
| KaggleSet                   | Specifies the Kaggle dataset: training, public leaderboard, or private leaderboard.            |
| KaggleWeight                | Weight normalized within each Kaggle dataset.                                                 |

---

## Data Visualization and Preprocessing

### Removing Unnecessary Columns
- **EventId**: Unique identifier for each event.
- **KaggleSet**: Indicates the subset of the data (e.g., training, public leaderboard, private leaderboard).
- **KaggleWeight**: Normalized weights within each Kaggle dataset.

### Random Sampling
We perform random sampling to create a representative subset of the dataset, allowing for faster preprocessing.

### Mapping the Label to Numerical Values
The `Label` column is mapped to numerical values (`s` to 1 and `b` to 0).

### Dealing with Missing Values
Missing values, represented by `-999`, are imputed using the `SimpleImputer` class with a mean strategy.

### Outliers
Outliers are checked using boxplots for all columns.

### Feature Scaling
Given the significant number of outliers, `StandardScaler` is used for feature scaling.

### Crossing Certain Pairs of Columns
Scatter plots are generated for selected pairs of columns to visualize the distribution of signal and background events.

### Unbalance Ratio
The dataset contains significantly more background (0) events than signal (1) events, indicating a class imbalance.

### Feature Selection
A correlation matrix is used to identify highly correlated features, indicating redundancy in the dataset. Principal Component Analysis (PCA) is applied to retain 95% variance.

### Split the Data
The dataset is split into training and testing sets, maintaining the class distribution.

---

## Ensemble Learning

### Ensemble Learning Methods
- **Bagging (Bootstrap Aggregating)**: Combines multiple individual models (often decision trees) to produce a more robust final prediction.
- **Boosting**: Trains models sequentially, where each new model focuses on correcting the errors of the previous ensemble.

### Classifier Candidates
- **RandomForestClassifier (Bagging)**
- **XGBClassifier (Boosting)**
- **GradientBoostingClassifier (Boosting)**
- **GaussianNB (Naive Bayes)**

### Training, Predicting, and Storing Results
The models are trained and evaluated using different sampling methods (None, UnderSampling, OverSampling, SMOTE). The results are stored in a DataFrame.

---

## Evolutionary Learning Implementation with DEAP

### Genetic Programming Framework
- **Primitive Set**: Defines the basic operations and functions.
- **Fitness and Individual Classes**: Defines the fitness function and individual representation.
- **Toolbox Setup**: Sets up the genetic operators and evaluation functions.

### Training and Storing Metrics
The genetic programming models are trained using `eaSimple` and `eaMuPlusLambda` algorithms. The results are stored in a DataFrame.

### Bonus: Voting Classifier with the Final Population GP
A voting classifier is created from the top individuals in the final population and evaluated on the test set.

---

## Comparative Analysis

### Performance
- **Accuracy & F1-Score**: XGBoost consistently attains the highest accuracy and F1-score, followed by Random Forest and Gradient Boosting. GP methods achieve moderate performance.
- **Precision & Recall**: Ensemble methods dominate in precision and recall, with GP methods achieving moderate performance.

### Efficiency
- **Training Time**: Gradient Boosting and GP methods incur higher training times. XGBoost benefits from GPU support and achieves faster training times.
- **Testing Time**: GP Voting Ensemble has the highest inference cost. XGBoost maintains fast prediction times.

### Model Complexity
- **Interpretability**: Ensemble methods are seen as "black box" approaches, while GP solutions can be more transparent if they remain shallow.
- **Evolved Models vs. Ensemble Methods**: Both ensembles and GP can grow complex, with actual simplicity depending on hyperparameter constraints.

### Advantages and Disadvantages
- **Ensemble Learning**: Excellent predictive performance, scalable to large data, but limited global interpretability.
- **Evolutionary Learning**: Flexible approach for discovering novel model structures, but computationally costly and less accurate than top ensembles.
- **GP Voting Ensembles**: Can bolster GP performance but incurs high inference costs.

### Conclusion
XGBoost emerges as the optimal method for Higgs Boson detection, excelling in accuracy, speed, and predictive robustness. GP approaches, particularly `eaMuPlusLambda`, demonstrate greater effectiveness than `eaSimple` but remain slower and less accurate compared to leading ensemble methods.
