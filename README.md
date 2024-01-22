# Project: Adversarial Machine Learning Website Fingerprinting using Proxy
Website Fingerprinting Analysis of Encrypted Network Traffic by Machine Learning Techniques Using Proxy Server

## Abstract
In the world of computers, keeping information private and anonymous browsing are crucial. Imagine a simple scenario where you want to visit a website without anyone knowing. To achieve this, we use a proxy server, acting as a middleman between you and the web server. The proxy server ensures that your internet service provider sees only a connection to the proxy, not the actual website you're visiting.

However, there's a potential risk to anonymity called "website fingerprinting." This is when someone tries to figure out which site you visited by analyzing patterns in the data flow. They look at things like packet size, directions, timestamps, and loading time. To understand and defend against this, we collect data by capturing network traffic while browsing various web pages. We then use machine learning algorithms during training and testing to predict the likely website you visited through the proxy.

### Keywords
Website fingerprinting, proxy server, crawling, capturing traffic, interpolation, normalization, outlier detection, IQR, Z-score, isolation forest, Support Vector Machine (SVM), k-nearest neighbor, random forest, neural network, cross validation, confusion matrix.

Project is processed by following steps shown in Figure 1: Step by step flow diagram for project implementation

![Main_Framework (1)](https://github.com/khansiddique/Project-Adversarial-Machine-Learning-Website-Fingerprinting-Proxy/assets/44813868/63c7e4e0-cfed-4ed5-8878-98b3f3d6bd80)


## Explanation:

**Website Fingerprinting:** Identifying websites based on patterns in data flow.

**Proxy Server:** A middleman between the user and web server, ensuring anonymity.

**Crawling:** Visiting different web pages and collecting data for analysis.

**Capturing Traffic:** Recording network data to understand communication patterns.

**Interpolation:** Estimating values between known data points for a smoother analysis.

**Normalization:** Adjusting data to a standard scale for consistent comparisons.

**Outlier Detection, IQR, Z-Score, Isolation Forest:** Methods to identify abnormal patterns in data.

**Machine Learning Algorithms (SVM, k-NN, Random Forest, Neural Network):** Using AI to predict website visits based on training data.

**Cross Validation:** Ensuring model reliability by testing it on different data subsets.

**Confusion Matrix:** Evaluating the performance of machine learning models.
