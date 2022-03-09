# Detecting Relevant App Reviews for Software Evolution and Maintenance through Multimodal One-Class Learning
- Marcos Gôlo (ICMC/USP) | marcosgolo@usp.br (corresponding author)
- Adailton Ferreira Araujo (ICMC/USP) | adailton.araujo@usp.br
- Rafael Rossi (UFMS) | rafael.g.rossi@ufms.br
- Ricardo Marcacini (ICMC/USP) | ricardo.marcacini@icmc.usp.br

# Abstract
- **Context**: Mobile app reviews are a rich source of information for software evolution and maintenance. Several studies have shown the effectiveness of exploring relevant reviews in the software development lifecycle, such as release planning and requirements engineering tasks. Popular apps receive even millions of reviews, thereby making manual extraction of relevant information an impractical task. The literature presents several machine learning approaches to detect relevant reviews. However, these approaches use multi-class learning, implying more user effort for data labeling since users must label a significant set of relevant and irrelevant reviews. 

- **Objective**: This article investigates methods for detecting relevant app reviews considering scenarios with small sets of labeled data. We evaluated unimodal and multimodal representations, different labeling levels, as well as different app review domains and languages.

- **Method**: We present a one-class multimodal learning method for detecting relevant reviews. Our approaches have two main contributions. First, we use one-class learning that requires only the labeling of relevant app reviews, thereby minimizing the labeling effort. Second, to handle the smaller amount of labeled reviews without harming classification performance, we also present methods to improve feature extraction and reviews representation. We propose the Multimodal Autoencoder and the Multimodal Variational Autoencoder. The methods learn representations which explore both textual data and visual information based on the density of the reviews. Density information can be interpreted as a summary of the main topics or clusters extracted from the reviews.  

- **Results**: Our methods achieved competitive results even using only 25\% of labeled reviews compared to models that used the entire training set. Also, our multimodal approaches obtain the highest $F_1$-Score and AUC-ROC in twenty-three out of twenty-four scenarios.

- **Conclusion**: Our one-class multimodal methods proved to be a competitive alternative for detecting relevant reviews and promising for practical scenarios involving data-driven software evolution and maintenance.

# Proposal: Mutlimodal One-Class Learning
![Proposal](/images/proposal.png)

# Results
![Results](/images/results.png)

# Critical Diference
![Results](/images/nemenyi.png)

# File Organization
- Pipfiles: contains the versions of the libraries used
- Best Results: different organizations of the best results of each method in each of the scenarios
- Results: total results of each method in each scenario considering all parameters used
- Codes: source codes used for the study experiments








