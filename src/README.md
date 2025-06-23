## Directory overview
This README provides a brief overview of the scripts used for training and evaluating the methods.

- `train.ipynb` - Notebook for training the baseline model and applying subsequent fine-tuning strategies.
- `GAT` directory - Directory containing the script for training a Graph attention network (GAT).
- `baseline-with-unobserved-residues.ipynb` - Notebook for training the baseline model using embeddings that include unobserved residues for additional context.
- `decision-thresholds.ipynb` - file for determining the decision threshold of each finetuned method based on Mathew correlation coefficient
- `baseline-with-unobserved-residues.ipynb` - Notebook for determining optimal decision thresholds for each fine-tuned method based on the Matthews Correlation Coefficient (MCC).
 