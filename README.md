# Hidden in protein sequences: Predicting cryptic binding sites
Binding sites that exhibit significant conformational changes are often referred to as cryptic binding sites (**CBS**). As correct prediction of these sites heavily depend on the particular protein conformation, structure-based methods often fail to correctly predict these sites. In this repository, we present various sequence-based approaches including finetuning protein language models for improving the model's performance.

<p align="center">
  <img src="https://github.com/skrhakv/cryptic-finetuning/blob/master/img/4gi1A-1dteA.png?raw=true" />
</p>

*Example of a cryptic binding site from the CryptoBench dataset. Superposition of two conformations of Thermomyces lanuginosa lipase binding 16-hydroxypalmitic acid (red). The ligand-bound conformation (orange; PDB ID: 4gi1, chain A) reveals a binding pocket, while in the ligand-free conformation (green; PDB ID: 1dte, chain A), a shift in the α-helix closes the site (blue arrow), preventing fitting the ligand.*

## About
This work extends the [CryptoBench study](academic.oup.com/bioinformatics/article/41/1/btae745/7927823). We examine various approaches for training on the CryptoBench dataset, which includes over 1,000 protein structures annotated with cryptic binding sites. We begin with simple transfer learning approach and incrementally add steps to the model to evaluate their impact compared to the baseline.

To replicate our results or apply our methods to your own data, download the necessary files from [this link](https://owncloud.cesnet.cz/index.php/s/f3YEUJYyOrTZa12).

## Contact us 
Have questions or suggestions? Feel free to open [an issue!](https://github.com/skrhakv/cryptic-finetuning/issues)

## How to cite:
If you find our work useful, please cite [the paper](https://dl.acm.org/doi/10.1145/3765612.3767221):

- *Vít Škrhák, Hamza Gamouh, and David Hoksza. 2025. Hidden in protein sequences: Predicting cryptic binding sites. Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics. Association for Computing Machinery, New York, NY, USA, Article 47, 1–6. [https://doi.org/10.1145/3765612.3767221](https://doi.org/10.1145/3765612.3767221)*


or, if you prefer the `BibTeX` format:

```
@inbook{10.1145/3765612.3767221,
author = {Škrhák, Vít and Gamouh, Hamza and Hoksza, David},
title = {Hidden in protein sequences: Predicting cryptic binding sites},
year = {2025},
isbn = {9798400722004},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3765612.3767221},
abstract = {Identifying protein binding sites is essential for understanding biological processes and advancing drug discovery. However, cryptic binding sites - transient or hidden pockets that emerge only under certain conditions - pose a significant challenge due to their conformational flexibility. Traditional structure-based prediction methods often fail to detect these sites, as they rely on rigid protein conformations. Therefore, the focus of this study is directed towards sequence-based approaches, especially protein language models, which do not depend on particular protein conformation. Starting with a baseline method utilizing simple transfer learning, we explore a range of finetuning strategies, each evaluated against the baseline to assess the improvement of prediction performance. We incorporate multi-task learning by jointly training the model to predict protein flexibility, a key characteristic of cryptic sites. More importantly, we leverage data from datasets of general binding sites to further enhance model performance. With properly selected fine-tuning techniques, we were able to improve the prediction across all key metrics. Our results demonstrate that, while sequence-based models hold strong potential for uncovering cryptic binding sites, their performance can be significantly enhanced through carefully selected finetuning strategies and data sources.},
booktitle = {Proceedings of the 16th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics},
articleno = {47},
numpages = {6}
}
```

## License
This source code is licensed under the [MIT License](https://github.com/skrhakv/cryptic-finetuning/blob/master/LICENSE).
