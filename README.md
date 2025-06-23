# Hidden in protein sequences: Predicting cryptic binding sites
Binding sites that exhibit significant conformational changes are often referred to as cryptic binding sites (**CBS**). As correct prediction of these sites heavily depend on the particular protein conformation, structure-based methods often fail to correctly predict these sites. In this repository, we present various sequence-based approaches including finetuning protein language models for improving the model's performance.

![](https://github.com/skrhakv/cryptic-finetuning/blob/master/img/4gi1A-1dteA.png?raw=true)
*Example of a cryptic binding site from the CryptoBench dataset. Superposition of two conformations of Thermomyces lanuginosa lipase binding 16-hydroxypalmitic acid (red). The ligand-bound conformation (orange; PDB ID: 4gi1, chain A) reveals a binding pocket, while in the ligand-free conformation (green; PDB ID: 1dte, chain A), a shift in the α-helix closes the site (blue arrow), preventing fitting the ligand.*

## About
This work extends the CryptoBench dataset, which includes over 1,000 protein structures annotated with cryptic binding sites. We begin with simple transfer learning approach and incrementally add steps to the model to evaluate their impact compared to the baseline.

To replicate our results or apply our methods to your own data, download the necessary files from [this link](https://owncloud.cesnet.cz/index.php/s/f3YEUJYyOrTZa12).

## Contact us 
Have questions or suggestions? Feel free to open [an issue!](https://github.com/skrhakv/cryptic-finetuning/issues)

## License
This source code is licensed under the [MIT License](https://github.com/skrhakv/cryptic-finetuning/blob/master/LICENSE).