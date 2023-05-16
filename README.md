# SGCL

Codes of **Sgcl: Contrastive representation learning for signed graphs[C]//Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021.**

## ContrastGNN_whole & ContrastGNN/
ContrastGNN/ is actually the same with ContrastGNN_whole.py, where the difference is that ContrastGNN/ consists of multiple files and ContrastGNN_whole.py is a single file.

## Note
There is some differences between the code and the paper.
1. **The meaning of $\alpha$ and $\beta$ in the code is opposite to that in the paper!!!**
2. **Sign perturbation** corresponds to **args.augment=change**, **Connectivity perturbation** corresponds to **args.augment=delete**, **Composite** correponds to **args.augment=composite**.

## Parameter Settings
#### BitCoinAlpha
args.alpha = 1e-4, args.beta = 0.8, args.mask_ratio = 0.1
#### BitCoinOTC
args.alpha = 1e-4, args.beta = 0.2, args.mask_ratio = 0.1
