# Solving Structured Hierarchical Games Using Differential Backward Induction


The repository contains a Python implementation of Differential Backward Induction for the structured hierarchical games (SHGs). Each agent is assumed to have a one-dim strategy.

## Dependencies
[tqdm](https://tqdm.github.io/), [munch](https://pypi.org/project/munch/), [networkx](https://networkx.org/documentation/networkx-2.2/), pytorch, [hessian](https://github.com/mariogeiger/hessian)
## Usage: Public Good Games and Security Games

Run this command in the `SHG_DBI` folder.
```
python -m expPGG -mode <mode_name> -cfg_name <cfg_name>
```

TO conduct experiments of security game, you can use `expSec`  instead.

**Example:**

1. Run DBI:

```
python -m exp.expPGG -mode PGD -cfg_name pgg_cfg_test
```

2.   Run BRD:

```
python -m exp.expPGG -mode BRD -cfg_name pgg_cfg_test
```

3. Using BRD to evaluate DBI

```
python -m exp.expPGG -mode Eva_PGD -cfg_name pgg_cfg_test
```

4. Using BRD to evaluate BRD

```
python -m exp.expPGG -mode Eva_BRD -cfg_name pgg_cfg_test
```

**Configurations**

- The configurations are written using a simple tool from mmcv. [[doc](https://mmcv.readthedocs.io/en/latest/understand_mmcv/config.html)]
- Configurations are stored in `src/exp_configs/secg` and `src/exp_configs/npgg`. The configs used in the paper are provided. You can use your own configs following those examples.

**Results**

- Results are stored in `res/pgg_results` and `res/sec_results` as pickles.

<!-- ## Usage: work on your own game structure

```Python
from src.gradient import Gradients
from src.grad_algs import *
from src.HBRD import HBrd
from src.payoff_func import Payoff
```
First, you need to build your own payoff function -->
## Citation

If you used the DBI in your work, please cite us using the following BibTeX entry:

<pre>
@inproceedings{li2022solving,
  title={Solving structured hierarchical games using differential backward induction},
  author={Li, Zun and Jia, Feiran and Mate, Aditya and Jabbari, Shahin and Chakraborty, Mithun and Tambe, Milind and Vorobeychik, Yevgeniy},
  booktitle={Uncertainty in Artificial Intelligence},
  pages={1107--1117},
  year={2022},
  organization={PMLR}
}
</pre>
