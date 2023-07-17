# Hands-on Tutorial of SimPer

<p class="aligncenter">
    <a href="https://colab.research.google.com/github/YyzHarry/SimPer/blob/master/tutorial/tutorial.ipynb" target="_parent">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a> 
</p>


This is a hands-on tutorial for **SimPer: Simple Self-Supervised Learning of Periodic Targets** [[Paper]](https://arxiv.org/abs/2210.03115).

```bib
@inproceedings{yang2023simper,
  title={SimPer: Simple Self-Supervised Learning of Periodic Targets},
  author={Yang, Yuzhe and Liu, Xin and Wu, Jiang and Borac, Silviu and Katabi, Dina and Poh, Ming-Zher and McDuff, Daniel},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=EKpMeEV0hOo}
}
```

In this notebook, we will provide a hands-on tutorial for SimPer on the [RotatingDigits dataset](https://arxiv.org/pdf/2210.03115.pdf), as a quick overview on how to perform practical self-supervised learning with SimPer on custom periodic learning datasets.

You can directly open it via Colab: [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YyzHarry/SimPer/blob/master/tutorial/tutorial.ipynb), or using jupyter notebook with the following instructions.

Required packages:
```bash
pip install --upgrade pip
pip install --upgrade jupyter notebook
```

Then, please clone this repository to your computer using:

```bash
git clone https://github.com/YyzHarry/SimPer.git
```

After cloning is finished, you may go to the directory of this tutorial and run

```bash
jupyter notebook --port 8888
```

to start a jupyter notebook and access it through the browser. Finally, let's explore the notebook `tutorial.ipynb` prepared by us!
