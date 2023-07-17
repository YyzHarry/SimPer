# SimPer: Simple Self-Supervised Learning of Periodic Targets

This repository contains the implementation code for paper: <br>
__[SimPer: Simple Self-Supervised Learning of Periodic Targets](https://arxiv.org/abs/2210.03115)__ <br>
Yuzhe Yang, Xin Liu, Jiang Wu, Silviu Borac, Dina Katabi, Ming-Zher Poh, Daniel McDuff <br>
_11th International Conference on Learning Representations (ICLR 2023), **Notable-Top-5% & Oral**_ <br>
[[Project Page](https://simper.csail.mit.edu/)] [[Paper](https://arxiv.org/abs/2210.03115)] [[Video](https://youtu.be/uEezGU3P_-I)] [Blog Post] [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YyzHarry/SimPer/blob/master/tutorial/tutorial.ipynb)
___
<p align="center">
    <img src="assets/simper.png" width="800"> <br>
<b>SimPer</b> learns robust <i><b>periodic representations</b></i> with high frequency resolution.
</p>


## Periodic SSL: Brief Introduction for SimPer
<p align="center">
    <img src="assets/motivation.gif" width="700"> 
</p>

From human physiology to environmental evolution, important processes in nature often exhibit meaningful and strong __periodic__ or __quasi-periodic__ changes. Due to their inherent label scarcity, learning useful representations for periodic tasks with limited or no supervision is of great benefit. Yet, existing self-supervised learning (SSL) methods overlook the intrinsic periodicity in data, and fail to learn representations that capture periodic or frequency attributes.

We present _SimPer_, a simple contrastive SSL regime for learning periodic information in data. To exploit the periodic inductive bias, SimPer introduces customized ___periodicity-invariant___ and ___periodicity-variant___ augmentations, ___periodic feature similarity measures___, and a ___generalized contrastive loss___ for learning efficient and robust periodic representations.

We benchmark SimPer on common real-world tasks in _human behavior analysis_, _environmental sensing_, and _healthcare_ domains. Further analysis also highlights its intriguing properties including better data efficiency, robustness to spurious correlations, and generalization to distribution shifts.


## Apply SimPer on Customized Datasets
To apply SimPer on customized datasets, you will need to define the following key components. (Check out [SimPer tutorial](https://github.com/YyzHarry/SimPer/tree/main/tutorial) for RotatingDigits dataset.)

#### #1: Periodicity-Variant and Invariant Augmentations *(see [src/augmentation.py](./src/augmentation.py))*
For (periodicity-)invariant augmentations, one could refer to SOTA contrastive methods (e.g., SimCLR). For periodicity-variant augmentations, we propose speed / frequency augmentation:
```python
import tensorflow as tf
import tensorflow_probability as tfp

def arbitrary_speed_subsample(frames, speed, max_frame_len, img_size, channels, **kwargs):
    ...

    x_ref = tf.range(0, speed * (len(frames) - 0.5), speed, dtype=tf.float32)
    x_ref = tf.stack([x_ref] * (img_size * img_size * channels))
    new_frames = tfp.math.batch_interp_regular_1d_grid(
        x=x_ref,
        x_ref_min=[0] * (img_size * img_size * channels),
        x_ref_max=[len(frames)] * (img_size * img_size * channels),
        y_ref=tf.transpose(tf.reshape(frames, [len(frames), -1]))
    )
    sequence = tf.reshape(
        tf.transpose(new_frames), frames.shape.as_list()
    )[:tf.cast(max_frame_len, tf.int32)]

    ...
```

#### #2: Periodic Feature Similarity *(see [src/simper.py](./src/simper.py))*
We provide practical instantiations to capture the periodic feature similarity, e.g., maximum cross-correlation:
```python
import tensorflow as tf

@tf.function
def _max_cross_corr(feats_1, feats_2):
    feats_2 = tf.cast(feats_2, feats_1.dtype)
    feats_1 = feats_1 - tf.math.reduce_mean(feats_1, axis=-1, keepdims=True)
    feats_2 = feats_2 - tf.math.reduce_mean(feats_2, axis=-1, keepdims=True)

    min_N = min(feats_1.shape[-1], feats_2.shape[-1])
    padded_N = max(feats_1.shape[-1], feats_2.shape[-1]) * 2
    feats_1_pad = tf.pad(feats_1, tf.constant([[0, 0], [0, padded_N - feats_1.shape[-1]]]))
    feats_2_pad = tf.pad(feats_2, tf.constant([[0, 0], [0, padded_N - feats_2.shape[-1]]]))

    X = tf.signal.rfft(feats_1_pad) * tf.math.conj(tf.signal.rfft(feats_2_pad))
    power_norm = tf.cast(tf.math.reduce_std(feats_1, axis=-1, keepdims=True) *
                         tf.math.reduce_std(feats_2, axis=-1, keepdims=True), X.dtype)
    power_norm = tf.where(tf.equal(power_norm, 0), tf.ones_like(power_norm), power_norm)
    X = X / power_norm

    cc = tf.signal.irfft(X) / (min_N - 1)
    max_cc = tf.math.reduce_max(cc, axis=-1)
    return max_cc
```

#### #3: Generalized InfoNCE Loss over Continuous Targets *(see [src/simper.py](./src/simper.py))*
First define label distance for continuous targets:
```python
import tensorflow as tf

def label_distance(labels_1, labels_2, dist_fn='l1', label_temperature=0.1):
    if dist_fn == 'l1':
        dist_mat = - tf.math.abs(labels_1[:, :, None] - labels_2[:, None, :])
    elif dist_fn == 'l2':
        ...

    return tf.nn.softmax(dist_mat / label_temperature, axis=-1)
```
Then calculate a weighted loss over all augmented pairs (soft regression variant):
```python
for features, labels in zip(all_features, all_labels):
    feat_dist = ...
    label_dist = ...
    criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    loss += criterion(y_pred=feat_dist, y_true=label_dist)
```


## Updates
- __[07/2023]__ We provide a [hands-on tutorial](https://github.com/YyzHarry/SimPer/tree/main/tutorial) of SimPer. Check it out! [![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YyzHarry/SimPer/blob/master/tutorial/tutorial.ipynb)
- __[06/2023]__ Check out the [Oral talk video](https://youtu.be/uEezGU3P_-I) (15 mins) for our paper.
- __[02/2023]__ Paper accepted to ICLR 2023 as __Notable-Top-5% & Oral Presentation__.
- __[10/2022]__ [arXiv version](https://arxiv.org/abs/2210.03115) posted. The code is currently under cleaning. Please stay tuned for updates.


## Citation
```bib
@inproceedings{yang2023simper,
  title={SimPer: Simple Self-Supervised Learning of Periodic Targets},
  author={Yang, Yuzhe and Liu, Xin and Wu, Jiang and Borac, Silviu and Katabi, Dina and Poh, Ming-Zher and McDuff, Daniel},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=EKpMeEV0hOo}
}
```
