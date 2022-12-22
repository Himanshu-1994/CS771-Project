# CS771-Project

This repo is an implementation of [this](https://arxiv.org/pdf/2112.10003.pdf) paper by Lüddecke et al which proposes a novel architecture built upon CLIP to segment images based on user prompts. Currently, this repo supports only text prompts.

To train this model, run the following command

```
python train.py --name=pc --batch-size=64 --max-iterations=2000 --learning-rate=0.001 --wd=1e-4
```

These are the most important arguments with ```train.py```

```yaml
name: set any name here
batch-size: [16, 32, 64, ..] (default 64)
max-iterations: set any feasible value here (default 20000)
ckpt: no. of iterations after which model is saved (default 1000)
image-size: (default 352)
amp: Automatic Mixed Precision (default True)
mix: set to False (only text support for now)
num_workers: number of workers in DataLoader (default 4)
lr: Learning Rate (default 0.001)
lrs: LR Scheduler (default cosine)
negative-prob: Negative Sampling (default 0.2)
wd: Weight decay with AdamW Optimizer (default 1e-4)

```
For other training arguments, you can look at the ```train.py``` script.

To then check for mIOU, mAP metrics using your trained model, you can run the following command

```
python score.py pc
```

Be sure to use the same experiment name in the command above as you did while training, to make sure the ```score.py``` script loads the correct model for inference.

For a quick failure case analysis on the HuggingFace model from [here](https://huggingface.co/CIDAS/clipseg-rd64-refined) run the following command

```
python failure_test.py
```

This will create a folder called ```failure_cases``` and save all inferences with mIOU less than 0.5 for 5 different thresholds (0.1, 0.2, 0.3, 0.4, 0.5)




