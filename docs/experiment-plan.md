# Experiment plan

Task 1 is evaluated on training-time optimization and the quality of the
analysis. Each run should record configuration, training loss, wall-clock
time, CPU/RAM use, and caption examples on the fixed validation split.

## Baseline

- Model: `Salesforce/blip-image-captioning-base`
- Training: none
- Validation inference: 5 images on CPU
- Mean caption generation time: 2.36 seconds/image
- Observation: the pretrained model identifies broad scene content but misses
  remote-sensing detail and occasionally produces implausible phrases.

## Planned comparisons

| Run | Change | Why it is measured |
| --- | --- | --- |
| B0 | Pretrained BLIP, no VRSBench fine-tuning | Reference caption quality and CPU latency |
| T1 | Frozen visual encoder, train language decoder | Reduces backpropagation cost and memory |
| T2 | Same as T1 with learning-rate comparison | Measures convergence and stability |
| T3 | Gradient accumulation versus direct updates | Measures memory and time trade-off |

Do not claim results for T1-T3 until their logs and validation outputs exist.

## Observed CPU pilot: T1 smoke test

| Measurement | Result |
| --- | --- |
| Training examples | 25 |
| Epochs | 1 |
| Learning rate | 1e-5 |
| Caption token limit | 96 |
| Frozen component | BLIP visual encoder |
| Trainable parameters | 137,881,148 |
| Initial training loss | 5.2700 |
| Final training loss | 4.1453 |
| Mean training loss | 4.4930 |
| Total time | 70.78 seconds |
| Mean step time | 2.83 seconds |

The run confirms that the disk-conscious ZIP data path and frozen-encoder
training configuration work on CPU. This short pilot shows a lower final loss,
but it is not enough to claim an improvement in caption quality; validation
captions must be generated from saved fine-tuned weights first.
