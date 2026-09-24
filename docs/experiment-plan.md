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
