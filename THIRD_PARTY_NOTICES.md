# Third-party notices

## MMLU-Pro dataset

This repository contains material from the MMLU-Pro benchmark, published by
TIGER-Lab on Hugging Face:

- Source: <https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro>
- Revision used: `b189ec765aa7ed75c8acfea42df31fdae71f97be`, downloaded on
  4 August 2026
- Licence: MIT, as declared on the dataset card (`license: mit`)
- Paper: Wang, Y. et al. (2024). *MMLU-Pro: A More Robust and Challenging
  Multi-Task Language Understanding Benchmark.* NeurIPS 2024 Datasets and
  Benchmarks Track.

The dataset card names no individual copyright holder, so the material is
credited to TIGER-Lab and the MMLU-Pro authors.

### Where it appears in this repository

- `data/raw/mmlu_pro/`: the original test and validation files, unchanged.
- `data/frozen/mmlu_pro_v1/`: 300 experimental and 20 pilot questions selected
  from the test split. The question text and options are unchanged; this
  project added stable question IDs and stores the answer keys in separate
  files.
- `data/public_viewer/accepted_experiment_v1.json`: six of those questions,
  shown in the public replay interface.

### MIT licence

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
