
# Hibou: A Family of Foundational Vision Transformers for Pathology

[https://arxiv.org/abs/2406.05074](https://arxiv.org/abs/2406.05074)

## Updates

* **08.19.2024**: We release a CellViT-Hibou model, which is a hybrid model combining the CellViT and Hibou architectures. This model comes under CC BY-NC-SA 4.0 license. Check the `segmentation_example.ipynb` notebook for an example of how to use the model.
CellViT-Hibou is a model trained on PanNuke dataset for panoptic cell segmentation. It can segment and classify cells and tissues. For more information visit the original CellViT repository [here](https://github.com/TIO-IKIM/CellViT). Huge thanks to the authors of CellViT for their amazing work!

* **08.09.2024**: We are excited to announce the release of Hibou-L under the Apache 2.0 license. You can find Hibou-L on Hugging Face 🤗 [here](https://huggingface.co/histai/hibou-L).

## Introduction

This repository contains the code to run the Hibou-B model locally. For inquiries about accessing Hibou-L on CellDX, please contact us at [models@hist.ai](mailto:models@hist.ai).

## Getting Started

### Using HuggingFace

The easiest way to use the Hibou-B model is through the HuggingFace repository. Run the following code to get started:

```python
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("histai/hibou-b", trust_remote_code=True)
model = AutoModel.from_pretrained("histai/hibou-b", trust_remote_code=True)
```

We use a customized implementation of the DINOv2 architecture from the transformers library to add support for registers, which requires the `trust_remote_code=True` flag.

### Using the Model Directly

If you prefer to use the model without the transformers library, follow these steps:

1. **Install the requirements and the package:**

    ```bash
    git clone https://github.com/HistAI/hibou.git
    cd hibou
    pip install -r requirements.txt && pip install -e .
    ```

2. **Download the model weights:**

    [Hibou-B Weights](https://drive.google.com/file/d/12ICd_-yJWMYYo5OskMmc9SHJAQivAtS7/view?usp=sharing)

3. **Load the model with the following code:**

    ```python
    from hibou import build_model

    model = build_model("weights-path")
    ```

### Example Notebook

For more information, refer to the [example.ipynb](example.ipynb) notebook.

## Metrics
**Table: Linear probing benchmarks reporting top-1 accuracy.**

*Metrics for Virchow and RudolfV are derived from the respective papers, as these models are not open-sourced.*

| Dataset   | Phikon | Kaiko-B8 | Virchow* | RudolfV* | Prov-GigaPath | Hibou-B | Hibou-L |
|-----------|--------|----------|----------|----------|---------------|---------|---------|
| CRC-100K  | 0.917  | 0.949    | 0.968*   | **0.973*** | 0.968         | 0.955   | 0.966   |
| PCAM      | 0.916  | 0.919    | 0.933*   | 0.944*   | **0.947**     | 0.946   | 0.943   |
| MHIST     | 0.791  | 0.832    | 0.834*   | 0.821*   | 0.839         | 0.812   | **0.849** |
| MSI-CRC   | 0.750  | 0.786    | -        | 0.755*   | 0.771         | 0.779   | **0.797** |
| MSI-STAD  | 0.760  | 0.814    | -        | 0.788*   | 0.784         | 0.797   | **0.825** |
| TIL-DET   | 0.944  | **0.945** | -        | 0.943*   | 0.939         | 0.942   | 0.943   |
| **AVG (1-3)** | 0.875  | 0.900    | 0.912    | 0.913    | 0.918         | 0.904   | **0.919** |
| **AVG (1-6)** | 0.846  | 0.874    | -        | 0.871    | 0.875         | 0.872   | **0.887** |


## License

This repository is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full license text.

## Acknowledgements

We would like to thank the authors of the DINOv2 repository, upon which this project is built. The original repository can be found [here](https://github.com/facebookresearch/dinov2).

---

Feel free to reach out at [dmitry@hist.ai](mailto:dmitry@hist.ai) if you have any questions or need further assistance!

## Citation

If you use our work, please cite:

```
@misc{nechaev2024hibou,
    title={Hibou: A Family of Foundational Vision Transformers for Pathology},
    author={Dmitry Nechaev and Alexey Pchelnikov and Ekaterina Ivanova},
    year={2024},
    eprint={2406.05074},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```
