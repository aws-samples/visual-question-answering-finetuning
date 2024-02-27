# Visual Question Answering using LLMs
## Generating fashion product descriptionsby fine-tuning a Vision-Language Model with Amazon SageMaker and Amazon Bedrock


This repository implements a machine learning training and inference regiment, using GenAI to answer questions based on provided images. Pre-trained models exist to achive such tasks, however they are a) unable to adapt to domain-specific scenarios - hence why we need to fine tune and b) do not display the capability to be deployed into production environments.

To solve this problem, this post shows how to predict domain-specific product attributes from product images by fine-tuning a VLM (Vision Language Models) on a fashion dataset using Amazon SageMaker, and then use Amazon Bedrock to generate product descriptions using the predicted attributes as input.

## Data and Usecase

The data used in this repository is taken from [Kaggle Fashion Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset) and the usecase we try to solve is that of an e-commerce website generating captions for products, at task that has historically been very time consuming. High-quality product descriptions improve searchability through SEO, as well as improve customer satisfaction by allowing the customers to make informed decisions.

## VLM Model

The model used in this repo is the [BLIP-2 Model](https://arxiv.org/abs/2301.12597) and more specifically, a variant of it using [Flan-T5-XL](https://huggingface.co/Salesforce/blip2-flan-t5-xl). 

The following diagram illustrates the overview of BLIP-2:

![Blip-2 Diagram](./diagrams/blip-2.jpg)

## Solution Overview

The solution can be broken down in 2 sections, marked green and blue in the achitecture below: a) fine-tuning in green and b) inference in blue.

![Solution Overview](./diagrams/solution_overview.jpg)

### Fine-Tuning

1. The data is downloaded in an S3 bucket
2. A subset of the data is used to fine-tune a model, using a Sagemaker training job
3. The fine-tuned model artefacts are then stored in an S3 bucket to be used for inference

### Inference

1. The model artefacts are used to spin up a Sagemaker endpoint
2. The endpoint is then invoked with a question and returns a json-like response containing the relevant product attributes
3. The response is then passed to Amazon Bedrock alongside with a prompt (which is not visible or controllable by the end used), which formats the response before returning it.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

