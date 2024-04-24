# Imputation Strategies in Time Series based on Language Models
This project comprises the code for the Master's thesis 
"Imputation Strategies in Times Series based on Language Models", 
which was written for the final examination of the Master of Science 
in Informatics at the Hamburg University of Applied Sciences.

Short Summary of the Master's thesis:

The imputation of time series is an important task for downstream time series tasks and
therefore has a direct influence on the methods used. The advanced methods for imputation
use deep learning methods and transformer architectures, which are particularly
suitable for processing sequential data. The significant successes of the transformer architecture
within large language models in the field of natural language processing are to
be transferred to the field of time series analysis in this thesis. In particular, this thesis
analyses the suitability of large language models for the imputation of time series and
compares different open source models. Within a defined experimental setup, language
models with up to seven billion parameters are used, which compete with smaller language
models. To adapt the models, the attention layers of the language models are adapted
using PEFT methods. The results of the experiments show that smaller language models,
which were trained using a denoising autoencoding approach, can produce equally
good results as large models. This means that, in addition to the number of optimisable
parameters, the type of training is also decisive for the success of the imputation
performance. 

For more details see the master's thesis document (see contact information).

See the evaluation notebook for details on the experiments and results.

The code for investigating the aspects mentioned in the master's thesis includes various components. At the core 
of the implementation are the models, which can be found in the `models` folder.

## Installation and preparation
For the installation of the necessary libraries, the `requirements.txt` file is used. 
Here are the main libraries required for the execution of the code.

The data records can be found at the following addresses:
- ETT datasets: https://github.com/zhouhaoyi/ETDataset
- ECL datasets: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
- Weather datasets: https://www.bgc-jena.mpg.de/wetter/

For the execution of the experiments of the larger language models you have to obtain
the model weights from the Huggingface model hub. The models used in the experiments are:
- BART: https://huggingface.co/facebook/bart-base
- BERT: https://huggingface.co/google-bert/bert-base-uncased
- GPT-2: https://huggingface.co/openai-community/gpt2
- Llama2: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- Phi-2: https://huggingface.co/microsoft/phi-2

## Run experiments

To execute the different experiments, bash files are located in the `scripts/` folder. There is a separate folder for each model. The 
bash files for the advanced experiments regarding component fine-tuning and the comparison of
different PEFT methods are located in the `gpt_imp` folder, as these experiments have been conducted using the GPT model. 
An example of how to run an experiment is `bash scripts/gpt_imp/ETTh1.sh`. 
The results of the individual experiments are stored in the `results` folder.


## Contact
If you have any questions or suggestions, please feel free to contact at 
michel.jacobsen@haw-hamburg.de

## Acknowledgments
Many thanks to the following repositories, which were used for the implementation of the models and experiments:
- https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All
- https://github.com/thuml/Time-Series-Library
- https://github.com/Bjarten/early-stopping-pytorch


