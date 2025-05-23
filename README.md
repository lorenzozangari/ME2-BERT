# ME²-BERT - Moral Foundation Prediction through Events and Emotions

Source code of the framework presented in [*ME²-BERT: Are Events and Emotions what you need for Moral Foundation Prediction?*](https://github.com/lorenzozangari/ME2-BERT), presented at COLING 2025.

> Moralities, emotions, and events are complex aspects of human cognition, which are often treated separately since capturing their combined effects is challenging, especially due to the lack of annotated data. Leveraging their interrelations hence becomes crucial for advancing the understanding of human moral behaviors.
In this work, we propose ME²-BERT, the first holistic  framework for fine-tuning a pre-trained language model like BERT to the task of moral foundation prediction. ME²-BERT integrates events and emotions for learning domain-invariant morality-relevant text representations. 
Our extensive experiments show that ME²-BERT outperforms existing state-of-the-art methods for moral foundation prediction, with an average percentage increase up to 35% in the out-of-domain scenario.


**[HuggingFace](https://huggingface.co/lorenzozan/ME2-BERT) | [Paper](https://github.com/lorenzozangari/ME2-BERT) | [WebApp](https://huggingface.co/spaces/lorenzozan/ME2-BERT)**



**For an example of its usage, refer to the  [me2bert_example.ipynb](me2bert_example.ipynb) notebook.**

## Data
We trained ME²-BERT on the [E2MoCase](https://arxiv.org/pdf/2409.09001) dataset (available upon request). However, you can use any dataset that includes events and/or emotions to train our framework. The data you use must be in CSV format and must contain (at least) the following columns:

- `text`: Input text  
- `event`: List of events in JSON format  
- `care`, `harm`, `fairness`, `cheating`, `loyalty`, `betrayal`, `authority`, `subversion`, `purity`, `degradation`: Real-valued scores (within 0 and 1) associated with moral values  
- `anticipation`, `trust`, `disgust`, `joy`, `optimism`, `surprise`, `love`, `anger`, `sadness`, `pessimism`, `fear`: Real-valued scores (within 0 and 1) associated with emotion values

Note that it is necessary to have rows both with and without events to use our domain identification strategy. If you wish to adopt a different event-based domain identification strategy, modify the script *input_data/load_data.py*.

Below, is an example of a single row of the dataset:

**text**:  
```
"Mystery without an answer: Where is Meredith's murderer? 
Amanda Knox was acquitted of murdering Meredith Kercher. 
But if it wasn't her, then who killed the British woman with 43 stab wounds?"
```

**event**:

```
[
  {"mention": "murder", "entities": {"Amanda Knox": "murderer", "Meredith Kercher": "victim"}},
  {"mention": "kill", "entities": {"Amanda Knox": "murderer", "Meredith Kercher": "victim"}}
]
```

**Moral columns**:
| care | harm | fairness | cheating | loyalty | betrayal | authority | subversion | purity | degradation |
|-------------|-------------|-----------------|-----------------|----------------|-----------------|------------------|-------------------|---------------|--------------------|
| 0.0         | 0.985  | 0.0             | 0.901      | 0.0            | 0.910        | 0.0              | 0.0               | 0.0           | 0.221        |

**Emotion columns**:
| anticipation | trust | disgust | joy  | optimism | surprise | love | anger | sadness | pessimism | fear |
|--------------|-------|---------|------|----------|----------|------|-------|---------|-----------|------|
| 0.0          | 0.0   | 0.521   | 0.0  | 0.0      | 0.0      | 0.0  | 0.5   | 0.0     | 0.0       | 0.0  |


### Evaluation Data
The evaluation data we used in our paper can be found at the following links:

- [Moral Foundation Twitter Corpus (MFTC)](https://osf.io/k5n7y/)
- [Moral Foundation Reddit Corpus (MFRC)](https://huggingface.co/datasets/USC-MOLA-Lab/MFRC)
- [Extended Moral Foundation Dictionary (eMFD)](https://osf.io/vw85e/)
- [MoralEvents](https://github.com/launchnlp/MOKA) 


  
## Usage

To train the model, run the **train.py** script with the following command:
```
python train.py --data_path './data/e2mocase_full.csv' --transformation --contrastive --batch_size 8 --n_epoch 10 --device 0 --pretrained_model 'bert-base-uncased' --max_seq_len 256
```

which fine-tunes a BERT-based model (--pretrained_model) on the E2MoCase dataset (--data_path)  for 10 epochs (--epoch) and with batch size equal to 8 (--batch-size)- using the denoising auto-encoder as transformation function (--transformation) and the contrastive term (--contrastive).

You can check out all the parameters in the *utils/params.py* script.

### Input arguments


The following are the input arguments used in the script, along with their descriptions and default values:

- `--data_path` (default: `./data/e2mocase_full.csv`):  
  Path to the input data file.

- `--device` (default: `7`):  
  GPU device index to use for computation. Use `-1` for CPU.

- `--seed` (default: `72`):  
  Random seed for ensuring reproducibility across runs.

- `--pretrained_model` (default: `bert-base-uncased`):  
  Name of the pre-trained language model used for fine-tuning.

- `--max_seq_len` (default: `256`):  
  Maximum sequence length for tokenization. Longer sequences will be truncated if truncation is enabled.

- `--no_gate` (default: `False`):  
  Flag to disable the use of the gate component in the model.

- `--padding` (default: `"max_length"`):  
  Strategy for padding sequences. For example, use `"max_length"` for padding all sequences to the same length.

- `--no_truncation` (default: `False`):  
  Flag to disable truncation. If `False`, sequences exceeding `max_seq_len` will be truncated.


- `--mf_classes` (default: `5`):  
  Number of classes for moral foundations. Use `-1` for automatic detection.

- `--batch_size` (default: `8`):  
  Number of samples per batch during training and evaluation.

- `--n_epoch` (default: `10`):  
  Total number of training epochs.

- `--dropout` (default: `0.3`):  
  Dropout rate for regularization.

- `--lr` (default: `0.00005`):  
  Learning rate for the optimizer.

- `--lambda_con` (default: `1`):  
  Scaling factor for the contrastive loss term.

- `--alpha` (default: `10`):  
  Controls the rate of decay in the learning rate schedule.

- `--beta` (default: `0.25`):  
  Modulates the sharpness of the decay in the learning rate.

- `--gamma` (default: `10`):  
  Influences the scaling of domain adaptation and/or contrastive loss.

- `--lambda_trans` (default: `1.0`):  
  Scaling factor for the autoencoder loss.

- `--num_no_adv` (default: `5`):  
  Number of initial epochs without adversarial learning.

- `--num_epoch_save` (default: `5`):  
  Frequency (in epochs) at which the best model is saved.

- `--save_data` (default: `False`):  
  Flag to save the input data if it does not exist, or load it if it does.

- `--output_dir` (default: `./artifacts`):  
  Directory where performance scores and results are saved.

- `--checkpoint_dir` (default: `./checkpoint`):  
  Directory for storing checkpoint models.

- `--contrastive` (default: `False`):  
  Flag to enable contrastive learning.

- `--transformation` (default: `False`):  
  Flag to enable the use of an autoencoder for transformation tasks.

## Requirements
The required libraries are listed in the *requirements.txt*.

Models were trained using an Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz CUDA 11.8, GPU NVIDIA A30 environment.

## References

```
@inproceedings{zangari-etal-2025-me2,
    title = "{ME}2-{BERT}: Are Events and Emotions what you need for Moral Foundation Prediction?",
    author = "Zangari, Lorenzo  and
      Greco, Candida M.  and
      Picca, Davide  and
      Tagarelli, Andrea",
    booktitle = "Proceedings of the 31st International Conference on Computational Linguistics (COLING)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.coling-main.638/",
    pages = "9516--9532",
    year = "2025",
    abstract = "Moralities, emotions, and events are complex aspects of human cognition, which are often treated separately since capturing their combined effects is challenging, especially due to the lack of annotated data. Leveraging their interrelations hence becomes crucial for advancing the understanding of human moral behaviors. In this work, we propose ME2-BERT, the first holistic framework for fine-tuning a pre-trained language model like BERT to the task of moral foundation prediction. ME2-BERT integrates events and emotions for learning domain-invariant morality-relevant text representations. Our extensive experiments show that ME2-BERT outperforms existing state-of-the-art methods for moral foundation prediction, with an average increase up to 35{\%} in the out-of-domain scenario."
}
```
