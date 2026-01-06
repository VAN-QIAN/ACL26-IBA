# ACL26-Submission 70
This is the core code implication of our ACL Submission 70.
Our core prompt template has been outlined in Appendix E and corresponding implication can be found under path `model/prompt_templates.py`.

This file mainly organize as follows:
1. Dataset Preparation, including external Knowledge Base
2. Key commands to reproduce retrieval and generation.

You may also follow the instruction in EchoSight repo to set up the dataset, external knowledge base and initial image-to-image retrieval.

# Environments
```
pip install -f .requirements.txt
```
To run EchoSight, you may need to adjust transformers version to 4.37.2.

# Dataset
The total downloading process may take storage space more than 1TB. You may optimize the image downloading by removing zipped files in a timely manner.

Please also refer to the instruction of [EchoSight](https://github.com/Go2Heart/EchoSight/tree/main?tab=readme-ov-file#vqa-questions) for downloading images.

Key bullets are outlined as follows.
1. InfoSeek dataset can be obtained through their hugging face repo. You may need to unzip the tar.gz files to a target directory then the file `infoseek_val.jsonl` can map to correct image root path.

2. E-VQA images consist of two dataset Google Landmarks Dataset V2(denoted as GLD_image_path) and iNaturalist 2021 (denoted as iNat_image_path). You may check the scripts of original repos to download the images.
    - For GLD, you may use the modified script we provided under scripts folder. `bash /download_GLD.sh train 499` and `bash /download_GLD.sh test 19`
    - For iNaturalist 2021, also check and put id2name file `val_id2name.json` in the dataset folder which will be used for path mapping. [Link](https://drive.google.com/file/d/1cYzo4qewPABFuoMhpME4j2DWAA_Y-l2L/view?usp=drive_link) provided by EchoSight

# External Knowledge Base

Please check EchoSight's instructions for more details.
Basically you will have the FAISS index files and Wikipedia Data corpus (in json format) and you may identify their path in the following commands.

# Retrieval

Please reuse the code of [EchoSight](https://github.com/Go2Heart/EchoSight/tree/main?tab=readme-ov-file#script-details-1) to conduct the initial image-to-image retrieval.

Once the retrieval metadata is obtained, you may use our code to conduct the identification process illustrated in our paper.

# Metadata preparation

## Identify only one entity
1. Set up the environment variables
```
export TEST_FILE=$PWD/infoseek_test.csv
export RETRIEVAL_RESULTS=$PWD/$PATH-OF-EchoSight-TestReranker's-Output$
export KNOWLEDGE_BASE=$PWD/wiki_100_dict_v4.json
export OUTPUT_DIR=$PWD/InfoSeek-Metadata
mkdir -p "$OUTPUT_DIR"
```

2. adjust `--identification_top_k` to match k of initial retrieval
```

python -m src.run_pipeline prepare \
  --test_file "$TEST_FILE" \
  --retrieval_results "$RETRIEVAL_RESULTS" \
  --knowledge_base "$KNOWLEDGE_BASE" \
  --metadata_path "$OUTPUT_DIR/metadata.jsonl" \
  --context_mode section
  --qwen_model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --qwen_device cuda:0 \
  --identification_top_k 20
```

## Identify top3 possible entities and use the identification score to re-rank sections

```
python -m src.topk.run_top3_identification_pipeline prepare \
  --test_file "$TEST_FILE" \
  --retrieval_results "$RETRIEVAL_RESULTS" \
  --knowledge_base "$KNOWLEDGE_BASE" \
  --metadata_path "$OUTPUT_DIR/topk_metadata_bge.jsonl" \
  --entity_top_k 3 \
  --identification_score_top_k 3 \
  --identification_include_similarity \
  --section_reranker_backend bge \
  --section_reranker BAAI/bge-reranker-v2-m3 \
  --section_score_weight 1.0
```

# Generation
Use the metadata obtained from above identification.
```
python -m src.run_pipeline answer \
  --metadata_path "$OUTPUT_DIR/metadata.jsonl" \
  --output_path "$OUTPUT_DIR/answers.jsonl" \
  --knowledge_base "$KNOWLEDGE_BASE" \
  --qwen_model_name Qwen/Qwen2.5-VL-7B-Instruct \
  --qwen_device cuda:0
  --dataset_name $DATASET_NAME$

```

# Additional Instructions for reproducing baselines

## EchoSight

Adjust the path variables accordingly and you should be able to successfully execute their scripts to conduct retrieval with their pre-trained re-ranker.

## ReflectiVA

You may check the answer to their [issue](https://github.com/aimagelab/ReflectiVA/issues/8) to address the environment issues.
The reply also mentioned how to execute their code.
Please also note that the evaluation of ReflectiVA could be extremely long(more than 400 GPU hours with A40 GPU) hence you should consider parallel execution using the slurm scripts in original repo

# Acknowledgement

We would like to thank the authors of EchoSight and ReflectiVA for their released code and checkpoint.
Such open-sourced works largely eased our efforts to conduct real experiments for a fair comparison.

