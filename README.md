# PEFT Fine-Tuning for Secure Code Generation

This repository contains the implementation and scripts for fine-tuning the `Salesforce/codegen-2B-multi` model using Parameter Efficient Fine-Tuning (PEFT) techniques, specifically **LoRA** (Low-Rank Adaptation) and **IA3** (Instance-Aware Adaptive Attention). The objective of the experiment is to evaluate the effectiveness of fine-tuned models in generating secure C/C++ code by identifying vulnerabilities aligned with the Common Weakness Enumeration (CWE) Top 25 list for 2023.

## Repository Structure


├── ia3_ft_1gpu_3epo_final.sh
├── lora_ft_1gpu_3epo_final.sh
├── requirements.txt
├── test_data.json
├── dataset_preparation/
│   └── prepare_data.py
└── training_insights_analysis/
    ├── IA3_finetune_21080491.log
    ├── LoRA_finetune_21080493.log
    ├── ia3_training_logs.log
    ├── insight_analysis_of_ft.ipynb
    ├── lora_training_logs.log
    └── trainable_parameter.py


## Prerequisites

To ensure a successful replication of the experiments, install the required dependencies using the `requirements.txt` file.

### Installation Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/peft-codegen-security.git
    cd peft-codegen-security
    ```

2. **Set up a virtual environment (recommended):**

    ```bash
    python3 -m venv peft_env
    source peft_env/bin/activate   # On Windows, use peft_env\Scripts\activate
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Experiments

### Step 1: Dataset Preparation

#### Steps to Prepare the Dataset

1. **Download the Original Dataset**  
   You can obtain the original dataset from [here](https://github.com/jp2425/SCoPE).

2. **Run the Preparation Script**  
   To process the dataset, navigate to the root folder of the repository and run the following command:

   ```bash
   python dataset_preparation/prepare_data.py
   ```
2. **Output**
    The processed dataset will be saved as 'test_data.json' in the same directory as the original dataset.



### Step 2: Fine-Tuning the Models

#### Running LoRA Fine-Tuning

Execute the following command to fine-tune the model using the LoRA technique:

    ```bash
    bash lora_ft_1gpu_3epo_final.sh
    ```

#### Running IA3 Fine-Tuning

Execute the following command to fine-tune the model using the IA3 technique:

    ```bash
    bash ia3_ft_1gpu_3epo_final.sh
    ```

### Both scripts perform the following tasks:

    - Load the base model (`Salesforce/codegen-2B-multi`).
    - Apply PEFT fine-tuning (LoRA/IA3).
    - Save checkpoints at regular intervals.


### Step 3: Generating Code from Fine-Tuned Models

Once the models are fine-tuned, generate code using:

    ```bash
    python LoRA_ft_script_final.py  # LoRA fine-tuned model
    python IA3_ft_script_final.py   # IA3 fine-tuned model
    ```

This process generates C/C++ code samples for vulnerability assessment.

### Step 4: Static Code Analysis

To analyze the generated code for security vulnerabilities, run the Flawfinder tool:

    ```bash
    flawfinder --html --out=report.html path/to/generated/code/
    ```

This will generate an HTML report summarizing potential vulnerabilities in the code.

## Contributing

Contributions are welcome! If you wish to contribute:

1. **Fork** the repository.
2. **Create a new branch** with your improvements.
3. **Submit a pull request** for review.

---

## License

This project is licensed under the **MIT License**.

---

## Contact

For further questions or support, contact:

- **E-mail:** [mdaha](mailto:mdaha@ntnu.no)
