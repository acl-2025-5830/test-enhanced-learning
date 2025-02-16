import os 
import datasets

data_dir = os.path.join(os.getcwd(), "../dataset/")
K = 1024

domain2name = {
    "medicine": "pubmed",
    "finance": "bloomberg",
}

method2path = {
    "CPT": "anonymous4459/TEL_train_datasets",
    "IT": "anonymous4459/TEL_train_datasets",
    "CPT-IT": "anonymous4459/TEL_train_datasets",
    "TEL": "anonymous4459/TEL_train_datasets",
    "InstPT": "anonymous4459/InstPT_train_datasets",
}

def build_cpt(
    tokenizer=None,
    domain=None,
    method=None,
):
    
    tokenizer_name = tokenizer.name_or_path.split("/")[-1]
    save_path = os.path.join(data_dir, f"{domain2name[domain]}-cpt", tokenizer_name)

    def tokenize_dataset(examples):
        input_ids_list = []
        labels_list = []

        for text in examples['text']:
            input_ids = tokenizer(text)['input_ids'][:tokenizer.model_max_length]
            labels = input_ids.copy()
            assert len(input_ids) == len(labels)

            input_ids_list.append(input_ids)
            labels_list.append(labels)
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
        }
    try:
        ds = datasets.load_from_disk(os.path.join(save_path, "tokenized_data"))
    except:
        repo_id = method2path[method]
        print("Load dataset from: ", repo_id, "Domain: ", domain)
        ds = datasets.load_dataset(
            repo_id,
            domain,
            split="train",
        )

        ds = ds.map(
            tokenize_dataset,
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names,
            num_proc=64,
        )
        ds.save_to_disk(os.path.join(save_path, "tokenized_data"))
    
    ds.set_format("torch")
    return ds


def build_it(
    tokenizer=None,
    domain=None,
    method=None,
):
    tokenizer_name = tokenizer.name_or_path.split("/")[-1]
    save_path = os.path.join(data_dir, f"{domain2name[domain]}-it", tokenizer_name)

    def tokenize_dataset(examples):
        input_ids_list = []
        labels_list = []

        for q1, q2, q3, a1, a2, a3 in zip(
            examples['question1'],
            examples['question2'],
            examples['question3'],
            examples['answer1'],
            examples['answer2'],
            examples['answer3'],
        ):
            q1_ids = tokenizer("Question: " + q1 + "\nAnswer: ")['input_ids']
            q1_label = [-100 for _ in range(len(q1_ids))]

            q2_ids = tokenizer("Question: " + q2 + "\nAnswer: ", add_special_tokens=False)['input_ids']
            q2_label = [-100 for _ in range(len(q2_ids))]

            q3_ids = tokenizer("Question: " + q3 + "\nAnswer: ", add_special_tokens=False)['input_ids']
            q3_label = [-100 for _ in range(len(q3_ids))]

            a1_ids = a1_label = tokenizer(a1 + "\n\n", add_special_tokens=False)['input_ids']
            a2_ids = a2_label = tokenizer(a2 + "\n\n", add_special_tokens=False)['input_ids']
            a3_ids = a3_label = tokenizer(a3 + "\n\n", add_special_tokens=False)['input_ids']

            input_ids = q1_ids + a1_ids + q2_ids + a2_ids + q3_ids + a3_ids
            labels = q1_label + a1_label + q2_label + a2_label + q3_label + a3_label
            assert len(input_ids) == len(labels)

            input_ids_list.append(input_ids[:tokenizer.model_max_length])
            labels_list.append(labels[:tokenizer.model_max_length])
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
        }
    try:
        ds = datasets.load_from_disk(os.path.join(save_path, "tokenized_data"))
    except:
        repo_id = method2path[method]
        print("Load dataset from: ", repo_id, "Domain: ", domain)
        ds = datasets.load_dataset(
            repo_id,
            domain,
            split="train",
        )

        ds = ds.map(
            tokenize_dataset,
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names,
            num_proc=64,
        )
        ds.save_to_disk(os.path.join(save_path, "tokenized_data"))
    
    ds.set_format("torch")
    return ds



def build_TEL(
    tokenizer=None,
    domain=None,
    method=None,
):
    tokenizer_name = tokenizer.name_or_path.split("/")[-1]
    save_path = os.path.join(data_dir, f"{domain2name[domain]}-TEL-A", tokenizer_name)

    def tokenize_dataset(examples):
        input_ids_list = []
        labels_list = []

        for text, q1, q2, q3, a1, a2, a3 in zip(
            examples['text'],
            examples['question1'],
            examples['question2'],
            examples['question3'],
            examples['answer1'],
            examples['answer2'],
            examples['answer3'],
        ):
            text_ids = text_label = tokenizer(text + "\n")['input_ids']
            q1_ids = tokenizer("Question: " + q1 + "\nAnswer: ", add_special_tokens=False)['input_ids']
            q1_label = [-100 for _ in range(len(q1_ids))]

            q2_ids = tokenizer("Question: " + q2 + "\nAnswer: ", add_special_tokens=False)['input_ids']
            q2_label = [-100 for _ in range(len(q2_ids))]

            q3_ids = tokenizer("Question: " + q3 + "\nAnswer: ", add_special_tokens=False)['input_ids']
            q3_label = [-100 for _ in range(len(q3_ids))]

            a1_ids = a1_label = tokenizer(a1 + "\n\n", add_special_tokens=False)['input_ids']
            a2_ids = a2_label = tokenizer(a2 + "\n\n", add_special_tokens=False)['input_ids']
            a3_ids = a3_label = tokenizer(a3, add_special_tokens=False)['input_ids']

            input_ids = text_ids + q1_ids + a1_ids + q2_ids + a2_ids + q3_ids + a3_ids
            labels = text_label + q1_label + a1_label + q2_label + a2_label + q3_label + a3_label
            assert len(input_ids) == len(labels)

            input_ids_list.append(input_ids[:tokenizer.model_max_length])
            labels_list.append(labels[:tokenizer.model_max_length])
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
        }
    try:
        ds = datasets.load_from_disk(os.path.join(save_path, "tokenized_data"))
    except:
        repo_id = method2path[method]
        print("Load dataset from: ", repo_id, "Domain: ", domain)
        ds = datasets.load_dataset(
            repo_id,
            domain,
            split="train",
        )

        ds = ds.map(
            tokenize_dataset,
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names,
            num_proc=64,
        )

        ds = ds.filter(lambda x: len(x['input_ids']) <= 6100)
        ds.save_to_disk(os.path.join(save_path, "tokenized_data"))
    
    ds.set_format("torch")
    return ds



def build_INST_PT(
    tokenizer=None,
    domain=None,
    method=None,
):
    tokenizer_name = tokenizer.name_or_path.split("/")[-1]
    save_path = os.path.join(data_dir, f"{domain2name[domain]}-INST-PT", tokenizer_name)

    def tokenize_dataset(examples):
        input_ids_list = []
        labels_list = []

        for text, q_list, a_list in zip(
            examples['text'],
            examples['question'],
            examples['answer'],
        ):
            input_ids = tokenizer(text + "\n")['input_ids']

            for q, a in zip(q_list, a_list):
                q_ids = tokenizer("Question: " + q + "\nAnswer: ", add_special_tokens=False)['input_ids']
                a_ids = tokenizer(a + "\n\n", add_special_tokens=False)['input_ids']

                input_ids += q_ids + a_ids

            labels = input_ids
            assert len(input_ids) == len(labels)

            input_ids_list.append(input_ids[:tokenizer.model_max_length])
            labels_list.append(labels[:tokenizer.model_max_length])
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
        }
    try:
        ds = datasets.load_from_disk(os.path.join(save_path, "tokenized_data"))
    except:
        repo_id = method2path[method]
        print("Load dataset from: ", repo_id, "Domain: ", domain)
        ds = datasets.load_dataset(
            repo_id,
            domain,
            split="train",
        )

        ds = ds.map(
            tokenize_dataset,
            batched=True,
            batch_size=1000,
            remove_columns=ds.column_names,
            num_proc=64,
        )

        ds = ds.filter(lambda x: len(x['input_ids']) <= 6100)
        ds.save_to_disk(os.path.join(save_path, "tokenized_data"))
    
    ds.set_format("torch")
    return ds

