from transformers import (
    AutoTokenizer,
    AutoModel,
)
import torch
from tqdm import tqdm
import os
from dotenv import load_dotenv
from loguru import logger
from trident2.logger.setup_logger import setup_logger


def extend_tokenizer_and_model(base_model, new_tokens, save_dir):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    logger.info(f"Tokenizer initial vocab size: {len(tokenizer)}")
    # Add these tokens to the tokenizer
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    logger.info(f"Added {num_added_tokens} new tokens.")
    logger.info(f"Tokenizer extended vocab size: {len(tokenizer)}")

    # Load the model corresponding to the tokenizer
    model = AutoModel.from_pretrained(base_model, trust_remote_code=True)
    logger.info(
        f"tokenizer vocab_size: {len(tokenizer)} \n model vocab_size: {model.embeddings.word_embeddings.num_embeddings}"
    )

    # Resize the model's token embeddings to account for the new tokens
    model.resize_token_embeddings(len(tokenizer))
    logger.info("Resizing embeddings...")

    # if they are not same size raise an error
    if len(tokenizer) == model.embeddings.word_embeddings.num_embeddings:
        logger.success(
            f"tokenizer vocab_size: {len(tokenizer)} \n model vocab_size: {model.embeddings.word_embeddings.num_embeddings}"
        )
    else:
        logger.error(
            f"tokenizer vocab_size: {len(tokenizer)} \n model vocab_size: {model.embeddings.word_embeddings.num_embeddings}"
        )
        raise ValueError("Tokenizer and model vocab sizes do not match after resizing.")

    # Verify model and tokenizer behave the same as pretrained models
    logger.info("Verifying model and tokenizer behave the same as pretrained models...")

    old_model = AutoModel.from_pretrained(base_model, trust_remote_code=True)
    old_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    for token in tqdm(list(old_tokenizer.vocab.keys())):
        encoding = old_tokenizer(token, return_tensors="pt")
        out1 = model(
            encoding["input_ids"], encoding["attention_mask"]
        ).last_hidden_state
        out2 = old_model(
            encoding["input_ids"], encoding["attention_mask"]
        ).last_hidden_state
        with torch.no_grad():
            if not torch.equal(out1, out2):
                logger.error(f"tensor {out1} \ndoes not equal \n{out2}")
                raise ValueError(
                    f"Old model and new model do not behave the same for token {token}"
                )

    model.save_pretrained(f"{save_dir}", from_pt=True)
    tokenizer.save_pretrained(f"{save_dir}")

    logger.success(f"Model and tokenizer saved to {save_dir}")
    return model, tokenizer


def upload_to_hf(model, tokenizer, repo_name, token):
    from huggingface_hub import HfApi

    logger.info(f"Pushing model and tokenizer to Hugging Face Hub at {repo_name}...")
    model.push_to_hub(repo_name, token=token, safe_serialization=True)
    tokenizer.push_to_hub(repo_name, token=token)

    # Also upload folder to catch any files that might not be uploaded by push_to_hub
    api = HfApi()
    api.upload_folder(
        folder_path="ChemBERTa-1",
        repo_id=repo_name,
        token=token,
        repo_type="model",
        commit_message="Upload extended model and tokenizer with new tokens",
    )

    logger.success(f"Model and tokenizer pushed to Hugging Face Hub at {repo_name}")


def main():
    setup_logger()
    # List of new tokens to add (feel free to add more or adjust as necessary)
    new_tokens = [
        # endpoints
        "<EC50>",
        "<EC10>",
        "<LOEC>",
        "<NOEC>",
        # effects
        "<MOR>",
        "<ITX>",
        "<DVP>",
        "<MPH>",
        "<POP>",
        "<GRO>",
        "<REP>",
        "<BEH>",
        "<CAR>",
        "<PHY>",
        # administration routes
        "<intratracheal>",
        "<intracerebral>",
        "<implant>",
        "<static>",
        "<spray>",
        "<renewal>",
        "<intraperitoneal>",
        "<oral>",
        "<environmental>",
        "<flow_through>",
        "<intravenous>",
        "<food>",
        "<gavage>",
        "<direct_application>",
        "<subcutaneous>",
        "<inhalation>",
        "<culture_media>",
        "<dermal>",
        "<topical>",
        "<granular>",
        "<soaking>",
        "<intramuscular>",
        "<in_vitro>",
        "<drinking>",
        "<parenteral>",
        "<injection>",
        # grouped_lifestages
        "<early>",
        "<mid>",
        "<adult>",
        # Missingness token
        "<missing_lifestage>",
        "<missing_administration_route>",
        "<missing_effect>",
    ]

    logger.info("Extending tokenizer and model with new tokens...")

    model, tokenizer = extend_tokenizer_and_model(
        "StyrbjornKall/ChemBERTa-1", new_tokens=new_tokens, save_dir="ChemBERTa-1"
    )

    logger.success(
        f"Tokenizer and model extended successfully with new tokens:\n{new_tokens}"
    )

    # Verify tokenizer works as expected
    logger.info(
        "Verifying tokenizer works as expected with input:\n'c1ccccc1</s><MOR><fish><early>'"
    )
    logger.success(tokenizer("c1ccccc1</s><MOR><fish><early>").tokens())
    logger.success(tokenizer("c1ccccc1</s><missing_effect><fish><early>").tokens())

    # Push to hugging face hub
    load_dotenv()
    upload_to_hf(
        model=model,  # Replace with the actual model variable if needed
        tokenizer=tokenizer,
        repo_name="StyrbjornKall/Multimodal-ChemBERTa-1",
        token=os.getenv("HF_PUSH_TOKEN"),
    )


if __name__ == "__main__":
    main()
