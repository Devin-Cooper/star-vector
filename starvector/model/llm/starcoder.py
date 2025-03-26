import torch.nn as nn
import os
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer,
    )

class StarCoderModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(StarCoderModel, self).__init__()

        self.init_tokenizer(config.starcoder_model_name)
        
        self.max_length = config.max_length
        model_config = AutoConfig.from_pretrained(config.starcoder_model_name, trust_remote_code=True)
        kwargs = {}
        kwargs['trust_remote_code'] = True
        kwargs['torch_dtype'] = config.torch_dtype

        # Configure special tokens for generation
        model_config.eos_token_id = self.tokenizer.eos_token_id
        model_config.pad_token_id = self.tokenizer.pad_token_id
        model_config.bos_token_id = self.tokenizer.bos_token_id
        
        # Handle flash attention safely for Mac (MPS) compatibility
        try:
            # Only enable flash attention if explicitly specified and not on Mac/MPS
            use_flash_attn = config.use_flash_attn
            if os.environ.get("STARVECTOR_DISABLE_FLASH_ATTN", "0") == "1":
                use_flash_attn = False
                print("FlashAttention disabled via environment variable for Mac compatibility")
                
            if use_flash_attn:
                # Try to import flash_attn to see if it's available
                import flash_attn
                model_config.flash_attention = True
                model_config._attn_implementation = "flash_attention_2"
                print("FlashAttention 2 enabled")
            else:
                model_config.flash_attention = False
                model_config._attn_implementation = None
                print("Using standard attention implementation")
        except ImportError:
            # Flash attention not available, fallback to standard attention
            config.use_flash_attn = False
            model_config.flash_attention = False
            model_config._attn_implementation = None
            print("FlashAttention not available, using standard attention")
        
        # model = GPTBigCodeForCausalLM(config=model_config)
        model = AutoModelForCausalLM.from_pretrained(config.starcoder_model_name, config=model_config, **kwargs)
        model.resize_token_embeddings(len(self.tokenizer))
        self.transformer = model

        # Prompt the model after image
        self.prompt = '<svg'
        
        # Setup svg start text 
        self.svg_start_token = '\n<svg'

    def init_tokenizer(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        # Incude padding and eos tokens in the vocabulary
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.add_special_tokens({"eos_token": "[EOS]"})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})       
        
        self.svg_start_token = "<svg-start>"
        self.image_start_token = "<image-start>"
        self.text_start_token = "<caption-start>"
        
        self.tokenizer.add_tokens([self.svg_start_token, self.image_start_token, self.text_start_token])
        self.svg_start_token_id = self.tokenizer.encode(self.svg_start_token)[0]
