"""
会议数据向量化工具模块
支持BGE-base-zh-v1.5和GPT text-embedding-3-small两种embedding方法
处理不同模型的token长度限制和文本切块
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import re
from tqdm import tqdm
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import time
import json
from datetime import datetime

# BGE embedding相关
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    import torch
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    torch = None
    print("Warning: sentence_transformers or transformers not installed. BGE embedding will not be available.")

# OpenAI embedding相关
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. GPT embedding will not be available.")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """会议数据向量化处理器"""
    
    def __init__(self, 
                 bge_model_name: str = "Qwen/Qwen3-Embedding-8B",
                 gpt_model_name: str = "text-embedding-3-small",
                 max_bge_tokens: int = 32000,
                 max_gpt_tokens: int = 8192,  # GPT embedding模型的token限制
                 chunk_overlap: int = 50,
                 load_bge: bool = True,
                 load_gpt: bool = False,
                 local_model_dir: str = None):
        """
        初始化向量化处理器
        
        Args:
            bge_model_name: BGE模型名称
            gpt_model_name: GPT embedding模型名称
            max_bge_tokens: BGE模型最大token数 (通常512)
            max_gpt_tokens: GPT模型最大token数 (text-embedding-3-small: 8192)
            chunk_overlap: 文本切块重叠数
            load_bge: 是否加载BGE模型
            load_gpt: 是否加载GPT客户端
            local_model_dir: 本地模型缓存目录，如果为None则使用当前工作目录下的model文件夹
        """
        self.bge_model_name = bge_model_name
        self.gpt_model_name = gpt_model_name
        self.max_bge_tokens = max_bge_tokens
        self.max_gpt_tokens = max_gpt_tokens
        self.chunk_overlap = chunk_overlap
        
        # 设置本地模型目录
        if local_model_dir is None:
            # 使用当前工作目录下的model文件夹
            self.local_model_dir = os.path.join(os.getcwd(), "model")
        else:
            self.local_model_dir = local_model_dir
        
        # 确保模型目录存在
        os.makedirs(self.local_model_dir, exist_ok=True)
        logger.info(f"Using local model directory: {self.local_model_dir}")
        

        # 初始化BGE模型
        self.bge_model = None
        self.bge_tokenizer = None
        if load_bge and BGE_AVAILABLE:
            self._init_bge_model()
        
        # 初始化OpenAI客户端
        self.openai_client = None
        if load_gpt and OPENAI_AVAILABLE:
            self._init_openai_client()
    
    def _init_bge_model(self):
        """初始化BGE模型和分词器，默认使用GPU，模型缓存到本地目录"""
        try:
            import torch
            
            # 检查GPU可用性
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading BGE model: {self.bge_model_name}")
            logger.info(f"Using device: {device}")
            logger.info(f"Model cache directory: {self.local_model_dir}")
            
            # 设置环境变量，让transformers使用本地缓存目录
            os.environ['TRANSFORMERS_CACHE'] = self.local_model_dir
            os.environ['HF_HOME'] = self.local_model_dir
            
            # 检查是否是本地模型路径
            local_model_path = os.path.join(self.local_model_dir, "Qwen3-Embedding-8B")
            if os.path.exists(local_model_path):
                # 使用本地模型
                logger.info(f"Loading local Qwen3-Embedding-8B model from: {local_model_path}")
                # 使用标准配置，不使用FlashAttention2
                logger.info("Using standard attention implementation")
                
                self.bge_model = SentenceTransformer(
                    local_model_path,
                    trust_remote_code=True,
                    device=device,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,  # A100 推荐 bfloat16
                        "attn_implementation": "eager"  # 使用标准注意力实现
                    }
                )

                self.bge_tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path,
                    trust_remote_code=True
                )
            else:
                # 使用在线模型，按原来的方式加载
                # 添加trust_remote_code=True和use_safetensors=True来处理模型格式问题
                logger.info(f"未能从本地读取到模型，从在线读取Qwen3-Embedding-8B model from: {self.bge_model_name}")
                # 使用标准配置，不使用FlashAttention2
                logger.info("Using standard attention implementation for online model")
                
                self.bge_model = SentenceTransformer(
                    self.bge_model_name, 
                    device=device,
                    cache_folder=self.local_model_dir,
                    trust_remote_code=True,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,  # A100 推荐 bfloat16
                        "attn_implementation": "eager"  # 使用标准注意力实现
                    }
                )
                self.bge_tokenizer = AutoTokenizer.from_pretrained(
                    self.bge_model_name,
                    cache_dir=self.local_model_dir,
                    trust_remote_code=True
                )
            
            if device == "cuda":
                logger.info(f"BGE model loaded successfully on GPU: {torch.cuda.get_device_name()}")
            else:
                logger.info("BGE model loaded successfully on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            # 尝试使用不同的方法加载模型

    
    def _init_openai_client(self):
        """初始化OpenAI客户端"""
        try:
            # 加载环境变量
            load_dotenv("oai_embeddings.env", override=True)
            api_key = os.getenv("api_key")
            base_url = os.getenv("base_url")
            
            if not api_key:
                logger.error("OpenAI API key not found in environment variables")
                return
            
            self.openai_client = OpenAI(
                api_key=api_key,
                base_url=base_url if base_url else "https://api.openai.com/v1"
            )
            logger.info(f"OpenAI client initialized successfully for model: {self.gpt_model_name}")
            logger.info(f"GPT token limit: {self.max_gpt_tokens}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
    
    def count_tokens(self, text: str, model_type: str = "bge") -> int:
        """
        计算文本的token数量
        
        Args:
            text: 输入文本
            model_type: 模型类型 ("bge" 或 "gpt")
            
        Returns:
            估算的token数量
        """
        if not text:
            return 0
            
        if model_type == "bge" and self.bge_tokenizer:
            return len(self.bge_tokenizer.encode(text, add_special_tokens=True))
        else:
            # 使用简单估算：中文约1字符=1token，英文约4字符=1token
            # GPT模型的tokenizer更复杂，但这里使用保守估算
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            other_chars = len(text) - chinese_chars - english_chars
            
            # 保守估算，稍微高估以确保不超过限制
            estimated_tokens = chinese_chars * 1.2 + english_chars * 0.3 + other_chars * 0.5
            return int(estimated_tokens)
    
    def split_text_by_tokens(self, text: str, max_tokens: int, model_type: str = "bge") -> List[str]:
        """
        按token数切分文本
        
        Args:
            text: 要切分的文本
            max_tokens: 每个chunk的最大token数
            model_type: 模型类型，用于选择合适的tokenizer
            
        Returns:
            切分后的文本块列表
        """
        if not text or not text.strip():
            return []
        
        # 如果文本长度在限制内，直接返回
        if self.count_tokens(text, model_type) <= max_tokens:
            return [text]
        
        # 按句子切分
        sentences = re.split(r'[。！？；\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence, model_type)
            
            # 如果单个句子就超过限制，强制切分
            if sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_tokens = 0
                
                # 强制按字符切分超长句子
                chars_per_token = len(sentence) / max(sentence_tokens, 1)
                max_chars = int(max_tokens * chars_per_token * 0.8)  # 留更多余量
                
                for i in range(0, len(sentence), max_chars):
                    chunk = sentence[i:i + max_chars]
                    if chunk.strip():
                        chunks.append(chunk)
                continue
            
            # 检查添加这个句子是否会超过限制
            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens
        
        # 添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def get_bge_embeddings_batch(self, texts: List[str], batch_size: int = 256, skip_length_check: bool = True, 
                                output_path: str = None, checkpoint_interval: int = 10000) -> List[Optional[Union[np.ndarray, List[np.ndarray]]]]:
        """
        批量获取BGE向量，支持断点续传的统一处理方法
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小（1=逐个处理，>1=批处理）
            skip_length_check: 是否跳过长度检查
            output_path: 输出路径，用于断点续传
            checkpoint_interval: 每处理多少个文本保存一次检查点（默认10000）
            
        Returns:
            向量列表，每个元素可能是单个向量或向量列表（超长文本的切片向量）
        """
        if not self.bge_model:
            return [None] * len(texts)
        
        # 🔄 统一的断点续传处理
        return self._process_with_checkpoint(texts, batch_size, output_path, checkpoint_interval, skip_length_check)
    
    def _process_with_checkpoint(self, texts: List[str], batch_size: int = 256, output_path: str = None, 
                               checkpoint_interval: int = 10000, skip_length_check: bool = True) -> List[Optional[np.ndarray]]:
        """
        统一的带断点续传的处理方法（简化版）
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小（1=逐个处理，>1=批处理）
            output_path: 输出路径，用于断点续传
            checkpoint_interval: 每处理多少个文本保存一次检查点（默认10000）
            skip_length_check: 是否跳过长度检查
            
        Returns:
            向量列表
        """
        total_count = len(texts)
        
        # 🔄 加载断点
        start_index = 0
        if output_path:
            start_index, metadata = self._load_checkpoint(output_path)
            if start_index > 0:
                logger.info(f"🔄 Resuming from checkpoint: {start_index:,}/{total_count:,}")
                logger.info(f"🚀 Skipping loading of {start_index:,} previous results (already saved in embeddings folder)")
                logger.info(f"⚙️  Memory optimization: Only processing remaining {total_count - start_index:,} texts")
                logger.info(f"💾 Previous results are safely stored in part-xxxxx.parquet files")
        
        # 🎯 只为剩余未处理的数据创建数组
        remaining_count = total_count - start_index
        results = [None] * remaining_count
        logger.info(f"💾 Allocated memory for {remaining_count:,} remaining texts (instead of {total_count:,})")
        
        # 🚀 统一处理逻辑
        if batch_size <= 1:
            logger.info(f"🐌 Individual processing mode (batch_size={batch_size})")
        else:
            logger.info(f"🚀 Batch processing mode (batch_size={batch_size})")
        
        # 大数据集提醒
        remaining = total_count - start_index
        if remaining > 100000:
            logger.info(f"💡 Large dataset: {remaining:,} texts remaining, checkpoint every {checkpoint_interval:,} texts")
        
        try:
            current_index = start_index
            
            while current_index < total_count:
                # 确定当前batch的范围
                if batch_size <= 1:
                    # Individual processing
                    batch_end = current_index + 1
                    batch_texts = [texts[current_index]] if texts[current_index] else [""]
                else:
                    # Batch processing
                    batch_end = min(current_index + batch_size, total_count)
                    batch_texts = texts[current_index:batch_end]
                
                try:
                    # 处理当前batch（无论是1个还是多个）
                    if batch_size <= 1:
                        # Individual processing：不显示进度条
                        batch_embeddings = self.bge_model.encode(
                            batch_texts[0] if batch_texts else "",
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                        results[current_index - start_index] = batch_embeddings
                    else:
                        # Batch processing：显示进度条
                        batch_embeddings = self.bge_model.encode(
                            batch_texts,
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            batch_size=batch_size
                        )
                        # 保存批处理结果
                        for i, embedding in enumerate(batch_embeddings):
                            results[current_index + i - start_index] = embedding
                    
                    current_index = batch_end
                    
                    # 💾 每checkpoint_interval个文本保存一次检查点
                    if output_path and current_index % checkpoint_interval == 0:
                        self._save_checkpoint(output_path, current_index, total_count, results, start_index)
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(f"💾 Checkpoint: {current_index:,}/{total_count:,} ({current_index/total_count*100:.1f}%)")
                    
                    # 显示进度（每1000个显示一次）
                    elif current_index % 1000 == 0:
                        progress = current_index / total_count * 100
                        logger.info(f"📊 Progress: {current_index:,}/{total_count:,} ({progress:.1f}%)")
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and batch_size > 1:
                        # 批处理OOM，降级到individual processing
                        logger.error(f"💥 CUDA OOM! Switching to individual processing from index {current_index}")
                        batch_size = 1  # 降级到individual
                        if torch is not None and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                
                except Exception as e:
                    logger.error(f"Error processing at index {current_index}: {e}")
                    results[current_index - start_index] = None
                    current_index += 1
            
            # 🎉 处理完成，清理检查点
            if output_path:
                self._cleanup_checkpoint(output_path)
            
            logger.info(f"🎉 Processing completed: {total_count:,} texts!")
            return results
            
        except KeyboardInterrupt:
            # 用户中断，不保存避免文件损坏
            logger.warning("⚠️  Processing interrupted! No checkpoint saved to avoid file corruption.")
            logger.info(f"💡 Progress: {current_index:,}/{total_count:,}. Resume will start from last checkpoint.")
            raise
        
        except Exception as e:
            # 其他错误，不保存避免文件损坏
            logger.error(f"Error in processing: {e}")
            logger.info(f"💡 Progress: {current_index:,}/{total_count:,}. Resume will start from last checkpoint.")
        return results
    
    def _get_bge_embedding_single(self, text: str) -> Optional[Union[np.ndarray, List[np.ndarray]]]:
        """
        获取单个文本的BGE向量（处理超长文本切块）
        
        Args:
            text: 输入文本
            
        Returns:
            向量数组或向量列表：
            - 如果文本在token限制内，返回单个向量数组
            - 如果文本超出限制被切分，返回切片向量列表
            - 如果文本为空或模型未加载则返回None
        """
        if not self.bge_model or not text or not text.strip():
            return None
        
        try:
            # 检查token长度并切分
            chunks = self.split_text_by_tokens(text, self.max_bge_tokens, "bge")
            
            if not chunks:
                return None
            
            if len(chunks) == 1:
                # 单个chunk，直接处理
                embedding = self.bge_model.encode(
                    chunks[0], 
                    normalize_embeddings=True,
                    show_progress_bar=False  # 禁用进度条
                )
                return embedding
            else:
                # 多个chunks，批量处理所有切片
                chunk_embeddings = self.bge_model.encode(
                    chunks,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=len(chunks)
                )
                return list(chunk_embeddings)  # 返回列表
                
        except Exception as e:
            logger.error(f"Error getting BGE embedding: {e}")
            return None
    
    def get_gpt_embeddings_batch(self, texts: List[str]) -> List[Optional[Union[np.ndarray, List[np.ndarray]]]]:
        """
        批量获取GPT向量，支持高效的批量API调用
        
        遵循OpenAI批量API约束：
        - 单次最多2048条文本
        - 总token数不超过30万tokens
        - 超长文本会被切块，保留所有切片向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表，每个元素可能是单个向量或向量列表（超长文本的切片向量）
        """
        if not self.openai_client:
            return [None] * len(texts)
        
        results = [None] * len(texts)  # 初始化结果列表，保持原始顺序
        
        # 预处理：分离需要切块的长文本和可直接处理的短文本
        long_text_indices = []
        short_text_data = []  # (original_index, text)
        
        for i, text in enumerate(texts):
            if not text or not text.strip():
                results[i] = None
                continue
                
            token_count = self.count_tokens(text, "gpt")
            if token_count > self.max_gpt_tokens:
                # 超长文本，需要单独处理
                long_text_indices.append(i)
            else:
                # 短文本，可以批量处理
                short_text_data.append((i, text))
        # 1. 先处理超长文本（单独处理，切块并保留所有向量）
        if long_text_indices:
            logger.info(f"Processing {len(long_text_indices)} long texts individually...")
            for idx in tqdm(long_text_indices, desc="Processing long texts"):
                text = texts[idx]
                chunks = self.split_text_by_tokens(text, self.max_gpt_tokens, "gpt")
                
                try:
                    chunk_embeddings = []
                    for chunk in chunks:
                        response = self.openai_client.embeddings.create(
                            input=chunk,
                            model=self.gpt_model_name
                        )
                        chunk_embeddings.append(np.array(response.data[0].embedding))
                        time.sleep(0.05)  # 避免请求过快
                    
                    # 保留所有切片向量，不计算平均
                    results[idx] = chunk_embeddings
                        
                except Exception as e:
                    logger.error(f"Error processing long text at index {idx}: {e}")
                    results[idx] = None
        
        # 2. 批量处理短文本
        if short_text_data:
            logger.info(f"Processing {len(short_text_data)} short texts in batches...")
            
            # 按照OpenAI约束智能分批
            batches = self._create_smart_batches(short_text_data)
            
            for batch_data in tqdm(batches, desc="Processing text batches"):
                try:
                    # 提取文本用于API调用
                    batch_texts = [data[1] for data in batch_data]
                    
                    # 批量API调用
                    response = self.openai_client.embeddings.create(
                        input=batch_texts,
                        model=self.gpt_model_name
                    )
                    
                    # 将结果映射回原始位置
                    for i, (original_idx, _) in enumerate(batch_data):
                        results[original_idx] = np.array(response.data[i].embedding)
                    
                    time.sleep(0.1)  # 避免请求过快
                    
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    # 如果批量失败，回退到单条处理
                    for original_idx, text in batch_data:
                        try:
                            response = self.openai_client.embeddings.create(
                                input=text,
                                model=self.gpt_model_name
                            )
                            results[original_idx] = np.array(response.data[0].embedding)
                            time.sleep(0.05)
                        except Exception as e2:
                            logger.error(f"Error processing text at index {original_idx}: {e2}")
                            results[original_idx] = None
        
        return results
    
    def _create_smart_batches(self, text_data: List[tuple], max_batch_size: int = 32000, max_tokens: int = 300000) -> List[List[tuple]]:
        """
        智能创建批次，遵循OpenAI约束
        
        Args:
            text_data: (original_index, text) 元组列表
            max_batch_size: 单批次最大文本数量
            max_tokens: 单批次最大token数
            
        Returns:
            批次列表，每个批次是 (original_index, text) 元组列表
        """
        batches = []
        current_batch = []
        current_tokens = 0
        
        for original_idx, text in text_data:
            text_tokens = self.count_tokens(text, "gpt")
            
            # 检查是否需要开始新批次
            if (len(current_batch) >= max_batch_size or 
                current_tokens + text_tokens > max_tokens) and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append((original_idx, text))
            current_tokens += text_tokens
        
        # 添加最后一个批次
        if current_batch:
            batches.append(current_batch)
        
        logger.info(f"Created {len(batches)} smart batches for {len(text_data)} texts")
        return batches
    
    def process_dataframe(self, 
                         df: pd.DataFrame, 
                         text_column: str,
                         use_bge: bool = True,
                         use_gpt: bool = False,
                         max_bge_tokens: int = 32000,
                         max_gpt_tokens: int = 128000,
                         batch_size: int = 256,
                         skip_length_check: bool = True,
                         output_path: str = None,
                         checkpoint_interval: int = 10000) -> pd.DataFrame:
        """
        处理DataFrame，添加向量化列
        
        Args:
            df: 输入DataFrame
            text_column: 文本列名
            use_bge: 是否使用BGE向量化
            use_gpt: 是否使用GPT向量化
            max_bge_tokens: BGE向量化的最大token数
            max_gpt_tokens: GPT向量化的最大token数
            batch_size: 批处理大小，40GB显存推荐256-512
            skip_length_check: 是否跳过长度检查，直接使用简单批处理（当你确定没有超长文本时）
        Returns:
            添加了向量列的DataFrame
        """
        result_df = df.copy()
        
        if text_column not in df.columns:
            logger.error(f"Column '{text_column}' not found in DataFrame")
            return result_df
        
        texts = df[text_column].fillna('').astype(str).tolist()
        
        # 分析文本长度分布（除非跳过长度检查）
        if not skip_length_check:
            logger.info("Analyzing text length distribution...")
            bge_long_texts = 0
            gpt_long_texts = 0
        
            # 检查文本长度是否超过最大token数
            for text in texts:
                bge_tokens = self.count_tokens(text, "bge")
                gpt_tokens = self.count_tokens(text, "gpt")
                
                if bge_tokens > self.max_bge_tokens:
                    bge_long_texts += 1
                if gpt_tokens > self.max_gpt_tokens:
                    gpt_long_texts += 1
                
            logger.info(f"BGE: {bge_long_texts}/{len(texts)} texts exceed {self.max_bge_tokens} tokens")
            logger.info(f"GPT: {gpt_long_texts}/{len(texts)} texts exceed {self.max_gpt_tokens} tokens")
        else:
            logger.info("⚡ Skipping text length analysis as requested!")
        
        # BGE向量化 - 使用高效批处理！
        if use_bge and self.bge_model:
            logger.info(f"Processing BGE embeddings with batch processing (batch_size={batch_size})...")
            bge_embeddings_raw = self.get_bge_embeddings_batch(
                texts, 
                batch_size=batch_size, 
                skip_length_check=skip_length_check,
                output_path=output_path,
                checkpoint_interval=checkpoint_interval
            )
            
            # 转换格式，
            bge_embeddings = []
            for embedding in bge_embeddings_raw:
                if embedding is not None:
                    if isinstance(embedding, list):
                        # 多个切片
                        bge_embeddings.append([emb.tolist() for emb in embedding])
                    else:
                        # 单个向量
                        bge_embeddings.append(embedding.tolist())
                else:
                    bge_embeddings.append(None)


            # 这里如果是中间断开过，然后继续处理，那么需要把之前的处理结果也加上，否则会报错
            result_df['bge_embedding'] = bge_embeddings  

        # GPT向量化（使用智能批量API调用）
        if use_gpt and self.openai_client:
            logger.info("Processing GPT embeddings with smart batching...")
            gpt_embeddings = self.get_gpt_embeddings_batch(texts)
            
            processed_gpt_embeddings = []
            for emb in gpt_embeddings:
                if emb is not None:
                    # 检查是否是向量列表（切片结果）
                    if isinstance(emb, list):
                        # 保存为列表格式，标记为多个切片
                        processed_gpt_embeddings.append([e.tolist() for e in emb])
                    else:
                        # 单个向量
                        processed_gpt_embeddings.append(emb.tolist())
                else:
                    processed_gpt_embeddings.append(None)
            
            result_df['gpt_embedding'] = processed_gpt_embeddings
        
        return result_df
    
    def save_embeddings(self, df: pd.DataFrame, output_path: str):
        """
        保存向量化结果
        
        Args:
            df: 包含向量的DataFrame
            output_path: 输出路径
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            if output_path.endswith('.xlsx'):
                df.to_excel(output_path, index=False)
            elif output_path.endswith('.csv'):
                df.to_csv(output_path, index=False)
            elif output_path.endswith('.parquet'):
                df.to_parquet(output_path, index=False)
            else:
                # 默认保存为Excel
                df.to_excel(output_path + '.xlsx', index=False)
            
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _get_checkpoint_path(self, output_path: str) -> str:
        """获取检查点文件路径"""
        base_path = os.path.splitext(output_path)[0]
        return f"{base_path}_checkpoint.json"
    
    def _save_checkpoint(self, output_path: str, processed_count: int, total_count: int, results: List, start_index: int = 0):
        """
        保存检查点（简化版）
        
        Args:
            output_path: 输出文件路径
            processed_count: 已处理数量
            total_count: 总数量
            results: 当前结果列表
        """
        try:
            checkpoint_path = self._get_checkpoint_path(output_path)
            
            # 保存检查点信息
            checkpoint_data = {
                'timestamp': datetime.now().isoformat(),
                'processed_count': processed_count,
                'total_count': total_count,
                'progress_percent': (processed_count / total_count * 100) if total_count > 0 else 0
            }
            
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # 🚀 分片保存：每1万条保存一个parquet文件
            if results and processed_count > 0:
                # 确定embeddings文件夹路径
                embeddings_dir = os.path.join(os.path.dirname(output_path), 'embeddings')
                os.makedirs(embeddings_dir, exist_ok=True)
                
                # 计算当前应该保存的chunk
                chunk_size = 10000
                current_chunk = (processed_count - 1) // chunk_size
                
                # 只保存当前chunk的数据
                abs_start_idx = current_chunk * chunk_size
                abs_end_idx = min(processed_count, (current_chunk + 1) * chunk_size)
                
                # 转换为results数组中的相对索引
                rel_start_idx = abs_start_idx - start_index
                rel_end_idx = abs_end_idx - start_index
                
                # 检查是否需要保存新的chunk
                chunk_file = os.path.join(embeddings_dir, f"part-{current_chunk:05d}.parquet")
                
                if (rel_start_idx >= 0 and rel_start_idx < len(results) and 
                    (processed_count % chunk_size == 0 or processed_count == total_count)):
                    chunk_results = results[rel_start_idx:rel_end_idx]
                    if chunk_results:
                        chunk_df = pd.DataFrame({
                            'embedding_result': chunk_results
                        })
                        
                        # 使用临时文件避免损坏
                        temp_chunk = chunk_file + ".tmp"
                        chunk_df.to_parquet(temp_chunk, index=False)
                        
                        # 原子性重命名
                        os.replace(temp_chunk, chunk_file)
                        
                        logger.info(f"💾 Saved part-{current_chunk:05d}.parquet: {len(chunk_results)} embeddings")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _load_checkpoint(self, output_path: str) -> Tuple[int, Dict]:
        """
        加载检查点（简化版）
        
        Returns:
            (已处理数量, 元数据)
        """
        try:
            checkpoint_path = self._get_checkpoint_path(output_path)
            
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                
                processed_count = checkpoint_data.get('processed_count', 0)
                
                # 检查embeddings文件夹中已有的分片文件
                embeddings_dir = os.path.join(os.path.dirname(output_path), 'embeddings')
                if os.path.exists(embeddings_dir):
                    part_files = [f for f in os.listdir(embeddings_dir) if f.startswith('part-') and f.endswith('.parquet')]
                    if part_files:
                        # 获取最大的part编号
                        max_part = max([int(f.split('-')[1].split('.')[0]) for f in part_files])
                        total_from_parts = (max_part + 1) * 10000  # 假设每个part有10000条
                        
                        logger.info(f"📂 Found {len(part_files)} existing part files (part-00000 to part-{max_part:05d})")
                        logger.info(f"📊 Estimated processed from parts: {total_from_parts:,}")
                        
                        # 如果part文件的数量与checkpoint不一致，以part文件为准
                        if abs(processed_count - total_from_parts) > 10000:
                            logger.warning(f"⚠️  Checkpoint count ({processed_count:,}) differs from part files ({total_from_parts:,})")
                            logger.info(f"📝 Using part files count: {total_from_parts:,}")
                            processed_count = total_from_parts
                
                logger.info(f"📂 Found checkpoint: {processed_count:,} items processed ({checkpoint_data.get('progress_percent', 0):.1f}%)")
                
                return processed_count, checkpoint_data
            
            return 0, {}
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return 0, {}
    
    def _cleanup_checkpoint(self, output_path: str):
        """清理检查点文件"""
        try:
            checkpoint_path = self._get_checkpoint_path(output_path)
            temp_output = f"{os.path.splitext(output_path)[0]}_temp.parquet"
            
            for file_path in [checkpoint_path, temp_output]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            logger.info("🗑️  Checkpoint files cleaned up (embeddings folder preserved)")
                
        except Exception as e:
            logger.error(f"Error cleaning up checkpoint: {e}")


def load_meeting_data(file_path: str, sheet_name: str = None) -> pd.DataFrame:
    """
    加载会议数据
    
    Args:
        file_path: 文件路径
        sheet_name: Excel工作表名称
        
    Returns:
        DataFrame
    """
    try:
        if file_path.endswith('.xlsx'):
            return pd.read_excel(file_path, sheet_name=sheet_name)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


# 便捷函数
def quick_embedding_pipeline(
    data_path: str,
    output_path: str,
    text_column: str,
    max_bge_tokens: int,
    sheet_name: str = None,
    use_bge: bool = True,
    use_gpt: bool = False,
    local_model_dir: str = None,
    batch_size: int = 64,  # 批处理大小，1=逐个处理
    checkpoint_interval: int = 10000,  # 每10000个文本保存一次检查点
    auto_resume: bool = True,
    skip_length_check: bool = True  # 自动从断点恢复
) -> pd.DataFrame:
    """
    快速向量化流水线，使用高效批量处理
    
    Args:
        data_path: 数据文件路径
        output_path: 输出文件路径
        text_column: 文本列名
        max_bge_tokens: BGE最大token数
        sheet_name: Excel工作表名称
        use_bge: 是否使用BGE
        use_gpt: 是否使用GPT
        local_model_dir: 本地模型缓存目录
        batch_size: 批处理大小，40GB显存推荐256-512
        skip_length_check: 是否跳过长度检查，直接使用简单批处理（当你确定没有超长文本时）
    Returns:
        处理后的DataFrame
    """
    # 加载数据
    df = load_meeting_data(data_path, sheet_name)
    if df.empty:
        logger.error("Failed to load data")
        return df
    
    # 初始化处理器
    processor = EmbeddingProcessor(load_bge=use_bge, load_gpt=use_gpt, local_model_dir=local_model_dir, max_bge_tokens=max_bge_tokens)
    
    # 处理向量化（支持断点续传的高效批量处理）
    result_df = processor.process_dataframe(
        df, 
        text_column=text_column,
        use_bge=use_bge,
        use_gpt=use_gpt,
        max_bge_tokens=max_bge_tokens,
        batch_size=batch_size,
        skip_length_check=skip_length_check,  # 简化：默认跳过长度检查
        output_path=output_path,
        checkpoint_interval=checkpoint_interval
    )
    
    # 🎯 不保存合并文件，只保留embeddings文件夹中的分片文件
    logger.info(f"✅ Processing completed! Results saved in embeddings folder as part-xxxxx.parquet files")
    embeddings_dir = os.path.join(os.path.dirname(output_path), 'embeddings')
    if os.path.exists(embeddings_dir):
        part_files = [f for f in os.listdir(embeddings_dir) if f.startswith('part-') and f.endswith('.parquet')]
        logger.info(f"📁 Total part files: {len(part_files)} in {embeddings_dir}")
    else:
        logger.warning("⚠️  No embeddings folder found")
    
    return result_df


if __name__ == "__main__":
    # 示例用法
    data_path = "/home/wangyu/project/data/company_info.parquet"
    output_path = "/home/wangyu/project/data/company_info_embeddings.parquet"
    
    result = quick_embedding_pipeline(
        data_path=data_path,
        output_path=output_path,
        text_column='comment',
        max_bge_tokens=32000,
        # sheet_name='message',
        batch_size=64,  # 批处理大小，1=逐个处理
        checkpoint_interval=10000,  # 每10000个文本保存检查点
        auto_resume=True,  # 自动从断点恢复
        use_bge=True,
        use_gpt=False,
        skip_length_check=True
    )
    
    print(f"Processed {len(result)} messages")
    
    # 处理BGE embedding维度统计
    if 'bge_embedding' in result.columns:
        bge_valid = result['bge_embedding'].dropna()
        if len(bge_valid) > 0:
            first_bge = bge_valid.iloc[0]
            if isinstance(first_bge, list) and len(first_bge) > 0:
                if isinstance(first_bge[0], list):
                    # 切片向量列表
                    print(f"BGE embedding: {len(first_bge)} chunks, each with {len(first_bge[0])} dimensions")
                else:
                    # 单个向量
                    print(f"BGE embedding dimension: {len(first_bge)}")
    
    # 处理GPT embedding维度统计
    if 'gpt_embedding' in result.columns:
        gpt_valid = result['gpt_embedding'].dropna()
        if len(gpt_valid) > 0:
            first_gpt = gpt_valid.iloc[0]
            if isinstance(first_gpt, list) and len(first_gpt) > 0:
                if isinstance(first_gpt[0], list):
                    # 切片向量列表
                    print(f"GPT embedding: {len(first_gpt)} chunks, each with {len(first_gpt[0])} dimensions")
                else:
                    # 单个向量
                    print(f"GPT embedding dimension: {len(first_gpt)}")
