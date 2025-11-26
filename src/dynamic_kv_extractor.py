"""
动态 KV 提取器
自动适配不同模型的 KV Cache 结构
支持跨层聚合、动态维度检测、量化模型等
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Dict, Any
import warnings


class DynamicKVExtractor:
    """
    动态 KV Cache 提取器
    
    特性：
    1. 自动检测 KV Cache 结构
    2. 支持跨层聚合（Cross-Layer Aggregation）
    3. 兼容量化模型（4-bit, 8-bit）
    4. 自动维度对齐
    5. 运行时维度验证
    """
    
    def __init__(
        self,
        aggregation_method: str = "concat",  # concat / mean / weighted
        use_all_layers: bool = True,
        layer_weights: Optional[List[float]] = None,
        validate_shapes: bool = True,
    ):
        """
        初始化 KV 提取器
        
        Args:
            aggregation_method: 聚合方法
                - concat: 拼接所有层（默认）
                - mean: 平均所有层
                - weighted: 加权聚合
            use_all_layers: 是否使用所有层
            layer_weights: 层权重（仅 weighted 方法使用）
            validate_shapes: 是否验证形状
        """
        self.aggregation_method = aggregation_method
        self.use_all_layers = use_all_layers
        self.layer_weights = layer_weights
        self.validate_shapes = validate_shapes
        
        # 缓存维度信息
        self._cached_dims: Dict[str, int] = {}
        self._structure_analyzed = False
    
    def analyze_kv_structure(
        self,
        past_key_values: Tuple,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        分析 KV Cache 结构
        
        Args:
            past_key_values: 模型输出的 past_key_values
            model_name: 模型名称（用于缓存）
            
        Returns:
            结构信息字典
        """
        if not past_key_values:
            raise ValueError("past_key_values is empty")
        
        # 获取第一层信息
        first_layer = past_key_values[0]
        k, v = first_layer
        
        # 基本形状信息
        B, H, T, D_h = k.shape
        num_layers = len(past_key_values)
        
        structure = {
            'num_layers': num_layers,
            'batch_size': B,
            'num_heads': H,
            'seq_length': T,
            'head_dim': D_h,
            'layer_dim': H * D_h,
            'total_dim': num_layers * H * D_h,
            'dtype': k.dtype,
            'device': k.device,
        }
        
        # 检查所有层是否一致
        all_consistent = True
        for i, layer_kv in enumerate(past_key_values):
            k_layer, v_layer = layer_kv
            if k_layer.shape != k.shape:
                warnings.warn(f"Layer {i} has different shape: {k_layer.shape} vs {k.shape}")
                all_consistent = False
        
        structure['layers_consistent'] = all_consistent
        
        # 缓存维度信息
        self._cached_dims[model_name] = structure['total_dim']
        self._structure_analyzed = True
        
        return structure
    
    def extract_kv(
        self,
        past_key_values: Tuple,
        return_keys_only: bool = True,
        model_name: str = "model",
        debug: bool = False,
    ) -> torch.Tensor:
        """
        提取 KV Cache（主函数）
        
        Args:
            past_key_values: 模型输出的 past_key_values
            return_keys_only: 是否只返回 Keys（默认 True）
            model_name: 模型名称
            debug: 是否打印调试信息
            
        Returns:
            提取的 KV tensor，形状为 [B, T, D]
        """
        # 分析结构
        if debug or not self._structure_analyzed:
            structure = self.analyze_kv_structure(past_key_values, model_name)
            if debug:
                print(f"\n[KV Structure Analysis] ({model_name}):")
                for key, value in structure.items():
                    print(f"   {key}: {value}")
        
        # 根据聚合方法提取
        if self.aggregation_method == "concat":
            kv_flat = self._extract_concat(past_key_values, return_keys_only)
        elif self.aggregation_method == "mean":
            kv_flat = self._extract_mean(past_key_values, return_keys_only)
        elif self.aggregation_method == "weighted":
            kv_flat = self._extract_weighted(past_key_values, return_keys_only)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        # 验证形状
        if self.validate_shapes:
            self._validate_output_shape(kv_flat, model_name)
        
        return kv_flat
    
    def _extract_concat(
        self,
        past_key_values: Tuple,
        return_keys_only: bool
    ) -> torch.Tensor:
        """
        拼接聚合：将所有层的 KV 在最后一个维度拼接
        
        这是跨层聚合（Cross-Layer Aggregation）的核心实现
        """
        if not self.use_all_layers:
            # 只使用最后一层
            k, v = past_key_values[-1]
            kv = k if return_keys_only else v
            # [B, H, T, D_h] -> [B, T, H * D_h]
            B, H, T, D_h = kv.shape
            return kv.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_h)
        
        # 使用所有层（跨层聚合）
        all_kvs = []
        for layer_kv in past_key_values:
            k, v = layer_kv
            kv = k if return_keys_only else v
            
            # 展平单层：[B, H, T, D_h] -> [B, T, H * D_h]
            B, H, T, D_h = kv.shape
            kv_flat = kv.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_h)
            all_kvs.append(kv_flat)
        
        # 在最后一个维度拼接所有层
        # 结果形状：[B, T, num_layers * H * D_h]
        kv_combined = torch.cat(all_kvs, dim=-1)
        
        return kv_combined
    
    def _extract_mean(
        self,
        past_key_values: Tuple,
        return_keys_only: bool
    ) -> torch.Tensor:
        """
        平均聚合：对所有层取平均
        """
        all_kvs = []
        for layer_kv in past_key_values:
            k, v = layer_kv
            kv = k if return_keys_only else v
            
            # 展平：[B, H, T, D_h] -> [B, T, H * D_h]
            B, H, T, D_h = kv.shape
            kv_flat = kv.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_h)
            all_kvs.append(kv_flat)
        
        # 堆叠并取平均
        kv_stacked = torch.stack(all_kvs, dim=0)  # [num_layers, B, T, D]
        kv_mean = kv_stacked.mean(dim=0)  # [B, T, D]
        
        return kv_mean
    
    def _extract_weighted(
        self,
        past_key_values: Tuple,
        return_keys_only: bool
    ) -> torch.Tensor:
        """
        加权聚合：使用指定权重聚合各层
        """
        num_layers = len(past_key_values)
        
        # 使用默认权重（如果未指定）
        if self.layer_weights is None:
            # 后面的层权重更大
            weights = torch.linspace(0.5, 1.0, num_layers)
        else:
            weights = torch.tensor(self.layer_weights)
        
        # 归一化权重
        weights = weights / weights.sum()
        
        all_kvs = []
        for layer_kv in past_key_values:
            k, v = layer_kv
            kv = k if return_keys_only else v
            
            # 展平
            B, H, T, D_h = kv.shape
            kv_flat = kv.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_h)
            all_kvs.append(kv_flat)
        
        # 加权求和
        kv_stacked = torch.stack(all_kvs, dim=0)  # [num_layers, B, T, D]
        weights = weights.view(-1, 1, 1, 1).to(kv_stacked.device)
        kv_weighted = (kv_stacked * weights).sum(dim=0)  # [B, T, D]
        
        return kv_weighted
    
    def _validate_output_shape(self, kv_tensor: torch.Tensor, model_name: str):
        """验证输出形状是否符合预期"""
        if model_name in self._cached_dims:
            expected_dim = self._cached_dims[model_name]
            actual_dim = kv_tensor.shape[-1]
            
            if actual_dim != expected_dim:
                warnings.warn(
                    f"Dimension mismatch for {model_name}: "
                    f"expected {expected_dim}, got {actual_dim}"
                )
    
    def get_output_dim(self, model_name: str = "model") -> Optional[int]:
        """获取缓存的输出维度"""
        return self._cached_dims.get(model_name)
    
    def print_extraction_info(self, model_name: str = "model"):
        """打印提取器配置信息"""
        print(f"\n[KV Extractor Configuration] ({model_name}):")
        print(f"   Aggregation Method: {self.aggregation_method}")
        print(f"   Use All Layers: {self.use_all_layers}")
        if self.aggregation_method == "weighted" and self.layer_weights:
            print(f"   Layer Weights: {self.layer_weights}")
        if model_name in self._cached_dims:
            print(f"   Output Dimension: {self._cached_dims[model_name]}")
        print()


class AdaptiveProjector(nn.Module):
    """
    自适应投影器
    根据检测到的维度动态初始化
    """
    
    def __init__(
        self,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_ratio: float = 1.0,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        """
        初始化自适应投影器
        
        Args:
            input_dim: 输入维度（可选，支持延迟初始化）
            output_dim: 输出维度（可选，支持延迟初始化）
            hidden_ratio: 隐藏层维度比例
            dropout: Dropout 比例
            activation: 激活函数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_ratio = hidden_ratio
        self.dropout = dropout
        self.activation = activation
        
        self._initialized = False
        
        # 如果维度已知，立即初始化
        if input_dim is not None and output_dim is not None:
            self._build_layers()
    
    def _build_layers(self):
        """构建网络层"""
        hidden_dim = int(self.input_dim * self.hidden_ratio)
        
        # 激活函数映射
        activation_map = {
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
        }
        
        self.layers = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            activation_map.get(self.activation, nn.SiLU()),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, self.output_dim),
        )
        
        self._initialized = True
        
        print(f"[OK] Adaptive Projector initialized: {self.input_dim} -> {self.output_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Total params: {sum(p.numel() for p in self.parameters()):,}")
    
    def initialize_from_tensors(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor
    ):
        """
        从张量推断维度并初始化
        
        Args:
            input_tensor: 输入样例
            output_tensor: 输出样例
        """
        if self._initialized:
            warnings.warn("Projector already initialized, skipping")
            return
        
        self.input_dim = input_tensor.shape[-1]
        self.output_dim = output_tensor.shape[-1]
        
        self._build_layers()
        
        # 测试前向传播
        with torch.no_grad():
            test_output = self.forward(input_tensor[:1])
            assert test_output.shape[-1] == self.output_dim, \
                f"Output dim mismatch: {test_output.shape[-1]} vs {self.output_dim}"
        
        print("[OK] Forward pass test successful")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if not self._initialized:
            raise RuntimeError(
                "Projector not initialized. Call initialize_from_tensors() first."
            )
        
        return self.layers(x)


# ====================================================================
# 便捷函数
# ====================================================================

def create_kv_extractor(
    aggregation_method: str = "concat",
    use_all_layers: bool = True,
) -> DynamicKVExtractor:
    """
    创建 KV 提取器（便捷函数）
    
    Args:
        aggregation_method: 聚合方法
        use_all_layers: 是否使用所有层
        
    Returns:
        DynamicKVExtractor 实例
    """
    return DynamicKVExtractor(
        aggregation_method=aggregation_method,
        use_all_layers=use_all_layers,
        validate_shapes=True,
    )


if __name__ == "__main__":
    # 测试示例
    print("[Testing Dynamic KV Extractor]\n")
    
    # 模拟 KV Cache 结构
    num_layers = 28
    batch_size = 2
    num_heads = 2
    seq_length = 512
    head_dim = 128
    
    fake_kv = []
    for _ in range(num_layers):
        k = torch.randn(batch_size, num_heads, seq_length, head_dim)
        v = torch.randn(batch_size, num_heads, seq_length, head_dim)
        fake_kv.append((k, v))
    
    fake_kv = tuple(fake_kv)
    
    # 创建提取器
    extractor = create_kv_extractor(aggregation_method="concat", use_all_layers=True)
    
    # 提取 KV
    kv_flat = extractor.extract_kv(fake_kv, model_name="test_model", debug=True)
    
    print(f"\n[OK] Extracted KV shape: {kv_flat.shape}")
    print(f"   Expected: [batch_size, seq_length, num_layers * num_heads * head_dim]")
    print(f"   Got: [{batch_size}, {seq_length}, {num_layers * num_heads * head_dim}]")
