import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Union

from diffusers.models import QwenImageTransformer2DModel
from diffusers.utils import logging
from diffusers.models.modeling_outputs import Transformer2DModelOutput

logger = logging.get_logger(__name__)


class MyQwenImageTransformer2DModel(QwenImageTransformer2DModel):
    """
    增加 text stream 的跨层 residual 注入功能
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.residual_origin_layer: Optional[int] = None
        self.residual_target_layers: List[int] = []
        self.residual_weights: Optional[torch.Tensor] = None

        self._saved_origin_text: Optional[torch.Tensor] = None

    @staticmethod
    def _standardize_tokenwise(x: torch.Tensor, eps: float = 1e-6):
        """
        对最后一维 (hidden_dim) 做 z-score：
        x_norm = (x - mean) / (std + eps)
        返回 (x_norm, mean, std)，方便之后把 scale/shift 加回去。
        形状保持和 x 一致。
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mean) / (std + eps)
        return x_norm, mean, std


    def set_residual_config(
        self,
        residual_origin_layer: Optional[int],
        residual_target_layers: Optional[Union[List[int], torch.Tensor]],
        residual_weights: Optional[Union[List[float], torch.Tensor]],
    ):
        if residual_origin_layer is None:
            self.residual_origin_layer = None
            self.residual_target_layers = []
            self.residual_weights = None
            self._saved_origin_text = None
            return

        # layers
        if isinstance(residual_target_layers, torch.Tensor):
            residual_target_layers = residual_target_layers.tolist()
        self.residual_target_layers = [int(i) for i in residual_target_layers]

        # weights
        if isinstance(residual_weights, (list, tuple)):
            residual_weights = torch.tensor(residual_weights, dtype=torch.float32)
        self.residual_weights = residual_weights

        self.residual_origin_layer = int(residual_origin_layer)
        self._saved_origin_text = None

        logger.info(
            f"[Residual] origin={self.residual_origin_layer}, "
            f"targets={self.residual_target_layers}, "
            f"weights={self.residual_weights}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        target_layers: Optional[List[int]] = None,
    ):

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        collect_txt_features = target_layers is not None and len(target_layers) > 0
        target_layers = sorted(set(target_layers)) if collect_txt_features else []
        txt_feats_list: List[torch.Tensor] = []
        context_embedder_output: List[torch.Tensor] = []

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        self._saved_origin_text = None  # 每次 forward 清空

        use_residual = (
            self.residual_origin_layer is not None
            and self.residual_weights is not None
            and len(self.residual_target_layers) > 0
        )

        target_set = set(self.residual_target_layers)

        if collect_txt_features:
            context_embedder_output.append(encoder_hidden_states.detach())

        for layer_idx, block in enumerate(self.transformer_blocks):

            # 1. origin layer：保存 text tokens 输入
            if use_residual and layer_idx == self.residual_origin_layer:
                self._saved_origin_text = encoder_hidden_states.detach()

            # # 2. target layers：注入 residual
            # if use_residual and layer_idx in target_set and self._saved_origin_text is not None:
            #     tid = self.residual_target_layers.index(layer_idx)
            #     w = self.residual_weights[tid].to(
            #         encoder_hidden_states.device, encoder_hidden_states.dtype
            #     )

            #     encoder_hidden_states = (
            #         encoder_hidden_states + w * self._saved_origin_text
            #     )
            # 2. target layers：注入 residual（标准化 → 残差 → LN → rescale & reshift）
            if use_residual and layer_idx in target_set and self._saved_origin_text is not None:

                tid = self.residual_target_layers.index(layer_idx)
                w = self.residual_weights[tid].to(
                    encoder_hidden_states.device, encoder_hidden_states.dtype
                )

                # 当前层 (target) text tokens
                target = encoder_hidden_states
                # origin layer 中保存的 text tokens
                origin = self._saved_origin_text.to(
                    encoder_hidden_states.device, encoder_hidden_states.dtype
                )

                # === 1) origin 与 target 各自 token-wise 标准化 (z-score) ===
                target_norm, target_mean, target_std = self._standardize_tokenwise(target)
                origin_norm, _, _ = self._standardize_tokenwise(origin)

                # === 2) 在标准化空间中 residual ===
                mixed_norm = target_norm + w * origin_norm

                # === 3) 对 residual 结果做一次 LayerNorm ===
                # LN 在最后一维 hidden_dim 上做归一化
                mixed_norm_ln = torch.nn.functional.layer_norm(
                    mixed_norm, 
                    normalized_shape=mixed_norm.shape[-1:],
                    eps=1e-6
                )

                # === 4) 用 target 的 mean/std 恢复回 target layer 的原始分布 ===
                encoder_hidden_states = mixed_norm_ln * target_std + target_mean


            # 3. 正常执行 block
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )

            if collect_txt_features and layer_idx in target_layers:
                txt_feats_list.append(encoder_hidden_states.detach())

        # 清空缓存
        self._saved_origin_text = None

        # ↓↓↓↓↓↓↓↓↓↓↓↓↓ 剩余部分保持官方一致 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if collect_txt_features:
            # 当需要文本特征时，总是返回字典，便于可视化脚本读取
            return {
                "sample": output,
                "txt_feats_list": txt_feats_list,
                "context_embedder_output": context_embedder_output,
            }

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
