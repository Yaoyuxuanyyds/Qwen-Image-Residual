from diffusers import QwenImagePipeline
from transformer import MyQwenImageTransformer2DModel
class MyQwenImagePipeline(QwenImagePipeline):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        residual_origin_layer=None,
        residual_target_layers=None,
        residual_weights=None,
        **kwargs,
    ):
        # 先加载官方 pipeline
        base_pipe = QwenImagePipeline.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )

        # 创建我们自己的 transformer
        cfg = base_pipe.transformer.config

        my_transformer = MyQwenImageTransformer2DModel(
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            num_layers=cfg.num_layers,
            attention_head_dim=cfg.attention_head_dim,
            num_attention_heads=cfg.num_attention_heads,
            joint_attention_dim=cfg.joint_attention_dim,
            guidance_embeds=cfg.guidance_embeds,
            axes_dims_rope=tuple(cfg.axes_dims_rope),
        )

        # 拷贝参数
        my_transformer.load_state_dict(base_pipe.transformer.state_dict())

        # ⭐⭐⭐ 必须：让 dtype 匹配 pipe（bfloat16）
        my_transformer.to(base_pipe.transformer.dtype)

        # 设置 residual
        my_transformer.set_residual_config(
            residual_origin_layer,
            residual_target_layers,
            residual_weights,
        )


        # 构建新的 pipeline
        return cls(
            scheduler=base_pipe.scheduler,
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            transformer=my_transformer,
        )