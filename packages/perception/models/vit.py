from transformers.models.vit.modeling_vit import ViTForImageClassification


class ViT(ViTForImageClassification):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, pixel_values=None, labels=None):
        outputs = super().forward(pixel_values=pixel_values, labels=labels)
        return outputs
