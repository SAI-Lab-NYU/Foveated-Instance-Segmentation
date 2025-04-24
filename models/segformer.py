import torch
from transformers import SegformerForSemanticSegmentation, SegformerConfig, AutoImageProcessor
from torch import nn
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F

__all__ = ['segformer']

class CustomSegformer(SegformerForSemanticSegmentation):
    def __init__(self, config, num_input=5):
        super().__init__(config)
        
        old_conv = self.segformer.encoder.patch_embeddings[0].proj
        new_in_channels = num_input

        new_conv = nn.Conv2d(
            in_channels=new_in_channels, 
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None  
        )

        self.segformer.encoder.patch_embeddings[0].proj = new_conv
        
    def forward(self, pixel_values, return_feature_maps=True):
        return_dict =  self.config.use_return_dict
        output_hidden_states = (
           self.config.output_hidden_states
        )
        #print(self.segformer)
        outputs = self.segformer(
            pixel_values,
            output_hidden_states=True,  
            return_dict=return_dict,
        )
        # print(f'00000000{return_dict}')
        encoder_hidden_states = outputs.hidden_states if return_dict else outputs[1]
        # print(f'1111111 segformer encoder output shape {len(encoder_hidden_states)}')
        # print(f'2222222 segformer encoder output shape {encoder_hidden_states[0].shape}') #torch.Size([20, 64, 20, 20])
        # print(f'3333333 segformer encoder output shape {encoder_hidden_states[1].shape}') #torch.Size([20, 128, 10, 10])
        # print(f'4444444 segformer encoder output shape {encoder_hidden_states[2].shape}') #torch.Size([20, 320, 5, 5])
        # print(f'5555555 segformer encoder output shape {encoder_hidden_states[3].shape}') #torch.Size([20, 512, 3, 3])
        
        x0_h, x0_w = encoder_hidden_states[0].size(2), encoder_hidden_states[0].size(3)
        encoder_hidden_states1 = F.interpolate(
            encoder_hidden_states[1], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        encoder_hidden_states2 = F.interpolate(
            encoder_hidden_states[2], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        encoder_hidden_states3 = F.interpolate(
            encoder_hidden_states[3], size=(x0_h, x0_w), mode='bilinear', align_corners=False)
        x = torch.cat([encoder_hidden_states[0], encoder_hidden_states1, encoder_hidden_states2, encoder_hidden_states3], 1)

        # print(self.decode_head)
        # logits = self.decode_head(encoder_hidden_states)
        # upsampled_logits = nn.functional.interpolate(
        #         logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False
        #     )
        return [x] 

def test_with_dummy_input(model):
    dummy_input = torch.randn(3, 3, 80, 80)
    with torch.no_grad():
        output = model(dummy_input)[0]
    
    print(f"Output shape: {output.shape}") 

def test_with_gradient(model):
    dummy_input = torch.randn(3, 3, 80, 80)
    dummy_target = torch.randn(3, 1, 80, 80)

    model.train()
    output = model(dummy_input)

    criterion = nn.MSELoss()
    loss = criterion(output, dummy_target)
    loss.backward()

    print("\nChecking gradients for trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"{name} has gradient: {param.grad.norm().item()}")
        elif param.requires_grad and param.grad is None:
            print(f"{name} has NO gradient!")

def segformer(pretrained=False, **kwargs):
    config = SegformerConfig()
    #config = SegformerConfig.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
    # b5
    config.num_labels = 960  
    config.depths = [3,6,40,3] #b5 [3,6,40,3] #larger b5 [4,8,40,6]
    config.hidden_sizes = [64, 128, 320, 512]
    config.strides = [1,2,2,2] 
    config.decoder_hidden_size = 256
    config.hidden_dropout_prob = 0.3
    config.attention_probs_dropout_prob = 0.2
    # config.hidden_dropout_prob = 0.2
    # config.classifier_dropout_prob = 0.2
    model = CustomSegformer(config=config, num_input = 3)

    # if pretrained:
    #     model.load_state_dict(load_url(model_urls['hrnetv2']), strict=False)

    return model


if __name__ == "__main__":
    from transformers import SegformerForSemanticSegmentation, SegformerConfig, AutoImageProcessor

# if version == 4:
# config = SegformerConfig.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
# config.num_labels = 1  
# else:
    config = SegformerConfig()#.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
    config.num_labels = 960  
    config.depths = [2,2,2,2]
    config.hidden_sizes = [64, 128, 320, 512]
    config.strides = [1,2,2,2] 
    config.decoder_hidden_size = 256
    config.hidden_dropout_prob = 0.3
    model = CustomSegformer(config=config, num_input = 3)
    test_with_dummy_input(model)
    #test_with_gradient(model)