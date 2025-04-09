import torch
from diffusers.models.attention import CrossAttention

class MyCrossAttnProcessor:
    # # 原版Pro注意机制
    # def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
    #     batch_size, sequence_length, _ = hidden_states.shape
    #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

    #     query = attn.to_q(hidden_states)

    #     encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
    #     key = attn.to_k(encoder_hidden_states)
    #     value = attn.to_v(encoder_hidden_states)

    #     query = attn.head_to_batch_dim(query)
    #     key = attn.head_to_batch_dim(key)
    #     value = attn.head_to_batch_dim(value)

    #     # Scaled dot-product attention with efficient matmul
    #     scale = (query.size(-1) ** -0.5)
    #     scores = torch.bmm(query, key.transpose(-1, -2)) * scale
    #     if attention_mask is not None:
    #         scores = scores + attention_mask
        
    #     attention_probs = torch.softmax(scores, dim=-1)
    #     # attention_probs = attn.get_attention_scores(query, key, attention_mask)
    #     # new bookkeeping to save the attn probs
    #     attn.attn_probs = attention_probs

    #     hidden_states = torch.bmm(attention_probs, value)
    #     hidden_states = attn.batch_to_head_dim(hidden_states)

    #     # linear proj
    #     hidden_states = attn.to_out[0](hidden_states)
    #     # dropout
    #     hidden_states = attn.to_out[1](hidden_states)

    #     return hidden_states

    # 窗口交叉注意
    def __init__(self, window_size=8):
        self.window_size = window_size
        
    def _partition_windows(self, x, window_size):
        """将输入张量划分为局部窗口"""
        B, L, C = x.shape
        H = W = int(L ** 0.5)  # 假设输入是方形的，例如 77 = 8.7...
        
        # 确保输入序列长度是完全平方数
        if H * H != L:
            # 如果不是完全平方数，向上取整
            H = W = int((L ** 0.5) + 0.5)
            # 填充到完全平方数
            pad_len = H * H - L
            if pad_len > 0:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_len))
        
        # 调整形状为 (B, H, W, C)
        x = x.view(B, H, W, C)
        
        # 计算需要的padding
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        
        Hp, Wp = H + pad_h, W + pad_w
        
        # 重塑为窗口
        x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size * window_size, C)
        
        return windows, (Hp, Wp), L  # 返回原始序列长度

    def _partition_attention_mask(self, attention_mask, num_heads, window_size, Hp, Wp):
        """将注意力掩码划分为窗口"""
        B = attention_mask.shape[0]
        
        # 调整掩码形状并填充
        attention_mask = attention_mask.view(B, -1)
        if attention_mask.size(1) < Hp * Wp:
            pad_len = Hp * Wp - attention_mask.size(1)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
        
        # 重塑为窗口结构
        attention_mask = attention_mask.view(B, Hp, Wp)
        attention_mask = attention_mask.view(
            B, Hp // window_size, window_size, Wp // window_size, window_size
        )
        attention_mask = attention_mask.permute(0, 1, 3, 2, 4).contiguous()
        
        # 调整为窗口批次
        num_windows = (Hp // window_size) * (Wp // window_size)
        attention_mask = attention_mask.view(B * num_windows, window_size * window_size)
        
        # 扩展维度以匹配注意力分数的形状
        attention_mask = attention_mask.view(B * num_windows, 1, 1, window_size * window_size)
        attention_mask = attention_mask.expand(-1, num_heads, window_size * window_size, -1)
        attention_mask = attention_mask.view(B * num_windows * num_heads, window_size * window_size, window_size * window_size)
        
        return attention_mask

    def _reverse_windows(self, windows, original_size, H, W, orig_L):
        """将窗口数据还原为原始形状"""
        B = original_size[0]
        window_size = self.window_size
        C = windows.size(-1)
        
        # 重建窗口形状
        windows = windows.view(B, H // window_size, W // window_size, 
                             window_size, window_size, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = windows.view(B, H, W, C)
        
        # 移除padding，恢复到原始序列长度
        x = x.view(B, -1, C)
        if x.size(1) > orig_L:
            x = x[:, :orig_L, :]
        
        return x

    def __call__(self, attn: CrossAttention, hidden_states, encoder_hidden_states=None, attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        # 将输入划分为窗口
        hidden_states_windows, (Hp, Wp), orig_L = self._partition_windows(hidden_states, self.window_size)
        
        query = attn.to_q(hidden_states_windows)

        if encoder_hidden_states is not None:
            encoder_hidden_states_windows, _, _ = self._partition_windows(encoder_hidden_states, self.window_size)
        else:
            encoder_hidden_states_windows = hidden_states_windows
            
        key = attn.to_k(encoder_hidden_states_windows)
        value = attn.to_v(encoder_hidden_states_windows)

        # 手动调整形状
        query = query.view(batch_size * attn.heads, -1, query.size(-1))
        key = key.view(batch_size * attn.heads, -1, key.size(-1))
        value = value.view(batch_size * attn.heads, -1, value.size(-1))

        # 调整注意力掩码
        if attention_mask is not None:
            attention_mask = self._partition_attention_mask(
                attention_mask, 
                attn.heads,  # 传入头数
                self.window_size, 
                Hp, 
                Wp
            )

        # 在窗口内计算注意力
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attn.attn_probs = attention_probs

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = hidden_states.view(batch_size, attn.heads, -1, hidden_states.size(-1))
        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        hidden_states = hidden_states.view(batch_size, -1, hidden_states.size(-1))

        # 将窗口数据还原为原始形状
        hidden_states = self._reverse_windows(hidden_states, 
                                            (batch_size, sequence_length, hidden_states.size(-1)),
                                            Hp, Wp, orig_L)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

"""
A function that prepares a U-Net model for training by enabling gradient computation 
for a specified set of parameters and setting the forward pass to be performed by a 
custom cross attention processor.

Parameters:
unet: A U-Net model.

Returns:
unet: The prepared U-Net model.
"""
def prep_unet(unet):
    # set the gradients for XA maps to be true
    for name, params in unet.named_parameters():
        if 'attn2' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False
    # replace the fwd function
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            # module.set_processor(MyCrossAttnProcessor())
            module.set_processor(MyCrossAttnProcessor(window_size=8))
    return unet
