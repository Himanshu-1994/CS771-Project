import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU



def get_prompt_list(type):
	if type=="plain":
		return ['{}']
	elif type=="fixed":
		return ['a photo of a {}.']
	elif type=="shuffle":
		return ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.']
	elif type=="shuffle+":
		return ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.',
					'a cropped photo of a {}.', 'a good photo of a {}.', 'a photo of one {}.',
					'a bad photo of a {}.', 'a photo of the {}.']

	else:
		raise ValueError("Incorrect Prompt Type")
	

def forward_multihead_attention(inp, mod, atten_mask = None):
	"""
	inp: num_tokens*batch_size*emb_dim
	"""


	attn_in = mod.ln_1(inp)
	q,k,v = F.linear(attn_in, mod.attn.in_proj_weight, mod.attn.in_proj_bias).chunk(3,dim=-1)

	# MultiHead Projection
	num_tokens, batch_size, embed_dim = q.size()
	num_heads = mod.attn.num_heads

	dim = embed_dim//num_heads
	scale = 1.0/(float(dim)**0.5)

	q = q.contiguous().view(-1, batch_size * num_heads, dim).transpose(0, 1)
	k = k.contiguous().view(-1, batch_size * num_heads, dim).transpose(0, 1)
	v = v.contiguous().view(-1, batch_size * num_heads, dim).transpose(0, 1)

	# Q -> (bs*num_heads, num_tokens, dim)

	# (Q.K^T)/sqrt(dim)
	attn_weights = torch.bmm(q, k.transpose(1,2)) * scale

	# bs*num_heads, num_tokens, num_tokens
	attn_weights = torch.softmax(attn_weights, dim=-1)

	# (bs*heads,tokens,tokens) * (bs*heads,tokens,dim) -> (bs*heads,tokens,dim)
	attn_out = torch.bmm(attn_weights, v)
	attn_out = attn_out.transpose(0,1).contiguous.view(-1,batch_size,num_heads*dim)

	interm = inp + mod.attn.out_proj(attn_out)
	out = interm + mod.mlp(mod.ln_2(interm))

	out = mod.ln_2(out)

	return out, attn_weights

