import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.activation import ReLU
import clip


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

class Clipbase(nn.Module):

	def __init__(self, reduce_dim, prompt_type, device='cpu'):
		super().__init__()
		self.prompt_list = get_prompt_list(prompt_type)

		version = "ViT-B/32"
		self.clip_model, _ = clip.load(version,device = device, jit='False')
		self.visual_clip = self.clip_model.visual

		for param in self.clip_model.parameters():
			param.requires_grad_(False)

		self.film_mul = nn.Linear(512, reduce_dim)
		self.film_add = nn.Linear(512, reduce_dim)


	def visual_forward(self, input, extract_layers=()):
		
		with torch.no_grad():

			inp_shape = input.shape[2:]
			x = self.visual_clip.conv1(input)

			# shape = [bs, channel, grid ** 2]
			x = x.reshape(x.shape[0],x.shape[1],-1)
			x = x.permute(0,2,1)

			# shape = [*, grid ** 2 + 1, channel]
			cls_emb = self.visual_clip.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
			x = torch.cat([cls_emb, x], dim=1)  # shape = [*, grid ** 2 + 1, width]

			assert x.shape[1]==50, f" Shape 1 of input should be 50, got: {x}"

			standard_n_tokens = 50
			x = x + self.visual_clip.positional_embedding.to(x.dtype)

			x = self.visual_clip.ln_pre(x)
			# [Token,BS,EMD]
			x = x.permute(1,0,2)
			activations = []

			for i, block in enumerate(self.visual_clip.transformer.resblocks):
				
				x, _ = forward_multihead_attention(x,block)

				if i in extract_layers:
					activations += [x]

			if self.visual_clip.proj:
				x = torch.matmul(x,self.visual_clip.proj)

			x = x.permute(1,0,2)
			x = self.visual_clip.ln_post(x[:,0,:])

			return x, activations

	def get_conditional_vec(self, conditional, batch_size):
	
		# Modify based on what is conditional [string]*batchsize etc.
		cond = self.compute_conditional(conditional)
		return cond

	def compute_conditional(self, conditional):

		dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		text_tokens = clip.tokenize(conditional).to(dev)
		cond = self.clip_model.encode_text(text_tokens)
		return cond