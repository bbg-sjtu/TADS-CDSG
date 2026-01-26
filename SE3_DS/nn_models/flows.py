import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F
import numpy as np

class BijectionNet(nn.Sequential):
	"""
	A sequential container of flows based on coupling layers.
	"""
	def __init__(self, num_dims, num_blocks, num_hidden, s_act='relu', t_act='relu', sigma=None,
				 coupling_network_type='fcnn', device='cuda'):
		self.num_dims = num_dims
		modules = []
		print('Using the {} for coupling layer'.format(coupling_network_type))
		mask = torch.arange(0, num_dims) % 2  # alternating inputs
		mask = mask.float()
		for _ in range(num_blocks):
			modules += [
				CouplingLayer(
					num_inputs=num_dims, num_hidden=num_hidden, mask=mask,
					s_act=s_act, t_act=t_act, sigma=sigma, base_network=coupling_network_type,device=device),
			]
			mask = 1 - mask  # flipping mask
		super(BijectionNet, self).__init__(*modules)

	def jacobian(self, inputs, mode='direct'):
		'''
		Finds the product of all jacobians
		'''
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		if mode == 'direct':
			for module in self._modules.values():
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				# inputs = module(inputs, mode)
		else:
			for module in reversed(self._modules.values()):
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				# inputs = module(inputs, mode)
		return J

	def forward(self, inputs, mode='direct'):
		""" Performs a forward or backward pass for flow modules.
		Args:
			inputs: a tuple of inputs and logdets
			mode: to run direct computation or inverse
		"""
		assert mode in ['direct', 'inverse']
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		if mode == 'direct':
			for module in self._modules.values():
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				inputs = module(inputs, mode)
		else:
			for module in reversed(self._modules.values()):
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				inputs = module(inputs, mode)
		return inputs, J

class CouplingLayer(nn.Module):
	""" An implementation of a coupling layer
	from RealNVP (https://arxiv.org/abs/1605.08803).
	"""

	def __init__(self, num_inputs, num_hidden, mask,
				 base_network='rffn', s_act='elu', t_act='elu', sigma=0.45, device='cuda'):
		super(CouplingLayer, self).__init__()

		self.num_inputs = num_inputs
		self.mask = mask.to(device)

		if base_network == 'fcnn':
			self.scale_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=s_act)
			self.translate_net = FCNN(in_dim=num_inputs, out_dim=num_inputs, hidden_dim=num_hidden, act=t_act)

			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.translate_net.network[-1].bias.data)

			nn.init.zeros_(self.scale_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].bias.data)

		elif base_network == 'rffn':
			print('Using random fouier feature with bandwidth = {}.'.format(sigma))
			self.scale_net = RFFN(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)
			self.translate_net = RFFN(in_dim=num_inputs, out_dim=num_inputs, nfeat=num_hidden, sigma=sigma)

			print('Initializing coupling layers as identity!')
			nn.init.zeros_(self.translate_net.network[-1].weight.data)
			nn.init.zeros_(self.scale_net.network[-1].weight.data)
		else:
			raise TypeError('The network type has not been defined')

	def forward(self, inputs, mode='direct'):
		mask = self.mask
		masked_inputs = inputs * mask

		log_s = self.scale_net(masked_inputs) * (1 - mask)
		t = self.translate_net(masked_inputs) * (1 - mask)

		if mode == 'direct':
			s = torch.exp(log_s)
			return inputs * s + t
		else:
			s = torch.exp(-log_s)
			return (inputs - t) * s

	def jacobian(self, inputs):
		return get_jacobian(self, inputs, inputs.size(-1))


class ConditionalBijectionNet(nn.Sequential):

	def __init__(self, num_dims, num_condition, num_blocks, num_hidden, s_act='relu', t_act='relu', sigma=None,
				 coupling_network_type='fcnn', device='cuda'):
		self.device=device
		self.num_dims = num_dims
		modules = []
		print('Using the {} for coupling layer'.format(coupling_network_type))
		mask = torch.arange(0, num_dims) % 2  # alternating inputs
		mask = mask.float()
		# mask = mask.to(device).float()
		for _ in range(num_blocks):
			modules += [
				ConditionalCouplingLayer(num_inputs=num_dims, num_condition=num_condition, num_hidden=num_hidden, mask=mask, s_act=s_act, t_act=t_act, device=self.device),
			]
			mask = 1 - mask  # flipping mask
		super(ConditionalBijectionNet, self).__init__(*modules)

	def jacobian(self, inputs, mode='direct'):
		'''
		Finds the product of all jacobians
		'''
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		if mode == 'direct':
			for module in self._modules.values():
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				# inputs = module(inputs, mode)
		else:
			for module in reversed(self._modules.values()):
				J_module = module.jacobian(inputs)
				J = torch.matmul(J_module, J)
				# inputs = module(inputs, mode)
		return J

	def forward(self, inputs, condition, mode='direct'):
		""" Performs a forward or backward pass for flow modules.
		Args:
			inputs: a tuple of inputs and logdets
			mode: to run direct computation or inverse
		"""
		assert mode in ['direct', 'inverse']
		batch_size = inputs.size(0)
		J = torch.eye(self.num_dims, device=inputs.device).unsqueeze(0).repeat(batch_size, 1, 1)

		if mode == 'direct':
			for module in self._modules.values():
				J_module = module.jacobian(inputs, condition)
				J = torch.matmul(J_module, J)
				inputs = module(inputs, condition, mode)
		else:
			for module in reversed(self._modules.values()):
				J_module = module.jacobian(inputs, condition)
				J = torch.matmul(J_module, J)
				inputs = module(inputs, condition, mode)
		return inputs, J
	
	def forward_noJ(self, inputs, condition, mode='direct'):
		""" Performs a forward or backward pass for flow modules.
		Args:
			inputs: a tuple of inputs and logdets
			mode: to run direct computation or inverse
		"""
		assert mode in ['direct', 'inverse']

		if mode == 'direct':
			for module in self._modules.values():
				inputs = module(inputs, condition, mode)
		else:
			for module in reversed(self._modules.values()):
				inputs = module(inputs, condition, mode)
		return inputs

class ConditionalCouplingLayer(nn.Module):

	def __init__(self, num_inputs, num_condition, num_hidden, mask, s_act='elu', t_act='elu',device='cuda'):
		super(ConditionalCouplingLayer, self).__init__()

		self.num_inputs = num_inputs
		self.mask = mask.to(device)

		self.scale_net = Conditional_FCNN(in_dim=num_inputs, condition_dim=num_condition, out_dim=num_inputs, hidden_dim=num_hidden, act=s_act)
		self.translate_net = Conditional_FCNN(in_dim=num_inputs, condition_dim=num_condition, out_dim=num_inputs, hidden_dim=num_hidden, act=t_act)

		nn.init.zeros_(self.translate_net.network2[-1].weight.data)
		nn.init.zeros_(self.translate_net.network2[-1].bias.data)
		nn.init.zeros_(self.scale_net.network2[-1].weight.data)
		nn.init.zeros_(self.scale_net.network2[-1].bias.data)

	def forward(self, inputs, condition, mode='direct'):
		mask = self.mask
		masked_inputs = inputs * mask
		# masked_inputs.requires_grad_(True)

		log_s = self.scale_net(masked_inputs, condition) * (1 - mask)
		t = self.translate_net(masked_inputs, condition) * (1 - mask)

		if mode == 'direct':
			s = torch.exp(log_s)
			return inputs * s + t
		else:
			s = torch.exp(-log_s)
			return (inputs - t) * s

	def jacobian(self, inputs, condition):
		return get_jacobian_condition(self, inputs, condition, inputs.size(-1))

class RFFN(nn.Module):
	"""
	Random Fourier features network.
	"""

	def __init__(self, in_dim, out_dim, nfeat, sigma=10.):
		super(RFFN, self).__init__()
		self.sigma = np.ones(in_dim) * sigma
		self.coeff = np.random.normal(0.0, 1.0, (nfeat, in_dim))
		self.coeff = self.coeff / self.sigma.reshape(1, len(self.sigma))
		self.offset = 2.0 * np.pi * np.random.rand(1, nfeat)

		self.network = nn.Sequential(
			LinearClamped(in_dim, nfeat, self.coeff, self.offset),
			Cos(),
			nn.Linear(nfeat, out_dim, bias=False)
		)

	def forward(self, x):
		return self.network(x)

class FCNN(nn.Module):
	'''
	2-layer fully connected neural network
	'''
	def __init__(self, in_dim, out_dim, hidden_dim, act='tanh'):
		super(FCNN, self).__init__()
		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
					   'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}

		act_func = activations[act]
		self.network = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, out_dim)
		)

	def forward(self, x):
		return self.network(x)

class Conditional_FCNN(nn.Module):
	'''
	2-layer conditional fully connected neural network
	'''
	def __init__(self, in_dim, condition_dim, out_dim, hidden_dim, act='tanh'):
		super(Conditional_FCNN, self).__init__()
		activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leaky_relu': nn.LeakyReLU,
					   'elu': nn.ELU, 'prelu': nn.PReLU, 'softplus': nn.Softplus}

		act_func = activations[act]

		self.network1 = nn.Sequential(
			nn.Linear(in_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim)
		)
		self.network2 = nn.Sequential(
			nn.Linear(hidden_dim*2, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, out_dim)
		)
		self.condition_layer = nn.Sequential(
			nn.Linear(condition_dim, hidden_dim), act_func(),
			nn.Linear(hidden_dim, hidden_dim)
		)

	def forward(self, x, c):
		h1 = self.network1(x)
		h2 = self.condition_layer(c)
		h = torch.cat((h1, h2), dim=1)
		return self.network2(h)

class LinearClamped(nn.Module):
	'''
	Linear layer with user-specified parameters (not to be learrned!)
	'''

	__constants__ = ['bias', 'in_features', 'out_features']

	def __init__(self, in_features, out_features, weights, bias_values, bias=True):
		super(LinearClamped, self).__init__()
		self.in_features = in_features
		self.out_features = out_features

		self.register_buffer('weight', torch.Tensor(weights))
		if bias:
			self.register_buffer('bias', torch.Tensor(bias_values))

	def forward(self, input):
		if input.dim() == 1:
			return F.linear(input.view(1, -1), self.weight, self.bias)
		return F.linear(input, self.weight, self.bias)

	def extra_repr(self):
		return 'in_features={}, out_features={}, bias={}'.format(
			self.in_features, self.out_features, self.bias is not None
		)


class Cos(nn.Module):
	"""
	Applies the cosine element-wise function
	"""

	def forward(self, inputs):
		return torch.cos(inputs)

def get_jacobian(net, x, output_dims, reshape_flag=True):
	"""

	"""
	if x.ndimension() == 1:
		n = 1
	else:
		n = x.size()[0]
	x_m = x.repeat(1, output_dims).view(-1, output_dims)
	x_m.requires_grad_(True)
	y_m = net(x_m)
	mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
	# y.backward(mask)
	J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
	if reshape_flag:
		J = J.reshape(n, output_dims, output_dims)
	return J

def get_jacobian_condition(net, x, c, output_dims, reshape_flag=True):

	if x.ndimension() == 1:
		n = 1
	else:
		n = x.size()[0]
	x_m = x.repeat(1, output_dims).view(-1, output_dims)
	c_m = c.repeat(1, output_dims).view(-1, c.size(-1))
	x_m.requires_grad_(True)
	y_m = net(x_m, c_m)
	mask = torch.eye(output_dims).repeat(n, 1).to(x.device)
	# y.backward(mask)
	J = autograd.grad(y_m, x_m, mask, create_graph=True)[0]
	if reshape_flag:
		J = J.reshape(n, output_dims, output_dims)
	return J