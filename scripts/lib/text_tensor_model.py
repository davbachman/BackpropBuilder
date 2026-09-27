"""Batched PyTorch interpreter of the app's elementary text graph.
No pretrained modules: parameters come directly from the graph template.
"""
import torch
from torch import nn
from torch.nn import functional as F


class TextTensorModel(nn.Module):
    def __init__(self, template):
        super().__init__()
        self.graph = template['graph']
        self.expressions = template['expressions']
        self.source = next(n for n in self.graph['nodes'] if n['type'] == 'dataset')
        self.data = self.source['params']['textData']
        self.nodes = {n['id']: n for n in self.graph['nodes']}
        self.inputs = {key: sorted([e for e in self.graph['edges'] if e['target'] == key], key=lambda e: e.get('inputSlot', 0)) for key in self.nodes}
        self.order = []
        pending = set(self.nodes)
        while pending:
            ready = [key for key in self.nodes if key in pending and all(e['source'] not in pending for e in self.inputs[key])]
            if not ready:
                raise ValueError('Graph has a cycle')
            self.order.extend(ready)
            pending.difference_update(ready)
        ids = [key for key, n in self.nodes.items() if n['type'] in ('weight', 'bias')]
        self.parameter_ids = ids
        self.weights = nn.ParameterList([nn.Parameter(torch.tensor(self.nodes[key]['params']['value']['data'], dtype=torch.float32).reshape(self.nodes[key]['params']['value']['shape'])) for key in ids])
        self.parameter_map = dict(zip(ids, self.weights))

    def arithmetic(self, expr, args):
        kind = expr['kind']
        if kind == 'number': return expr['value']
        if kind == 'input': return args[expr['index']]
        if kind == 'negate': return -self.arithmetic(expr['child'], args)
        a, b = self.arithmetic(expr['left'], args), self.arithmetic(expr['right'], args)
        op = expr['op']
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return a / b
        if op == '^': return a ** b
        raise ValueError(op)

    def forward(self, ids, lengths, target):
        batch, width = ids.shape
        mask = torch.arange(width, device=ids.device)[None, :] < lengths[:, None]
        positions = torch.arange(width, device=ids.device)[None, :].expand(batch, -1)
        positions = positions.masked_fill(~mask, 0)
        counts = ids.new_zeros((batch, len(self.data['vocabulary'])), dtype=torch.float32).scatter_add(1, ids, mask.float())
        features = [(ids, ['batch', 'token']) if self.data['representation'] == 'tokens' else (counts[:, None, :], ['batch', 'feature', 'feature']), (positions, ['batch', 'token']), (target.reshape(batch, 1, 1), ['batch', 'feature', 'feature'])]
        values = {}
        def read(edge):
            return features[edge.get('sourceSlot', 0)] if edge['source'] == self.source['id'] else values[edge['source']]
        prediction = loss = None
        for key in self.order:
            node = self.nodes[key]
            kind, p = node['type'], node['params']
            if kind == 'dataset': continue
            args = [read(e) for e in self.inputs[key]]
            tensors = [a[0] for a in args]
            axes = next((a[1] for a in args if 'batch' in a[1]), args[0][1] if args else [])
            if kind == 'tensor-transform': kind = p['transform']
            if kind in ('weight', 'bias'):
                value = self.parameter_map[key]; axes = ['feature'] * value.ndim
            elif kind in ('input', 'target'): value = tensors[0]
            elif kind == 'embedding': value = F.embedding(tensors[1].long(), tensors[0]); axes = args[1][1] + ['feature']
            elif kind == 'one-hot': value = F.one_hot(tensors[0].long(), p['numClasses']).float(); axes = args[0][1] + ['feature']
            elif kind == 'matmul': value = tensors[0] @ tensors[1]; axes = args[0][1][:-1] + args[1][1][-1:]
            elif kind == 'arithmetic': value = self.arithmetic(self.expressions[key], tensors)
            elif kind == 'dropout': value = F.dropout(tensors[0], p=p.get('dropoutRate', .1), training=self.training)
            elif kind == 'activation':
                value = {'relu': F.relu, 'sigmoid': torch.sigmoid, 'tanh': torch.tanh, 'identity': lambda x: x}[p.get('activation', 'identity')](tensors[0])
            elif kind == 'transpose':
                batched = args[0][1][0] == 'batch'
                permutation = p.get('axes', list(reversed(range(tensors[0].ndim - int(batched)))))
                if batched: permutation = [0] + [i + 1 for i in permutation]
                value = tensors[0].permute(permutation); axes = [args[0][1][i] for i in permutation]
            elif kind == 'mean':
                axis = p.get('axis', 0) + int(args[0][1][0] == 'batch')
                keep = p.get('keepDims', False)
                if args[0][1][axis] == 'token':
                    weights = mask.reshape(*mask.shape, *([1] * (tensors[0].ndim - 2))).float()
                    value = (tensors[0] * weights).sum(axis, keepdim=keep) / weights.sum(axis, keepdim=keep)
                else: value = tensors[0].mean(axis, keepdim=keep)
                axes = list(args[0][1])
                if keep: axes[axis] = 'feature'
                else: axes.pop(axis)
            elif kind == 'concat':
                axis = p.get('axis', 1) + int(args[0][1][0] == 'batch')
                if args[0][1][axis] == 'token': raise ValueError('Only feature concatenation is supported')
                value = torch.cat(tensors, dim=axis)
            elif kind == 'softmax':
                scores = tensors[0]
                if args[0][1][-1] == 'token': scores = scores.masked_fill(~mask[:, None, :], -1e9)
                value = scores.softmax(-1)
            elif kind == 'causal-mask': value = tensors[0].masked_fill(torch.triu(torch.ones_like(tensors[0], dtype=torch.bool), diagonal=1), -1e9)
            elif kind == 'layer-norm': value = F.layer_norm(tensors[0], (tensors[0].shape[-1],), tensors[1], tensors[2], eps=p.get('epsilon', 1e-5))
            elif kind == 'loss':
                prediction = tensors[0]
                probability = prediction.clamp(1e-7, 1-1e-7)
                loss = -(tensors[1] * probability.log() + (1-tensors[1]) * (1-probability).log()).mean()
                value = loss
            else: raise ValueError('Unsupported operation '+kind)
            values[key] = value, axes
        return prediction.reshape(-1), loss

    def graph_weights(self):
        return {key: {'shape': list(value.shape), 'data': value.detach().cpu().reshape(-1).tolist()} for key, value in self.parameter_map.items()}
