from torch import nn
import torch.nn.functional as F
import torch
from fuxictr.pytorch.models import BaseModel
from fuxictr.pytorch.layers import EmbeddingLayer, MLP_Layer

class GIN(BaseModel):
    def __init__(self, 
                 feature_map, 
                 model_id="GAIN", 
                 gpu=0,
                 task="binary_classification", 
                 learning_rate=1e-3, 
                 embedding_dim=16, 
                 strategy='hybrid',
                 units=40,
                 units_dict={},
                 threshold=0.0331,
                 top_k=9,                
                 dnn_hidden_units=[], 
                 dnn_hidden_activations="ReLU", 
                 dnn_dropout=0,
                 batch_norm=True, 
                 **kwargs):
        super(GIN, self).__init__(feature_map, 
                                model_id=model_id,
                                gpu=gpu, 
                                **kwargs) 

        num_fields = feature_map.num_fields
        self.num_fields = num_fields
        self.threshold = threshold
        self.strategy = strategy
        self.units_dict = units_dict
        self.top_k = top_k
        self.tau = 1
        
        if strategy == 'top-k':
            units = 0
            for _, v in units_dict.items():
                units += v
        print(units)
        self.embedding_layer = EmbeddingLayer(feature_map, embedding_dim)
        self.weights = nn.Parameter(torch.empty(units, num_fields).uniform_()) 

        self.ppn_batch_norm = nn.BatchNorm1d(embedding_dim*units) if batch_norm else None
        
        self.linear_layer = nn.Linear(units*embedding_dim, 1)

        self.dnn = MLP_Layer(input_dim=num_fields*embedding_dim,
                            output_dim=1, 
                            hidden_units=dnn_hidden_units,
                            hidden_activations=dnn_hidden_activations,
                            output_activation=None, 
                            dropout_rates=dnn_dropout, 
                            batch_norm=batch_norm, 
                            use_bias=True) if dnn_hidden_units else None

        self.output_layer = nn.Linear(2, 1)
        self.output_activation = self.get_output_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.reset_parameters()
        self.model_to_device()

        
    def forward(self, inputs):

        X, y = self.inputs_to_device(inputs)
        
        feature_emb = self.embedding_layer(X)

        ppn_out = self.gumbel_product_layer(feature_emb)
        ppn_out = self.linear_layer(self.ppn_batch_norm(ppn_out))

        dnn_out = self.dnn(feature_emb.flatten(start_dim=1))
        y_pred = self.output_layer(torch.cat([ppn_out, dnn_out], dim=-1))

        y_pred = self.output_activation(y_pred)
        
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def gumbel_product_layer(self, feature_emb):
        
        weights = F.softmax(self.weights, dim=-1)
        mu = torch.empty_like(weights).uniform_().cuda()
        gumbels = (torch.log(weights) - 0.1*torch.log(-torch.log(mu))) / self.tau
        order_soft = F.softmax(gumbels, dim=-1)
        order_hard = torch.zeros_like(weights)
           
        if self.strategy == 'threshold':  
            order_hard = torch.where(order_soft>self.threshold, 1, 0) 
        elif self.strategy == 'top-k': 
            order_list = []
            for k, v in self.units_dict.items():
                for _ in range(v):
                    order_list.append(k)

            for n in range(len(order_list)):
                _, idx = torch.topk(order_soft[n], order_list[n])
                order_hard[n, idx] = 1
        elif self.strategy == 'hybrid': 
            prob, idx = torch.topk(order_soft, self.top_k)

            for i in range(prob.shape[0]):
                for j in range(prob.shape[1]):
                    if order_soft[i, idx[i, j]] > self.threshold:
                        order_hard[i, idx[i, j]] = 1

        self.order = order_hard.detach()
        order_hard = (order_hard - order_soft).detach() + order_soft    
        result = torch.einsum('bmd,nm->bnmd', feature_emb, order_hard)
        result = torch.where(result.eq(0.), torch.ones_like(result), result) 
        result = (torch.prod(result, 2)).flatten(start_dim=1)  # b, N, m, D -> b, N, D -> b, N*D
        
        return result