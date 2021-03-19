import torch
import torch.nn as nn
from SparceMax import Sparsemax



class DenseFeatureLayer(nn.Module):

    def __init__(self, input_size, nrof_cat, emb_dim,
                 emb_columns, numeric_columns):
        super(DenseFeatureLayer, self).__init__()
        self.emb_dim = emb_dim
        self.emb_columns = emb_columns
        self.numeric_columns = numeric_columns

        self.embeddings = nn.ModuleDict()
        for i, col in enumerate(self.emb_columns):
            self.embeddings[col] = torch.nn.Embedding(nrof_cat[col], emb_dim)

        self.feature_bn = torch.nn.BatchNorm1d(input_size)


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
        numeric_feats = torch.tensor([])
        for key, val in input_data.items():
            if key in self.numeric_columns:
                numeric_feats = torch.cat((numeric_feats, val), dim=1)

        emb_output = None
        for i, col in enumerate(self.emb_columns):
            if emb_output is None:
                emb_output = self.embeddings[col](torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))
            else:
                emb_output = torch.cat(
                    [emb_output,
                     self.embeddings[col](torch.tensor(input_data[self.emb_columns[i]], dtype=torch.int64))],
                    dim=1)

        emb_output = torch.reshape(emb_output, [-1, len(self.emb_columns) * self.emb_dim])
        concat_input = torch.cat([numeric_feats, emb_output], dim=1)
        output = self.feature_bn(concat_input)

        return output

class GLULayer(nn.Module):

    def __init__(self, input_size, output_size):
        super(GLULayer, self).__init__()

        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc_bn = torch.nn.BatchNorm1d(output_size)
        self.outputsize = output_size


    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data):
        output = self.fc(input_data)
        output = self.fc_bn(output)
        output = torch.nn.functional.glu(output)

        return output


class FeatureTransformer(nn.Module):

    def __init__(self, shared_size, input_size, output_size, nrof_glu=2, is_shared=False):
        super(FeatureTransformer, self).__init__()

        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.is_shared = is_shared
        self.nrof_glu = nrof_glu
        self.outputsize = output_size
        self.glu_layers = nn.ModuleList()
        if is_shared:
            self.glu_layers.append(GLULayer(shared_size, self.outputsize))
            for i in range(nrof_glu-1):
                self.glu_layers.append(GLULayer(input_size, self.outputsize))
        else:
            for i in range(nrof_glu):
                self.glu_layers.append(GLULayer(input_size, self.outputsize))

    def forward(self, input_data):
        layer_input_data = input_data
        for i in range(self.nrof_glu):

            if i == 0 and self.is_shared:
                layer_input_data = self.glu_layers[0](layer_input_data)
            else:
                layer_input_data = torch.add(layer_input_data, self.glu_layers[i](layer_input_data))
                layer_input_data = torch.mul(layer_input_data, self.scale)

        return layer_input_data


class AttentiveTransformer(nn.Module):

    def __init__(self, input_size, output_size):
        super(AttentiveTransformer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        #self.fc.apply(self.init_weights)
        self.bn = nn.BatchNorm1d(output_size)
        self.Sparsemax = Sparsemax()

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def forward(self, input_data, prior_scales):
        output = self.fc(input_data)
        output = self.bn(output)
        output = torch.mul(output, prior_scales)
        output = self.Sparsemax(output)

        return output


class TabNet(nn.Module):
    def __init__(self, input_size, relax_factor, nrof_cat, emb_columns, numeric_columns,
                 nrof_glu=2, nrof_steps=6, emb_dim=5, nrof_targets=1, nd_dim=64, na_dim=64):
        super(TabNet, self).__init__()
        self.input_size = input_size
        self.nrof_steps = nrof_steps
        self.nd_dim = nd_dim
        self.na_dim = na_dim
        self.nrof_glu = nrof_glu
        self.relax_factor = relax_factor

        self.dl = DenseFeatureLayer(input_size, nrof_cat, emb_dim,
                 emb_columns, numeric_columns)

        self.bn = nn.BatchNorm1d(nd_dim)

        self.shared_feature_trasformer = FeatureTransformer(input_size, na_dim+nd_dim, (na_dim+nd_dim)*2, nrof_glu, is_shared=True)
        self.ft = FeatureTransformer(input_size, nd_dim+na_dim, (nd_dim+na_dim)*2, nrof_glu)

        self.steps_params = nn.ModuleList()
        self.final_linear = nn.Linear(nd_dim, nrof_targets)

        for i in range(nrof_steps):
            local_list = nn.ModuleList()
            local_list.append(AttentiveTransformer(self.na_dim, input_size))
            local_list.append(FeatureTransformer(input_size, nd_dim+na_dim, (nd_dim+na_dim)*2, nrof_glu))
            local_list.append(nn.ReLU())
            self.steps_params.append(local_list)



    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            # m.bias.data.fill_(0.001)

    def split(self, input):
        return input[: ,:self.nd_dim], input[:, self.na_dim:]


    def forward(self, input_data):

        origin_features = self.dl(input_data)
        prior_scales = torch.ones((origin_features.shape[0], self.input_size), dtype=torch.float32)
        final_output = torch.zeros((origin_features.shape[0], self.nd_dim), dtype=torch.float32)

        output = self.shared_feature_trasformer(origin_features)
        output = self.ft(output)
        _, att_output = self.split(output)



        for i in range(self.nrof_steps):
            att_tr, feat_tr, relu = self.steps_params[i]
            att_output = att_tr(att_output, prior_scales)
            prior_scales = torch.mul(prior_scales, (torch.sub(self.relax_factor, att_output)))
            mask = torch.mul(att_output, origin_features)
            output = self.shared_feature_trasformer(mask)
            output = feat_tr(output)
            final_output_l, att_output = self.split(output)
            #final_output_l = self.bn(final_output_l)
            final_output_l = relu(final_output_l)
            final_output = torch.add(final_output_l, final_output)

        final_output = self.final_linear(final_output)
        return final_output