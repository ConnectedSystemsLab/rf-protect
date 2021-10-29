import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from config import parsers

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class Generator(nn.Module):
    ''' C-RNN-GAN generator
    '''
    def __init__(self, num_feats, hidden_units=512, drop_prob=0.5, use_cuda=False):
        super(Generator, self).__init__()
        self.args = parsers()
        # params
        self.hidden_dim = hidden_units
        self.use_cuda = use_cuda
        self.num_feats = num_feats+1

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.label_embedding = nn.Embedding(5, self.args.latent_dim)
        self.embedding_fc = nn.Linear(self.args.latent_dim, self.args.seq_len)
        self.fc_layer1 = nn.Linear(in_features=(self.num_feats*2), out_features=hidden_units)
        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.fc_layer2 = nn.Linear(in_features=hidden_units, out_features=self.num_feats)

        self.lstm = nn.LSTM(input_size=self.num_feats, hidden_size=hidden_units,
                            num_layers=2, batch_first=True, dropout=drop_prob)
        self.fc_layer = nn.Linear(in_features=hidden_units, out_features=2)

        # self.model = nn.Sequential(
        #     nn.Linear(self.args.seq_len, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 512),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(512, 1024),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(1024, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, self.args.seq_len)
        #     # nn.Linear(1024, self.args.seq_len)
        # )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, z, states, labels):
        ''' Forward prop
        '''
        if self.use_cuda:
            z = z.cuda()
        # padding = torch.zeros([16, 1]).cuda()
        # labels = torch.cat([padding, labels.view(-1, 1)], dim=1)
        a = self.label_embedding(labels).view(z.shape[0], z.shape[1], 1)
        z1 = torch.cat([a, z], dim=2).view(z.shape[0], 3, -1)
        z2 = self.embedding_fc(z1)
###
        # gen_feats = self.model(z2).view(-1,100,2)
###
        z = z2.view(-1, self.args.seq_len, 3)
        # z: (batch_size, seq_len, num_feats)
        # z here is the uniformly random vector
        lstm_out, states = self.lstm(z, states)
        gen_feats = self.fc_layer(lstm_out)  #128,50,2
        # gen_feats = torch.tanh(gen_feats)*13
        # batch_size, seq_len, num_feats = z.shape
        #
        # # split to seq_len * (batch_size * num_feats)
        # z = torch.split(z, 1, dim=1)
        # z = [z_step.squeeze(dim=1) for z_step in z]
        #
        # # create dummy-previous-output for first timestep
        # # prev_gen = torch.empty([batch_size, num_feats]).uniform_()
        # prev_gen = 2*torch.rand(batch_size, num_feats)-1
        # if self.use_cuda:
        #     prev_gen = prev_gen.cuda()
        #
        # # manually process each timestep
        # state1, state2 = states # (h1, c1), (h2, c2)
        # gen_feats = []
        # for z_step in z:
        #     # concatenate current input features and previous timestep output features
        #     concat_in = torch.cat((z_step, prev_gen), dim=-1)# 把每个z的time step都拿出来，所以z_step是32*1，跟上一步生成的粘在一起，所以32*2作为Input
        #     out = F.relu(self.fc_layer1(concat_in)) # 32*2 -> 32*256
        #     h1, c1 = self.lstm_cell1(out, state1)
        #     h1 = self.dropout(h1) # feature dropout only (no recurrent dropout)
        #     h2, c2 = self.lstm_cell2(h1, state2)
        #     # prev_gen = torch.tanh(self.fc_layer2(h2))
        #     prev_gen = self.fc_layer2(h2)
        #
        #     gen_feats.append(prev_gen)
        #
        #     state1 = (h1, c1)
        #     state2 = (h2, c2)
        #
        # # seq_len * (batch_size * num_feats) -> (batch_size * seq_len * num_feats)
        # gen_feats = torch.stack(gen_feats, dim=1)
        #
        # states = (state1, state2)
        return gen_feats, states

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data
        layer_mult = 1
        self.num_layers = 2
        if self.use_cuda:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_())

        return hidden


class Discriminator(nn.Module):
    ''' C-RNN-GAN discrminator
    '''
    def __init__(self, num_feats, hidden_units=512, drop_prob=0.5, use_cuda=False):

        super(Discriminator, self).__init__()
        self.args = parsers()
        # params
        self.hidden_dim = hidden_units
        self.num_layers = 2
        self.use_cuda = use_cuda
        num_feats = num_feats + 1
        self.label_embedding = nn.Embedding(5, self.args.latent_dim)
        self.fc1 = nn.Linear(50, self.args.seq_len)
        self.embedding_fc = nn.Linear(self.args.seq_len, self.args.seq_len)

        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm = nn.LSTM(input_size=num_feats, hidden_size=hidden_units,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=self.args.Bidirct_D)
        if self.args.Bidirct_D:
            self.fc_layer = nn.Linear(in_features=(2*hidden_units), out_features=1)
        else:
            self.fc_layer = nn.Linear(in_features=hidden_units, out_features=1)
        self.fc_layer1 = nn.Linear(256, 1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, note_seq, state, labels):
        ''' Forward prop
        '''
        if self.use_cuda:
            note_seq = note_seq.cuda()

        a = self.label_embedding(labels).view(note_seq.shape[0], note_seq.shape[1], 1)
        # a = self.fc1(a).view(note_seq.shape[0], note_seq.shape[1], -1)
        z1 = torch.cat([a, note_seq], dim=2).view(note_seq.shape[0], 3, -1)
        z2 = self.embedding_fc(z1)
        note_seq = z2.view(-1, self.args.seq_len, 3)

        # note_seq: (batch_size, seq_len, num_feats)
        # note_seq = self.dropout(note_seq) # input with dropout
        # (batch_size, seq_len, num_directions*hidden_size)
        lstm_out, state = self.lstm(note_seq, state) #用了batch first,所以batch 在最前面，32,8,512(2*256)
        # (batch_size, seq_len, 1)
        # lstm_out1 = torch.tanh(lstm_out)
        out = self.fc_layer(lstm_out)
        out = torch.sigmoid(out) #32*8*1

        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        # (batch_size)
        out = torch.mean(out, dim=reduction_dims)  #每个batch算出一个mean, out的维度跟batch一样

        return out, lstm_out, state #32,, 32*8*512,

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data
        if self.args.Bidirct_D:
            layer_mult = 2 # for being bidirectional
        else:
            layer_mult = 1
        if self.use_cuda:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_())
        
        return hidden


class classifier(nn.Module):
    def __init__(self, num_feats, hidden_units=128, drop_prob=0.5, use_cuda=False):
        super(classifier, self).__init__()
        self.args = parsers()
        # params
        self.hidden_dim = hidden_units
        self.use_cuda = use_cuda
        self.num_feats = num_feats

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.lstm = nn.LSTM(input_size=self.num_feats, hidden_size=hidden_units,
                            num_layers=2, batch_first=True, dropout=drop_prob)
        self.fc_layer = nn.Linear(in_features=hidden_units, out_features=2)

        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3)
        self.pool = nn.MaxPool1d(3, stride=2)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.fc_layer2 = nn.Linear(21, 5)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            m.apply(weight_init)

    def forward(self, z, states):
        ''' Forward prop
        '''
        if self.use_cuda:
            z = z.cuda()

        lstm_out, states = self.lstm(z, states)
        lstm_out = lstm_out.contiguous().view(z.shape[0], self.hidden_dim, -1)
        x = F.relu(self.conv1(lstm_out))
        # x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        # x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layer2(x)
        gen_feats = F.log_softmax(x, dim=1)
        return gen_feats

    def get_feature(self, z, states):
        if self.use_cuda:
            z = z.cuda()
        lstm_out, states = self.lstm(z, states)
        lstm_out = lstm_out.contiguous().view(z.shape[0], self.hidden_dim, -1)
        x = F.relu(self.conv1(lstm_out))
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        # x = self.fc_layer2(x)
        return x

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data
        layer_mult = 1
        self.num_layers = 2
        if self.use_cuda:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).zero_())

        return hidden
