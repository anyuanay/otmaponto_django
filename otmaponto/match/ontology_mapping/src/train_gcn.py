import argparse
import os
import torch
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GAE, VGAE
from torch_geometric.utils import train_test_split_edges

from models.Anatomy import Anatomy
from models.GCNEncoder import GCNEncoder, LinearEncoder, VariationalGCNEncoder, VariationalLinearEncoder


def acc(pred, label, mask):
    correct = int(pred[mask].eq(label[mask]).sum().item())
    acc = correct / int(mask.sum())
    return acc


def train(_debug, _epoch):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if args.variational:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    if _debug:
        print(loss)
    if _epoch % 1000 == 0:
        print(loss)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def featurize(_gd, _model, _name):
    dataset = Anatomy(root=_gd, name=_name)
    data = dataset[0]
    _model.eval()
    _node_feat = _model.encode(data.x.to(device), data.edge_index.to(device)).detach().cpu()
    return _node_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="train_gcn",
                                     description="Trains GCN for node embeddings")
    parser.add_argument('-n', '--name', required=True, type=str,
                        help='The dataset name', dest='name')
    parser.add_argument('--variational', action='store_true')
    parser.add_argument('--linear', action='store_true')
    args = parser.parse_args()
    _name = args.name
    _wd = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    _gd = os.path.join(_wd, 'src', 'output', 'gcn', 'anatomy_single')
    _od = os.path.join(_gd, _name, '{}_features.pt'.format(_name))
    dataset = Anatomy(root=_gd, name=_name)
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data, val_ratio=.2, test_ratio=.2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_channels = 300
    num_features = dataset.num_features
    if not args.variational:
        if not args.linear:
            model = GAE(GCNEncoder(num_features, out_channels))
        else:
            model = GAE(LinearEncoder(num_features, out_channels))
    else:
        if args.linear:
            model = VGAE(VariationalLinearEncoder(num_features, out_channels))
        else:
            model = VGAE(VariationalGCNEncoder(num_features, out_channels))
    model.to(device)

    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(5000):
        loss = train(False, epoch)
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        if epoch % 10 == 0:
            print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    feat = featurize(_gd, model, _name)
    print(feat.shape)
    torch.save(feat, _od)




