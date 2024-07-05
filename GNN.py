import pickle as pkl
import matplotlib.pyplot as plt
import optuna

import torch
import torch_geometric
from torch_geometric.loader import DataLoader
from torch.optim import Adam, SGD

from models.models_together import CGNN, GAT, Transformer

import utilities
from dataset import Dataset

class GNN():
    def __init__(self, dataset_root, modelname, num_hidden_layers, num_hidden_channels, num_heads,
                lr=0.1, weight_decay=5e-4, batchsz=128, max_epoch=100):
        torch_geometric.seed_everything(4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dataset = self.loadDataset(dataset_root)
        self.loaders = self.splitTrainValTest(dataset=dataset, batchsz=batchsz)

        data = dataset[0]
        num_node_features = data.num_node_features
        num_edge_features = data.num_edge_features
        if hasattr(data, 'topo_features'):
            num_topo_features = data.topo_features.shape[1]
        else:
            num_topo_features = 0

        if modelname == 'CGNN':
            self.model = eval(modelname)(num_hidden_layers=num_hidden_layers, num_hidden_channels=num_hidden_channels,
                                         num_node_features=num_node_features,
                                         num_edge_features=num_edge_features,
                                         num_topo_features=num_topo_features,
                                         device=self.device)
            self.name = f'{modelname}_{num_hidden_layers}_{num_hidden_channels}_{lr}_{weight_decay}_{batchsz}'
        elif modelname == 'GAT' or modelname == 'Transformer':
            assert num_hidden_channels % num_heads == 0, f'{modelname} network must have hidden channels ' \
                                                         f'{num_hidden_channels} as multiples of head numbers {num_heads}'
            self.model = eval(modelname)(num_hidden_layers=num_hidden_layers, num_hidden_channels=num_hidden_channels,
                                         num_heads=num_heads,
                                         num_node_features=num_node_features,
                                         num_edge_features=num_edge_features,
                                         num_topo_features=num_topo_features,
                                         device=self.device)
            self.name = f'{modelname}_{num_hidden_layers}_{num_hidden_channels}_{num_heads}_{lr}_{weight_decay}_{batchsz}'
        self.model.to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.max_epoch = max_epoch
        self.logger = utilities.get_logger(f'./logfiles/{self.name}')

        self.train_loss = []
        self.val_loss = []
        self.min_val_loss = float('inf')
        self.early_schedule_step = 0
        self.test_loss = float('inf')

    def loadDataset(self, root):
        dataset = Dataset(root)
        dataset = dataset.shuffle()
        return dataset

    def splitTrainValTest(self, dataset, batchsz, train_ratio=0.8, val_ratio=0.1):
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        train_loader = DataLoader(dataset[: train_size], batch_size=batchsz)
        val_loader = DataLoader(dataset[train_size : train_size + val_size], batch_size=batchsz)
        test_loader = DataLoader(dataset[train_size + val_size :], batch_size=1)
        return [train_loader, val_loader, test_loader]

    def lossFunction(self, out, y):
        # return torch.sum(torch.square(out - y))
        return torch.sum(torch.abs(out - y))

    def saveGNNResults(self):
        results = {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'test_loss': self.test_loss
        }
        with open('./save/' + self.name + '.pkl', 'wb') as f:
            pkl.dump(results, f)

    def saveTrainedModel(self):
        with open('./save/' + self.name + '_model.pkl', 'wb') as f:
            pkl.dump(self.model, f)

    def loadTrainedModel(self):
        with open('./save/' + self.name + '_model.pkl', 'rb') as f:
            self.model = pkl.load(f)

    def plotLossValues(self):
        epoch = range(len(self.train_loss))

        fig = plt.figure()
        plt.plot(epoch, self.train_loss, 'k', label='train')
        plt.plot(epoch, self.val_loss, 'r', label='validation')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('./save/' + self.name + '.png')
        plt.close()

    def trainModel(self, loader):
        self.model.train()
        train_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(batch)
            loss = self.lossFunction(out, batch.y)
            loss.backward()
            # clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()
            train_loss += loss.detach().cpu().numpy()
        train_loss = train_loss / len(loader.dataset)
        return train_loss

    def evalModel(self, loader):
        self.model.eval()
        eval_loss = 0
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            loss = self.lossFunction(out, batch.y)
            eval_loss += loss.detach().cpu().numpy()
        eval_loss = eval_loss / len(loader.dataset)
        return eval_loss

    def run(self):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', min_lr=1e-5)
        train_loader, val_loader, test_loader = self.loaders[0], self.loaders[1], self.loaders[2]

        # self.logger.info('')
        for epoch in range(self.max_epoch):
            train_loss = self.trainModel(train_loader)
            val_loss = self.evalModel(val_loader)
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)
            self.logger.info(f'epoch = {epoch}, training loss = {train_loss}, validation loss = {val_loss}')
            scheduler.step(train_loss)

            if val_loss < self.min_val_loss:
                self.min_val_loss = val_loss
                self.early_schedule_step = 0
                self.saveTrainedModel()
            else:
                self.early_schedule_step += 1
                self.logger.info(f'Early stopping step {self.early_schedule_step}, the current validation loss {val_loss}'
                                 f' is larger than best value {self.min_val_loss}')

            # if self.early_schedule_step == 8:
            #     self.logger.info('Early stopped at epoch {}'.format(epoch))
            #     break

        self.loadTrainedModel()
        self.test_loss = self.evalModel(test_loader)
        self.logger.info('=' * 100)
        self.logger.info(f'The testing loss is {self.test_loss}')
        self.saveGNNResults()
        # self.plotLossValues()

class GNN_optuna():
    def __init__(self, dataset_root, num_trials=100):
        self.dataset_root = dataset_root
        self.num_trials = num_trials

    def save(self, study):
        with open('./save/GNN_optuna.pkl', 'wb') as f:
            pkl.dump(study, f)

    def objective(self, trial):
        modelname = trial.suggest_categorical('modelname', ['CGNN', 'GAT', 'Transformer'])
        num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 8)
        num_hidden_channels = trial.suggest_categorical('num_hidden_channels', [8, 16, 32, 64, 128, 256])
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4, 8])
        lr = trial.suggest_float('lr', 1e-3, 1e-1)
        weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-1)
        batchsz = trial.suggest_int('batchsz', 2, 48)

        GNN_trial = GNN(dataset_root=self.dataset_root, modelname=modelname, num_hidden_layers=num_hidden_layers, num_hidden_channels=num_hidden_channels,
            num_heads=num_heads, lr=lr, weight_decay=weight_decay, batchsz=batchsz, max_epoch=100)
        GNN_trial.run()
        return GNN_trial.test_loss

    def run(self):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=self.num_trials)

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items(): print("{}: {}".format(key, value))

        self.save(study)


if __name__ == '__main__':
    torch_geometric.seed_everything(4)
    dataset_root = 'datasets/'

    # GNN = GNN(dataset_root=dataset_root,
    #           modelname='CGNN',
    #           num_hidden_layers=1,
    #           num_hidden_channels=8,
    #           num_heads=1,
    #           lr=0.03630155814180275,
    #           weight_decay=0.08585870708132459,
    #           batchsz=3,
    #           max_epoch=100,
    #           )
    # GNN.run()

    GNN_optuna = GNN_optuna(dataset_root=dataset_root, num_trials=200)
    GNN_optuna.run()







