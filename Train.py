from network import TabNet
from data_loader import CustomDataset
import torch.utils.data as data_utils
from torch.optim import Adam
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.metrics.functional import accuracy
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings("ignore")



TRAIN_PATH = '/home/denis/repos/sber_risk_DL/week12/TabNet/data/train_adult.pickle'
VALID_PATH = '/home/denis/repos/sber_risk_DL/week12/TabNet/data/valid_adult.pickle'
BATCH_SIZE = 1200
EPOCHS = 30

#train_writer = SummaryWriter('./logs/train')
#valid_writer = SummaryWriter('./logs/valid')

print('Run train ...')

train_dataset = CustomDataset(TRAIN_PATH)
train_loader = data_utils.DataLoader(dataset=train_dataset,
                                          batch_size=BATCH_SIZE, shuffle=True)
vall_dataset = CustomDataset(VALID_PATH)
vall_loader = data_utils.DataLoader(dataset=vall_dataset,
                                         batch_size=BATCH_SIZE, shuffle=False)

emb_dim = 5
input_size = len(train_dataset.numeric_columns) + len(train_dataset.embedding_columns) * emb_dim
tabnet = TabNet(nd_dim=64, na_dim=64,  relax_factor=1, input_size=input_size, nrof_cat=train_dataset.nrof_emb_categories,
                emb_columns=train_dataset.embedding_columns, numeric_columns=train_dataset.numeric_columns,
                emb_dim=emb_dim, nrof_steps=7, nrof_glu=2)


optimizer = Adam(tabnet.parameters(), lr=1e-3)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]))

for epoch in range(EPOCHS):
    print(epoch)

    tabnet.train()

    batch_loss_train = []
    batch_acc_train = []
    gini_score_hist = []

    for features, label in train_loader:
        # Reset gradients
        optimizer.zero_grad()

        output = tabnet(features)

        output = torch.squeeze(output)
        # Calculate error and backpropagate
        loss = loss_fn(output, label)
        output = torch.sigmoid(output)
        loss.backward()
        acc = accuracy_score(label.numpy(), output.detach().numpy() > 0.5)
        gini_score = 2 * roc_auc_score(label.numpy(), output.detach().numpy()) - 1

        # Update weights with gradients
        optimizer.step()

        gini_score_hist.append(gini_score)
        batch_loss_train.append(loss.item())
        batch_acc_train.append(acc)


    #train_writer.add_scalar('CrossEntropyLoss', np.mean(batch_loss_train), epoch)
    #train_writer.add_scalar('Accuracy', np.mean(batch_acc_train), epoch)
    #train_writer.add_scalar('Gini', np.mean(gini_score_hist), epoch)

    print(f'train_acc = {np.mean(batch_acc_train)}')
    print(f'train_gini = {np.mean(gini_score_hist)}\n')

    batch_loss_vall = []
    batch_acc_vall = []
    gini_score_vall_hist = []

    tabnet.eval()
    with torch.no_grad():
        for features, label in vall_loader:
            vall_output = tabnet(features)
            vall_output = torch.squeeze(vall_output)
            vall_loss = loss_fn(vall_output, label)
            vall_output = torch.sigmoid(vall_output)
            vall_acc = accuracy_score(label.numpy(), vall_output.detach().numpy() > 0.5)
            gini_score_vall = 2 * roc_auc_score(label.numpy(), vall_output.detach().numpy()) - 1

            gini_score_vall_hist.append(gini_score_vall)
            batch_loss_vall.append(vall_loss.item())
            batch_acc_vall.append(vall_acc)

    #valid_writer.add_scalar('CrossEntropyLoss', np.mean(batch_loss_vall), epoch)
    #valid_writer.add_scalar('Accuracy', np.mean(batch_acc_vall), epoch)
    #valid_writer.add_scalar('Gini', np.mean(gini_score_vall_hist), epoch)
    print(f'vall_acc = {np.mean(batch_acc_vall)}')
    print(f'vall_gini = {np.mean(gini_score_vall_hist)}\n')
