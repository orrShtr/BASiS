import os
import torch


class CofigDataset(object):
    def __init__(self, X, y):
        self.size = len(y)
        self.data = X
        self.label = y

    def __getitem__(self, index):
        return (self.data[index],
                self.label[index])

    def __len__(self):
        return self.size


def save_model(model_path, current_epoch, model):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)