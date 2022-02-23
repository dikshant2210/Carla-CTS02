"""
Author: Dikshant Gupta
Time: 22.01.22 10:57
"""
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from ped_path_predictor.model import M2P3
from ped_path_predictor.utils import *

train_annotations = 'data/training_annotations/'

observed_frame_num = 15
predicting_frame_num = 20
batch_size = 512
train_samples = 1
test_samples = 1
epochs = 1000
latent_dim = 24


class PathPredictor:
    def __init__(self, model_path=None):
        self.model = M2P3(predict_frames=predicting_frame_num).cuda()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.optim = torch.optim.Adam(lr=0.0001, params=self.model.parameters())
        self.writer = SummaryWriter(log_dir="_out/m2p3/")

    def train(self):
        # Get training data (past and future pedestrian bounding boxes)
        obs_train, pred_train, train_paths = get_raw_data(train_annotations, observed_frame_num, predicting_frame_num)
        input_train = np.array(obs_train[0][:, :, 2:], dtype=np.float32)
        output_train = np.array(pred_train[0][:, :, 2:], dtype=np.float32)
        input_train, input_test, output_train, output_test = train_test_split(input_train, output_train, test_size=0.15)

        # make output relative to the last observed frame
        i_t = input_train[:, observed_frame_num - 1, :]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, predicting_frame_num, axis=1)
        output_train = output_train - i_t

        i_t = input_test[:, observed_frame_num - 1, :]
        i_t = np.expand_dims(i_t, axis=1)
        i_t = np.repeat(i_t, predicting_frame_num, axis=1)
        output_test = output_test - i_t

        input_train = np.transpose(input_train, (1, 0, 2))
        output_train = np.transpose(output_train, (1, 0, 2))
        input_test = np.transpose(input_test, (1, 0, 2))
        output_test = np.transpose(output_test, (1, 0, 2))
        print("Input train shape=", input_train.shape)
        print("Output train shape=", output_train.shape)

        count = 0
        best_eval = np.Inf

        for epoch in range(epochs):
            num_batches = int(np.floor(input_train.shape[1] / batch_size))
            ckp_loss = 0
            for i in range(num_batches):
                x = input_train[:, i * batch_size: i * batch_size + batch_size, :]
                y = output_train[:, i * batch_size: i * batch_size + batch_size, :]
                x = torch.from_numpy(x).cuda()
                y = torch.from_numpy(y).cuda()

                y_pred, mu, log_var = self.model([x, y])
                recons_loss = F.mse_loss(y_pred, y)
                kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2), dim=1)

                loss = kld_loss + recons_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                ckp_loss += loss.item()
                self.writer.add_scalar("loss", loss.item(), epoch * num_batches + i)
                if i % 10 == 0:
                    print("Epoch: {}, batch: {} Loss: {:.4f}".format(epoch + 1, i, ckp_loss / 10))
                    ckp_loss = 0

                    eval_loss = 0
                    test_batches = int(np.floor(input_test.shape[1] / batch_size))
                    for j in range(test_batches):
                        x = input_test[:, j * batch_size: j * batch_size + batch_size, :]
                        y = output_test[:, j * batch_size: j * batch_size + batch_size, :]
                        x = torch.from_numpy(x).cuda()
                        y = torch.from_numpy(y).cuda()
                        eval_loss += evaluate(self.model, x, y)
                    eval_loss /= test_batches
                    if eval_loss < best_eval:
                        torch.save(self.model.state_dict(), '_out/m2p3_{}.pth'.format(epoch * num_batches + i))
                        best_eval = eval_loss
                    self.writer.add_scalar("eval_loss", eval_loss, count)
                    count += 1

    def evaluate(self, x_test, y_test):
        with torch.no_grad():
            y_pred = self.model.inference(x_test)
            return F.mse_loss(y_pred, y_test).item()

    def get_single_prediction(self, x):
        # input of the size (observed_frame_num, 2)
        i_t = x[observed_frame_num - 1, :]
        x = np.array(x, dtype=np.float32)
        x = x.reshape((observed_frame_num, 1, 2))
        x = torch.from_numpy(x).cuda()
        with torch.no_grad():
            y = self.model.inference(x)

        # y shape = (predicted_frame_num, 1, 2)
        y = y.squeeze().cpu().numpy()
        i_t = np.expand_dims(i_t, axis=0)
        i_t = np.repeat(i_t, predicting_frame_num, axis=0)
        y = y + i_t
        return y


def evaluate(model, x_test, y_test):
    with torch.no_grad():
        y_pred = model.inference(x_test)
        return F.mse_loss(y_pred, y_test).item()


def train():
    # Get training data (past and future pedestrian bounding boxes)
    obs_train, pred_train, train_paths = get_raw_data(train_annotations, observed_frame_num, predicting_frame_num)
    input_train = np.array(obs_train[0][:, :, 2:], dtype=np.float32)
    output_train = np.array(pred_train[0][:, :, 2:], dtype=np.float32)
    input_train, input_test, output_train, output_test = train_test_split(input_train, output_train, test_size=0.15)

    # make output relative to the last observed frame
    i_t = input_train[:, observed_frame_num - 1, :]
    i_t = np.expand_dims(i_t, axis=1)
    i_t = np.repeat(i_t, predicting_frame_num, axis=1)
    output_train = output_train - i_t

    i_t = input_test[:, observed_frame_num - 1, :]
    i_t = np.expand_dims(i_t, axis=1)
    i_t = np.repeat(i_t, predicting_frame_num, axis=1)
    output_test = output_test - i_t

    input_train = np.transpose(input_train, (1, 0, 2))
    output_train = np.transpose(output_train, (1, 0, 2))
    input_test = np.transpose(input_test, (1, 0, 2))
    output_test = np.transpose(output_test, (1, 0, 2))
    print("Input train shape=", input_train.shape)
    print("Output train shape=", output_train.shape)

    model = M2P3().cuda()
    optim = torch.optim.Adam(lr=0.0001, params=model.parameters())
    writer = SummaryWriter(log_dir="_out/m2p3/")
    count = 0
    best_eval = np.Inf

    for epoch in range(epochs):
        num_batches = int(np.floor(input_train.shape[1] / batch_size))
        ckp_loss = 0
        for i in range(num_batches):
            x = input_train[:, i*batch_size: i*batch_size + batch_size, :]
            y = output_train[:, i*batch_size: i*batch_size + batch_size, :]
            x = torch.from_numpy(x).cuda()
            y = torch.from_numpy(y).cuda()

            y_pred, mu, log_var = model([x, y])
            recons_loss = F.mse_loss(y_pred, y)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=2), dim=1)

            loss = kld_loss + recons_loss

            optim.zero_grad()
            loss.backward()
            optim.step()
            ckp_loss += loss.item()
            writer.add_scalar("loss", loss.item(), epoch * num_batches + i)
            if i % 10 == 0:
                print("Epoch: {}, batch: {} Loss: {:.4f}".format(epoch + 1, i, ckp_loss / 10))
                ckp_loss = 0

                eval_loss = 0
                test_batches = int(np.floor(input_test.shape[1] / batch_size))
                for j in range(test_batches):
                    x = input_test[:, j * batch_size: j * batch_size + batch_size, :]
                    y = output_test[:, j * batch_size: j * batch_size + batch_size, :]
                    x = torch.from_numpy(x).cuda()
                    y = torch.from_numpy(y).cuda()
                    eval_loss += evaluate(model, x, y)
                eval_loss /= test_batches
                if eval_loss < best_eval:
                    torch.save(model.state_dict(), '_out/m2p3_{}.pth'.format(epoch * num_batches + i))
                    best_eval = eval_loss
                writer.add_scalar("eval_loss", eval_loss, count)
                count += 1


if __name__ == "__main__":
    array = np.array([[3.9918034076690674, 233.06858825683594],
                      [3.9065144062042236, 233.1765899658203],
                      [3.7870030403137207, 233.24139404296875],
                      [3.6476268768310547, 233.28028869628906],
                      [3.4928321838378906, 233.3036346435547],
                      [3.3240487575531006, 233.317626953125],
                      [3.1442394256591797, 233.32591247558594],
                      [2.964298963546753, 233.33053588867188],
                      [2.7843174934387207, 233.33311462402344],
                      [2.604323387145996, 233.33456420898438],
                      [2.4243252277374268, 233.3353729248047],
                      [2.244325876235962, 233.3358154296875],
                      [2.064326047897339, 233.33607482910156],
                      [1.8843259811401367, 233.33621215820312],
                      [1.7043260335922241, 233.33628845214844]], dtype=np.float32)
    p = PathPredictor("ped_path_predictor/_out/m2p3_70797.pth")
    time_taken = 0
    runs = 1
    for _ in range(runs):
        t0 = time.time()
        res = p.get_single_prediction(array)
        time_taken += (time.time() - t0) * 1000
    print("Time taken: {:.4f}ms".format(time_taken / runs))
    for node in res:
        print(node)
        print(round(node[0]), round(node[1]))
