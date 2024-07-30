import torch
import torch.nn as nn

def inv_mse(y_true, y_pred):
    mse_value = torch.sum((y_true - y_pred) ** 2)
    return -mse_value

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = torch.mean(x)
    my = torch.mean(y)
    xm, ym = x - mx, y - my
    r_num = torch.sum(xm * ym)
    r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2)) + 1e-5
    r = r_num / r_den
    r = torch.clamp(r, -1.0, 1.0)
    return torch.square(r)

class Regressor(nn.Module):
    def __init__(self, latent_dim=16):
        super(Regressor, self).__init__()
        self.fc1 = nn.Linear(1024, latent_dim * 4)
        self.fc2 = nn.Linear(latent_dim * 4, latent_dim * 2)
        self.fc3 = nn.Linear(latent_dim * 2, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class Encoder(nn.Module):
    def __init__(self, ft_bank_baseline=16):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(1, ft_bank_baseline, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(ft_bank_baseline)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(ft_bank_baseline, ft_bank_baseline * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(ft_bank_baseline * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(ft_bank_baseline * 2, ft_bank_baseline * 4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(ft_bank_baseline * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(ft_bank_baseline * 4, ft_bank_baseline * 2, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(ft_bank_baseline * 2)
        self.pool4 = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, start_dim=1)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim=1024, latent_dim=16, l2_reg=0.1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim * 4)
        self.fc2 = nn.Linear(latent_dim * 4, latent_dim * 2)
        self.fc3 = nn.Linear(latent_dim * 2, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

class GAN:
    def __init__(self, lr=0.0002, ft_bank_baseline=16, latent_dim=16):
        self.lr = lr

        self.regressor = Regressor().to(device)
        self.encoder = Encoder().to(device)
        self.classifier = Classifier().to(device)

        self.optimizer_regressor = optim.Adam(self.regressor.parameters(), lr=self.lr)
        self.optimizer_encoder = optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.optimizer_classifier = optim.Adam(self.classifier.parameters(), lr=self.lr)

    def train(self, epochs, training, testing, testing_raw, batch_size=64, fold=0):
        train_data_aug, train_dx_aug, train_age_aug, train_sex_aug = training
        test_data_aug, test_dx_aug, test_age_aug, test_sex_aug = testing
        test_data, test_dx, test_age, test_sex = testing_raw

        dc_age = np.zeros((int(epochs / 10) + 1,))
        min_dc = 0
        for epoch in range(epochs):
            # Select a random batch of images
            idx_perm = np.random.permutation(int(train_data_aug.shape[0] / 2))
            ctrl_idx = idx_perm[:batch_size]
            idx_perm = np.random.permutation(int(train_data_aug.shape[0] / 2))
            idx = idx_perm[:int(batch_size / 2)]
            idx = np.concatenate((idx, idx + int(train_data_aug.shape[0] / 2)))

            training_feature_batch = torch.tensor(train_data_aug[idx]).float().to(device)
            dx_batch = torch.tensor(train_dx_aug[idx]).float().to(device)
            age_batch = torch.tensor(train_age_aug[idx]).float().to(device)

            training_feature_ctrl_batch = torch.tensor(train_data_aug[ctrl_idx]).float().to(device)
            age_ctrl_batch = torch.tensor(train_age_aug[ctrl_idx]).float().to(device)

            # Train regressor
            encoded_feature_ctrl_batch = self.encoder(training_feature_ctrl_batch[:, :32, :, :, :])
            r_loss = F.mse_loss(self.regressor(encoded_feature_ctrl_batch), age_ctrl_batch)
            self.optimizer_regressor.zero_grad()
            r_loss.backward()
            self.optimizer_regressor.step()

            # Train distiller
            g_loss = correlation_coefficient_loss(self.regressor(encoded_feature_ctrl_batch), age_ctrl_batch)
            self.optimizer_encoder.zero_grad()
            g_loss.backward()
            self.optimizer_encoder.step()

            # Train encoder & classifier
            c_loss = F.binary_cross_entropy_with_logits(self.classifier(self.encoder(training_feature_batch[:, :32, :, :, :])), dx_batch)
            self.optimizer_encoder.zero_grad()
            self.optimizer_classifier.zero_grad()
            c_loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_classifier.step()

            # Flip & re-do everything
            training_feature_batch = torch.tensor(np.flip(training_feature_batch.cpu().numpy(), 1)).float().to(device)
            training_feature_ctrl_batch = torch.tensor(np.flip(training_feature_ctrl_batch.cpu().numpy(), 1)).float().to(device)

            encoded_feature_ctrl_batch = self.encoder(training_feature_ctrl_batch[:, :32, :, :, :])
            r_loss = F.mse_loss(self.regressor(encoded_feature_ctrl_batch), age_ctrl_batch)
            self.optimizer_regressor.zero_grad()
            r_loss.backward()
            self.optimizer_regressor.step()

            g_loss = correlation_coefficient_loss(self.regressor(encoded_feature_ctrl_batch), age_ctrl_batch)
            self.optimizer_encoder.zero_grad()
            g_loss.backward()
            self.optimizer_encoder.step()

            c_loss = F.binary_cross_entropy_with_logits(self.classifier(self.encoder(training_feature_batch[:, :32, :, :, :])), dx_batch)
            self.optimizer_encoder.zero_grad()
            self.optimizer_classifier.zero_grad()
            c_loss.backward()
            self.optimizer_encoder.step()
            self.optimizer_classifier.step()

            # Log the result
            if epoch % 10 == 0:
                test_feature = torch.tensor(test_data[:, :32, :, :, :]).float().to(device)
                encoded_feature_test = self.encoder(test_feature)
                pred = self.regressor(encoded_feature_test).cpu().detach().numpy()
                print(f"Epoch {epoch}: MSE: {mean_squared_error(test_age, pred)}, R^2: {r2_score(test_age, pred)}")

                if epoch % 50 == 0:
                    pred = np.squeeze(pred)
                    dc_age[int(epoch / 10)] = dcor.distance_correlation_sqr(pred, test_age)
                    if dc_age[int(epoch / 10)] >= min_dc:
                        min_dc = dc_age[int(epoch / 10)]
                        np.save(f'model/min_pred_{fold}.npy', pred)
                        np.save(f'model/min_test_{fold}.npy', test_age)
                        torch.save(self.encoder.state_dict(), f'model/encoder_{fold}.pth')
                        torch.save(self.regressor.state_dict(), f'model/regressor_{fold}.pth')
                        torch.save(self.classifier.state_dict(), f'model/classifier_{fold}.pth')

def train_model(epochs, training, testing, testing_raw, ft_bank_baseline=16, lr=0.0002, latent_dim=16, fold=0):
    gan = GAN(lr=lr, ft_bank_baseline=ft_bank_baseline, latent_dim=latent_dim)
    gan.train(epochs, training, testing, testing_raw, fold=fold)
