pass

class TestMnist(pl.LightningModule):

    def __init__(self, batch_size, loss_type='distance', relu_slope = 0., constant_split_val=False):
        super(TestMnist, self).__init__()
        # mnist images are (1, 28, 28) (channels, width, height) 
        self.batch_size = batch_size
        self.num_classes = 10
        self.activation = nn.LeakyReLU(negative_slope=relu_slope)
        self.loss_type = loss_type
        self.constant_split_val = constant_split_val

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            self.activation,
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        self.activation(out),

        out = self.fc2(out)
        out = torch.softmax(out, dim=1)
        return out

    def full_loss(self, logits, prefix='train'):
        s = logits.shape[0]
        logits, transformed_logits = logits[:s//2], logits[s//2:]
        mean_logits = torch.mean(logits, axis=0)

        if self.loss_type == 'distance':
            loss1 = torch.linalg.norm(mean_logits-torch.ones_like(mean_logits)/self.num_classes)
            loss2 = torch.mean(torch.linalg.norm(logits-torch.ones_like(logits)/self.num_classes, dim=1))
        
        if self.loss_type == 'entropy':
            loss1 = self.entropy_loss(mean_logits)
            loss2 = torch.mean(self.entropy_loss(logits))

        loss3 = torch.mean(torch.linalg.norm(logits - transformed_logits, dim=1))

        #TODO: experiment with coefficients 
        loss = 1.*loss1 - 1.*loss2 + 1.*loss3
        return {prefix + '_loss' : loss, prefix + '_loss1' : loss1, prefix + '_loss2' : loss2, prefix + '_loss3' : loss3}

    def entropy_loss(self, logits):
        ent = logits * torch.log(logits+0.0001)
        return ent.sum(axis=-1)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        transformed = train_transform(x)
        x = torch.cat([x, transformed])
        logits = self.forward(x)
        loss_dict = self.full_loss(logits, 'train')
        self.log_dict(loss_dict, prog_bar=True)
        loss = loss_dict['train_loss']
        return loss

    def get_predictions(self, batch):
        '''
        Returns arrays (start probabilities, end probabilities) on given batch 
        '''    
        x, _ = batch
        with torch.no_grad():
            logits = self.forward(x)
        return logits

    def full_eval(self, eval_batch_size = 1000, mode='val'):
        old_bs = self.batch_size
        self.batch_size = eval_batch_size
        if mode == 'val':
            dataloader = self.val_dataloader()
        elif mode == 'train':
            dataloader = self.train_dataloader()
        else:
            print('Wrong mode selected')
            return None
        logits_list = []
        labels_list = []
        for batch_ndx, batch in enumerate(dataloader):
            _, labels = batch
            batch_logits = self.get_predictions(batch)
            logits_list.append(batch_logits)
            labels_list.append(labels)
        logits = torch.cat(logits_list)
        true_labels = torch.cat(labels_list)
        _, predicted_labels = torch.max(logits, dim=1)
        logits, true_labels, predicted_labels = numpify_list([logits, true_labels, predicted_labels])
        self.batch_size = old_bs
        return {'logits' : logits, 'true_labels' : true_labels, 'predicted_labels' : predicted_labels}
    
    def print_n(self, n, x):
        if self.counter < n:
            print('out = ', x)
            self.counter += 1

    def validation_step(self, batch, batch_idx):
        x, y = batch
        transformed = train_transform(x)
        x = torch.cat([x, transformed])
        logits = self.forward(x)
        loss_dict = self.full_loss(logits, 'val')

        _, predicted_labels = torch.max(logits[:logits.shape[0]//2], dim=1)

        d = {'val_loss' : loss_dict['val_loss'], 'true_labels' : y, 'predicted_labels' : predicted_labels, 'logits' : logits[:logits.shape[0]//2]}
        #self.log('val_loss', loss_dict['val_loss'], prog_bar=True)
        return d

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        true_labels = torch.cat([x['true_labels'] for x in outputs])
        predicted_labels = torch.cat([x['predicted_labels'] for x in outputs])
        logits = torch.cat([x['logits'] for x in outputs])
        acc = get_cluster_acc(true_labels.cpu().detach().numpy(), predicted_labels.cpu().detach().numpy())
        val_results.append({'true_labels' : true_labels, 'predicted_labels' : predicted_labels, 'logits' : logits, 'acc' : acc})
        #print(val_results)
        #print(Counter(true_labels.cpu().detach().numpy()))
        d = {'val_loss' : avg_loss, 'val_acc' : acc}
        self.log_dict(d, prog_bar=True)

    def prepare_data(self):
        # transforms for images
        transform=transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))])
        
        # prepare transforms standard to MNIST
        mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        
        #self.mnist_train, _ = random_split(mnist_train, [60000, 0])
        #_, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if self.constant_split_val:
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000], generator=torch.Generator().manual_seed(665))
        else:
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self,mnist_test, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return optimizer

# train
def train_model(batch_size = 100, max_epochs = 30, deterministic=False):
    model = TestMnist(batch_size, constant_split_val = deterministic)
    #val_results = []
    trainer = pl.Trainer(deterministic=deterministic, gpus=1, max_epochs = max_epochs, progress_bar_refresh_rate=20, accumulate_grad_batches=1)#, gradient_clip_val = 0.1)
    trainer.fit(model)
    #print(val_results)
    return model#, val_results

def train_n_models(num_models, batch_size = 100, max_epochs = 30, use_one_seed = True, seed_number = 666):
    models = []
    #if use_one_seed:
    #    #print('Seeding')
    #    pl.seed_everything(seed_number)
    for i in range(num_models):
        model = train_model(batch_size=batch_size, max_epochs=max_epochs, deterministic=use_one_seed)
        model.eval()
        models.append(model)
        #all_val_results.append(val_result)
        #print(models)
    return models#, all_val_results
