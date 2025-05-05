# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")
# When using the GPU, it is important that your model and all data are on the 
# same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    # The required pre-processing depends on the pre-trained model you choose 
    # below. 
    # See https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models
    train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory (VRAM if on GPU, RAM if on CPU)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=64,
                            shuffle=False,
                            pin_memory=True, num_workers=16) # I only use 8 cause cpu :(  edit: 16 got a gpu :)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    # more info here: https://pytorch.org/vision/stable/models.html)
    #model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model = models.resnet50(pretrained=True)
    model.fc = nn.Identity()
    model.to(device)
    embedding_size = 2048 #512 
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates. 
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            start_idx = i * train_loader.batch_size
            end_idx = min(start_idx + train_loader.batch_size, num_images)
            embeddings[start_idx:end_idx] = outputs.cpu().numpy()

    np.save('dataset/embeddings.npy', embeddings)


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]

    
    embeddings = np.load('dataset/embeddings.npy')
    
    # TODO: Normalize the embeddings     
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)     #LIKE THIS????                

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you 
# don't run out of memory (VRAM if on GPU, RAM if on CPU)
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        # Attention: If you get type errors you can modify the type of the
        # labels here
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self, embedding_dim=2048, output_dim=1, dropout_prob=0.2):
        
        super().__init__()
  
        self.fc1 = nn.Linear(embedding_dim * 3, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)  
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)  
        self.fc3 = nn.Linear(512, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_prob)  
        self.fc4 = nn.Linear(256, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc4(x))
        return x

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.to(device)

    #criterion = ContrastiveLoss()


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)





    n_epochs = 20
   

    #X shape: 64, 1536  batch size 64, 3 images of size 2048 
    for epoch in range(n_epochs):  
        
        running_loss = 0.0
        model.train()      
        for i, (X, y) in enumerate(train_loader):
            
            #batch_size = X.size(0) 
            #width = X.size(1)
            
            optimizer.zero_grad()
            outputs = model(X.to(device))  #X[:, :512]
            loss = criterion(outputs, y.unsqueeze(1).float().to(device))
            
            loss.backward()
            optimizer.step()
            
            
            running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{n_epochs}], Iteration [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            
            
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {running_loss / len(train_loader)}")
        
    return model

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad():
        for [x_batch] in loader:
            
            
            x_batch = x_batch.to(device)
               
            outputs = model(x_batch)
            outputs = outputs.cpu().numpy()
            outputs = outputs.flatten()  # Flatten the array to convert it into a 1D array
            
            
            for k in range(len(outputs)):
                if (outputs[k] >= 0.5): outputs[k] = 1
                else: outputs[k] = 0
                predictions.append(outputs[k])
        
         
            
            
            
        predictions = np.vstack(predictions)
    np.savetxt("results_1.txt", predictions, fmt='%i')
    
    
    
def val_model(model, loader):
   
    model.eval()
    predictions = []
    
    counter_corr = 0
    counter = 0
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch, y] in loader:
            
            
            x_batch= x_batch.to(device)
            
            
            
            outputs = model(x_batch)
            outputs = outputs.cpu().numpy()
            outputs = outputs.flatten() 
            
            
            # Rounding the predictions to 0 or 1
            for k in range(len(outputs)):
                
                if (outputs[k] >= 0.5): outputs[k] = 1
                else: outputs[k] = 0
                
                predictions.append(outputs[k])
                counter+=1
                
                if y[k] == outputs[k]:
                    counter_corr += 1
            

                print("acc:" + str(counter_corr/counter))
                print(str(counter_corr) + "/" + str(counter))
                
            
            
        predictions = np.vstack(predictions)
    #np.savetxt("results.txt", predictions, fmt='%i')
    


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training data
    X, y = get_data(TRAIN_TRIPLETS)
    # Create data loaders for the training data
    
    total = len(X)
    factor = 1
    
    train_loader = create_loader_from_np(X[:int(factor*total)], y[:int(factor*total)], train = True, batch_size=64)
    #val_loader = create_loader_from_np(X[int(factor*total):], y[int(factor*total):], train = True, batch_size=64)
    
    # delete the loaded training data to save memory, as the data loader copies
    del X
    del y

    # repeat for testing data
    X_test, y_test = get_data(TEST_TRIPLETS, train=False)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)
    del X_test
    del y_test

    # define a model and train it
    model = train_model(train_loader)
    
    # test the model on the test data
    #val_model(model, val_loader)
    test_model(model, test_loader)
    
    
    print("Results saved to results.txt")
