import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt



__add__  = ['model_traing_and_validation_loop', 'learning_algorithm']

# This helps to migrate our train to either the cpu or gpu if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#  This help to calculate the log of softmax of the final output
criteria = nn.CrossEntropyLoss()

def learning_algorithm(model, learning_rate=0.001):
    r"""
        This function returns the adam optimizer
        Args:
            model: This is the nane of our custom built algorithm
            learning_rate: This specicies the learning rate
        Example:
            optimizer = learning_algorithm(model, learning_rate=0.001)
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)
 

def model_traing_and_validation_loop(Model, n_epochs, optimizer, train_data_loader, validation_data_loader, save_path='model.pt'):
    r"""
        This returns the train model
        Args:
            Model: This is the name of our custom built network
            n_epoch: This the number f times we want to train for
            optimizer: This is the optimizer used for retain the model till it converge at it's local minima
            train_data_loader: This is our train data 
            validation_data_loader: This our validation 
            save_path: This is the path path we want to save our trained model weight
        Example:
            trained_model = model_traing_and_validation_loop(Model, 20, train_data_loader, validation_data_loader, save_path='model.pt')
    """
    
    n_epochs = n_epochs
    saving_criteria_of_model = 0
    training_loss_array = []
    validation_loss_array = []
    Model = Model.to(device)

    for i in range(n_epochs):
        total_test_data = 0
        total_train_data = 0
        correct_test_data = 0
        training_loss = 0
        validation_loss = 0

        for data, target in train_data_loader:

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            logit = Model(data)
            loss = criteria(logit, target)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()*data.size(0)

        with torch.no_grad():

            for data, target in validation_data_loader:
                data, target = data.to(device), target.to(device)
                logit = Model(data)
                _, prediction = torch.max(logit, 1) 
                loss = criteria(logit, target)
                total_test_data += target.size(0)
                correct_test_data += (prediction == target).sum().item()
                validation_loss += loss.item()*data.size(0)
                
        training_loss = training_loss / len(train_data_loader.dataset)
        validation_loss = validation_loss / total_test_data
        training_loss_array.append(training_loss)
        validation_loss_array.append(validation_loss)
        validation_accuracy = correct_test_data / total_test_data

        print(f'{i+1} / {n_epochs} Training loss: {training_loss}, Validation_loss: {validation_loss}, Validation_Accuracy: {validation_accuracy}')

        if saving_criteria_of_model < validation_accuracy:
            torch.save(Model, save_path)
            saving_criteria_of_model = validation_accuracy      
            print('--------------------------Saving Model---------------------------')
         
        
    plt.figure(figsize=(20, 4))     
    x_axis = (range(n_epochs))  
    plt.plot(x_axis, training_loss_array, 'r', validation_loss_array, 'b')  
    plt.title('A gragh of training loss vs validation loss')   
    plt.legend(['train loss', 'validation loss'])  
    plt.xlabel('Number of Epochs')  
    plt.ylabel('Loss')
        
    return Model
