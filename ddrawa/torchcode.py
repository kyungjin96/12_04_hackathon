import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import pickle

def learn():
    dataset = datasets.ImageFolder(
        '/content/drive/MyDrive/JJJ/datata',
        transforms.Compose([
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    )
    # print(dataset)
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 40 ,40 ])
    # print(len(train_dataset))
    # print('-'*30)
    # print(len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    model = models.alexnet(pretrained=True)

    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)
    device = torch.device('cuda')
    model = model.to(device)

    NUM_EPOCHS = 10
    BEST_MODEL_PATH = 'present.pth'
    best_accuracy = 0.0

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(NUM_EPOCHS):
        
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
        
        test_error_count = 0.0
        for images, labels in iter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
        
        test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
        print('%d: %f' % (epoch, test_accuracy))
        if test_accuracy > best_accuracy:
            sd = model.state_dict()
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            # torch.save(model.state_dict(), BEST_MODEL_PATH, _use_new_zipfile_serialization=False)
            best_accuracy = test_accuracy
if __name__ == '__main__':

    learn()
