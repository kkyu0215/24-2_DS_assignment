import torch
import torch.nn as nn
import numpy as np
import clip

class_names = [
        "antelope",
        "badger",
        "bat",
        "bear",
        "bee",
        "beetle",
        "bison",
        "boar",
        "butterfly",
        "cat",
        "caterpillar",
        "chimpanzee",
        "cockroach",
        "cow",
        "coyote",
        "crab",
        "crow",
        "deer",
        "dog",
        "dolphin",
        "donkey",
        "dragonfly",
        "duck",
        "eagle",
        "elephant",
        "flamingo",
        "fly",
        "fox",
        "goat",
        "goldfish",
        "goose",
        "gorilla",
        "grasshopper",
        "hamster",
        "hare",
        "hedgehog",
        "hippopotamus",
        "hornbill",
        "horse",
        "hummingbird",
        "hyena",
        "jellyfish",
        "kangaroo",
        "koala",
        "ladybugs",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "mosquito",
        "moth",
        "mouse",
        "octopus",
        "okapi",
        "orangutan",
        "otter",
        "owl",
        "ox",
        "oyster",
        "panda",
        "parrot",
        "pelecaniformes",
        "penguin",
        "pig",
        "pigeon",
        "porcupine",
        "possum",
        "raccoon",
        "rat",
        "reindeer",
        "rhinoceros",
        "sandpiper",
        "seahorse",
        "seal",
        "shark",
        "sheep",
        "snake",
        "sparrow",
        "squid",
        "squirrel",
        "starfish",
        "swan",
        "tiger",
        "turkey",
        "turtle",
        "whale",
        "wolf",
        "wombat",
        "woodpecker",
        "zebra"
      ]

class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        #self.classifier = nn.Linear(512, 90)  # Assuming 90 classes, adjust accordingly
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # Learnable logit scale


    def forward(self, images, labels):

        text_labels = [self.class_names[label] for label in labels.cpu().numpy()]
        tokenized_labels = clip.tokenize(text_labels).to(images.device)

        with torch.no_grad():
            # features = self.clip_model.encode_image(images)

            # Extract image and text features
            image_features = self.clip_model.encode_image(images)
            label_features = self.clip_model.encode_text(tokenized_labels)

        # Normalize features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        label_features = label_features / label_features.norm(dim=1, keepdim=True)

        # Compute cosine similarity (logits)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ label_features.t()
        logits_per_label = logits_per_image.t()

        
        #return self.classifier(features.float())
        return logits_per_image, logits_per_label

