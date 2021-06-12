import torch
import os
import numpy as np
import cv2
import torchvision.transforms as T
from torch.utils.data import Dataset
from scipy import spatial

from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image
from ..utils import getFiles
from .src.resnet import Resnet50, Resnet18
from .src.colorHist import getMask, getColorHist

ALLOWED_EXTENSIONS: Tuple = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'
)

# class AllDataset(Dataset):
#     def __init__(self, images_dir: str):
#         # transform: Callable,
#         self.images_dir: str = images_dir
#         # self.transform = transform    
#         self.samples: List[str] = getFiles(images_dir, allowedExtensions=ALLOWED_EXTENSIONS)

#     def __len__(self) -> int:
#         return len(self.samples)

#     def __getitem__(self, index: int) -> str:
#         image_path = self.samples[index]
#         image = Image.open(image_path).convert("RGB")
#         # image = self.transform(image)
#         return image


class CbDescriptor:
    def __init__(self, checkpointPath: str, useColor: bool = False, colorHistSize: int = 512):
        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.checkpoint: Dict[str, Any] = torch.load(checkpointPath, map_location="cpu")
        # self.config: Dict[str, Any] = self.checkpoint["config"]
        self.batchSize: int = 30
        # if not self.batchSize:
            # self.
            # self.batchSize = self.config["classes_per_batch"] * self.config["samples_per_class"]
        self.n_cpus: int = cpu_count()
        if self.n_cpus >= self.batchSize:
            self.n_workers: int = self.batchSize
        else:
            self.n_workers: int = self.n_cpus
        
        self.model = Resnet50(128)
        # self.model = Resnet18(embedding_size=132)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.useColorHist = useColor
        self.colorHistSize = colorHistSize

        # self.referenceSet = AllDataset(imagesDir)
        # self.batchSize = batchSize
        # self.loader = DataLoader(
        #     self.referenceSet, 
            #     num_workers=self.n_workers
        # )
    
    def describe(self, imagePath: str, getImage:bool = False):
        image: Image.Image = Image.open(imagePath).convert("RGB")
        # data = np.array(image) 
        # red, green, blue = data.T
        # data = np.array([blue, green, red])
        # data = data.transpose()
        # image = Image.fromarray(data)

        # image = cv2.imread(os.path.join(imagePath))
        input_tensor: torch.Tensor = self.transform(image).unsqueeze(dim=0).to(self.device)
        
        # input_tensor = torch.zeros(1,3,448, 448).to(self.device)
        embedding: torch.Tensor = self.model(input_tensor)
        
        embedding = embedding.detach().cpu().numpy()
        embedding = np.squeeze(embedding, axis=0)

        if self.useColorHist:
            head, tail = os.path.split(imagePath)
            # segmentedPath = "algorithms/bagOfWords/segmentedData"
            segmentedPath = "data/vab_t_database_extended/segmented"
            image = cv2.imread(os.path.join(segmentedPath, tail))
            # image = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
            histColor = getColorHist(image)
            histColor /= histColor.sum()
            embedding = np.concatenate([embedding, histColor])
            
        if getImage:
            return embedding, cv2.imread(imagePath)
        return embedding
    
    def compare(self, featuresA: np.ndarray, featuresB: np.ndarray):
        if self.useColorHist:
            histColorA = featuresA[-self.colorHistSize:].astype(np.float32)
            histA = featuresA[:-self.colorHistSize]
            histColorB = featuresB[-self.colorHistSize:].astype(np.float32)
            histB = featuresB[:-self.colorHistSize]
            colorDist = self.chi2_distance_color(histColorA, histColorB)
            return spatial.distance.cosine(histA, histB) * colorDist
            return np.linalg.norm(histA - histB) * colorDist
        return spatial.distance.cosine(featuresA, featuresB)
        return np.linalg.norm(featuresA - featuresB)
    
    def chi2_distance_color(self, histA, histB, eps = 1e-10):
        """chi2 distance normalized, normalization base on the max posible distance

        Args:
            histA (np.ndarray): histA
            histB (np.ndarray): histN
            eps (float, optional): epsilon. Defaults to 1e-10.

        Returns:
            float: Distance in range [0, 1]
        """
        # compute the chi-sqaured distance
        countA = histA.sum()
        countB = histB.sum()
        maxDist = countA + countB

        d = 0.5 * np.sum(((histA - histB) ** 2) / (histA + histB + eps))

        # return the chi-sqaured distance
        return d / maxDist